from collections import OrderedDict

import torch
import torch.nn as nn
import random
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg, HybridEmbed, PatchEmbed, Block
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from mask_const import DIVISION_MASKS_14_14


class CompVisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def split_input(self, x, M):
        precomputed_masks = DIVISION_MASKS_14_14[M]

        mask_id = random.randint(0, len(precomputed_masks) - 1)
        masks = precomputed_masks[mask_id]
        masks = [torch.tensor(mask).unsqueeze(0) for mask in masks]

        masks = [torch.cat([torch.ones(mask.shape[0], 1, dtype=bool, device=mask.device), mask.flatten(1)], dim=1) for mask in masks]

        xs = []
        for mask in masks:
            if mask.shape[0] == 1:
                xs.append(x[:, mask[0, :]].reshape(x.shape[0], -1, x.shape[-1]))
            else:
                xs.append(x[mask].reshape(x.shape[0], -1, x.shape[-1]))
        return xs

    def comp_forward_afterK(self, x, out_feat_keys, K, M):
        if out_feat_keys is None:
            out_feat_keys = []

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        xs = self.split_input(x, M)
        out_feats = [("BLK" if "BLK" in k else "CLS", int(k[len("concat___"):]) if "concat" in k else 1) for k in
                     out_feat_keys]
        n_blk_save = max([n for name, n in out_feats if name == "BLK"] + [0])
        n_cls_save = max([n for name, n in out_feats if name == "CLS"] + [0])

        after_i_blocks_save_blk = len(self.blocks) - n_blk_save + 1
        after_i_blocks_save_cls = len(self.blocks) - n_cls_save + 1
        after_i_blocks_save = min(after_i_blocks_save_blk, after_i_blocks_save_cls)
        assert (after_i_blocks_save >= K)

        # run separately
        subencoder = nn.Sequential(*self.blocks[:K])
        xs = [subencoder(x) for x in xs]

        # mixing
        xs_cls = torch.stack([x[:, [0], :] for x in xs])
        xs_feats = [x[:, 1:, :] for x in xs]
        x = torch.cat([xs_cls.mean(dim=0)] + xs_feats, dim=1)

        for blk in self.blocks[K:after_i_blocks_save]:
            x = blk(x)

        # extract
        blk_feats = []
        cls_feats = []

        if after_i_blocks_save >= after_i_blocks_save_blk:
            blk_feats.append(self.norm(x))
        if after_i_blocks_save >= after_i_blocks_save_cls:
            cls_feats.append(self.norm(x[:, 0, :]))

        for i, blk in enumerate(self.blocks[after_i_blocks_save:]):
            x = blk(x)
            if i + after_i_blocks_save >= after_i_blocks_save_blk:
                blk_feats.append(self.norm(x))
            if i + after_i_blocks_save >= after_i_blocks_save_cls:
                cls_feats.append(self.norm(x[:, 0, :]))

        if len(out_feats) > 0:
            output = [
                torch.cat(cls_feats[-n:], dim=-1) if feat == "CLS" else torch.cat(blk_feats[-n:], dim=-1)
                for feat, n in out_feats
            ]
        else:
            output = x[:, 0, :]
        output = self.pre_logits(output)
        return output

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)[:, 0]
        x = self.pre_logits(x)
        return x

    def forward(self, sample):
        x, K, M = sample
        x = self.comp_forward_afterK(x, ['lastCLS'], K, M)
        x = self.head(x)
        return x
