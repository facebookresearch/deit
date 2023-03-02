import torch.nn as nn
import torch.nn.functional as F
import torch

def compute_mask(t, N, M):
    out_channel, in_channel = t.shape
    percentile = N / M
    t_reshaped = t.reshape(out_channel, -1, M)
    #print(t_reshaped.shape)
    mask = torch.ones_like(t)
    mask_reshaped = mask.reshape(out_channel, -1, M)
    
    nparams_topprune = int(M * (1-percentile)) 
    if nparams_topprune != 0:
        topk = torch.topk(torch.abs(t_reshaped), k=nparams_topprune, largest=False, dim = -1)
        #print(topk.indices)
        mask_reshaped = mask_reshaped.scatter(dim = -1, index = topk.indices, value = 0)
    
    return mask_reshaped.reshape(out_channel, in_channel)

class SparseLinearSuper(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.ones(out_features))
        else:
            self.bias = None
        
        self.sparsity_config = (4, 4)
        self.mask = torch.ones_like(self.weight)
        self.set_sample_config(self.sparsity_config)

    def set_sample_config(self, sample_config):
        self.sparsity_config = sample_config
        self._set_mask()
        
    def _set_mask(self):
        n, m = self.sparsity_config
        self.mask = compute_mask(self.weight, n, m)


    def forward(self, x):
        weight = self.weight * self.mask
        #weight = self.weight
        if self.bias is not None:
            x = F.linear(x, weight, self.bias)
        else:
            x = F.linear(x, weight)

        return x
    
    def num_pruned_params(self):
        return int(torch.sum(self.mask==0).item())


if __name__ == '__main__':
    m = SparseLinearSuper(12, 12)
    input = torch.randn(12)
    print(m(input))
    m.set_sample_config((2,4))
    print(m(input))
    print(m.num_pruned_params())
    #print(sum(p.numel() for p in m.parameters() if p.requires_grad))
    