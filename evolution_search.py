import random
import utils
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
import json
import os
import copy
import random

from engine import evaluate
from datasets import build_dataset
from pathlib import Path
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model
from torchvision import datasets, transforms
from collections import defaultdict
import yaml
from yaml.loader import SafeLoader
import model_sparse
from sparsity_factory import get_model_sparsity, weight_pruner_loader

class RandomCandGenerator():
    def __init__(self, sparsity_config):
        self.sparsity_config               = sparsity_config
        self.num_candidates_per_block = len(sparsity_config[0]) # might have bug if each block has different number of choices
        self.config_length            = len(sparsity_config)    # e.g., the len of DeiT-S is 48 (12 blocks, each has qkv, fc1, fc2, and linear projection)
        self.m = defaultdict(list)        # m: the magic dictionary with {index: cand_config}
        #random.seed(seed)
        v = []                            # v: a temp vector for function rec()
        self.rec(v, self.m)
        
    def calc(self, v):                    # generate the unique index for each candidate
        res = 0
        for i in range(self.num_candidates_per_block):
            res += i * v[i]
        return res

    def rec(self, v, m, idx=0, cur=0):    # recursively enumerate all possible candidates and attach unique indexes for them
        if idx == (self.num_candidates_per_block-1) :
            v.append(self.config_length - cur)
            m[self.calc(v)].append(copy.copy(v))
            v.pop()
            return

        i = self.config_length - cur
        while i >= 0:
            v.append(i)
            self.rec(v, m, idx+1, cur+i)
            v.pop()
            i -= 1
            
    def random(self):                     # generate a random index and return its corresponding candidate
        row = random.choice(random.choice(self.m))
        ratios = []
        for num, ratio in zip(row, [i for i in range(self.num_candidates_per_block)]):
            ratios += [ratio] * num
        random.shuffle(ratios)
        res = []
        for idx, ratio in enumerate(ratios):
            res.append(tuple(self.sparsity_config[idx][ratio])) # Fixme: 
        return res                        # return a cand_config



class EvolutionSearcher():
    def __init__(self, args, model, model_without_ddp, sparsity_config, val_loader, output_dir, config):
        self.model = model
        self.model_without_ddp = model_without_ddp
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.parameters_limits = args.param_limits
        self.min_parameters_limits = args.min_param_limits
        self.val_loader = val_loader
        self.output_dir = output_dir
        self.s_prob =args.s_prob
        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []
        self.top_accuracies = []
        self.cand_params = []
        self.sparsity_config = config['sparsity']['choices']
        
        self.rcg = RandomCandGenerator(self.sparsity_config)

    def save_checkpoint(self):

        info = {}
        info['top_accuracies'] = self.top_accuracies
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        checkpoint_path = os.path.join(self.output_dir, "checkpoint-{}.pth.tar".format(self.epoch))
        torch.save(info, checkpoint_path)
        print('save checkpoint to', checkpoint_path)

    def is_legal(self, cand):
        assert isinstance(cand, tuple)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False

        
        self.model_without_ddp.set_sample_config(cand)
        print(cand)
        n_parameters = self.model_without_ddp.num_params() / 1e6
        info['params'] = n_parameters # sparsity level
        print(n_parameters)

        if info['params'] > self.parameters_limits:
            print('parameters limit exceed')
            return False

        if info['params'] < self.min_parameters_limits:
            print('under minimum parameters limit')
            return False

        print("rank:", utils.get_rank(), cand, info['params'])
        eval_stats = evaluate(self.val_loader, self.model, 'cuda')

        info['acc'] = eval_stats['acc1']

        info['visited'] = True

        return True

    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        print('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                print(cands)
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand

    def get_random_cand(self):
        
        cand_tuple = self.rcg.random()

        return tuple(cand_tuple)

    def get_random(self, num):
        print('random select ........')
        cand_iter = self.stack_random_cand(self.get_random_cand)
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            print('random {}/{}'.format(len(self.candidates), num))
        print('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob, s_prob):
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(random.choice(self.keep_top_k[k]))

            # sparsity ratio
            for idx in range(len(self.sparsity_config)):
                random_s = random.random()
                if random_s < m_prob:
                    cand[idx] = random.choice(self.sparsity_config[idx])
                    
            return tuple(cand)


        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('mutation {}/{}'.format(len(res), mutation_num))

        print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():

            p1 = random.choice(self.keep_top_k[k])
            p2 = random.choice(self.keep_top_k[k])
            max_iters_tmp = 50
            while len(p1) != len(p2) and max_iters_tmp > 0:
                max_iters_tmp -= 1
                p1 = random.choice(self.keep_top_k[k])
                p2 = random.choice(self.keep_top_k[k])
            return tuple(random.choice([i, j]) for i, j in zip(p1, p2))

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('crossover {}/{}'.format(len(res), crossover_num))

        print('crossover_num = {}'.format(len(res)))
        return res

    def search(self):
        print(
            'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
                self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        # self.load_checkpoint()

        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            print('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['acc'])
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['acc'])

            print('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            tmp_accuracy = []
            for i, cand in enumerate(self.keep_top_k[50]):
                print('No.{} {} Top-1 val acc = {}, params = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['acc'], self.vis_dict[cand]['params']))
                tmp_accuracy.append(self.vis_dict[cand]['acc'])
            self.top_accuracies.append(tmp_accuracy)

            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob, self.s_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1

            self.save_checkpoint()

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int)
    # data-params
    
    # evolution search parameters
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--population-num', type=int, default=50)
    parser.add_argument('--m_prob', type=float, default=0.2)
    parser.add_argument('--s_prob', type=float, default=0.4)
    parser.add_argument('--crossover-num', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--mutation-num', type=int, default=25)
    parser.add_argument('--param-limits', type=float, default=5.6)
    parser.add_argument('--min-param-limits', type=float, default=5)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Sparsity correlated arguments
    parser.add_argument('--sparsity-config', default='', type=str, help='path to the sparsity yaml file')

    return parser

def main(args):
    utils.init_distributed_mode(args)

    print(args)

    cudnn.benchmark = True

    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    dataset_val = datasets.ImageFolder(
        os.path.join(args.data_path, 'val'),
        transform=transform)
    '''

    dataset_val, _ = build_dataset(is_train=False, args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=int(2 * args.batch_size),
        sampler=sampler_val, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )

    with open(args.sparsity_config) as f:
        sparsity_config = yaml.load(f, Loader=SafeLoader)  


    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=True,
        num_classes=1000,
        drop_rate=0,
        drop_path_rate=0,
        img_size=args.input_size
    )
    
    model.cuda()
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    t = time.time()
    searcher = EvolutionSearcher(args, model, model_without_ddp, sparsity_config, val_loader=data_loader_val, output_dir = args.output_dir, config=sparsity_config)

    searcher.search()

    print('total searching time = {:.2f} hours'.format(
        (time.time() - t) / 3600))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('AutoFormer evolution search', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


# CUDA_VISIBLE_DEVICES=3 python evolution_svd.py --data-path /dev/shm/imagenet/ --output_dir BASE_EA_13_16.5 --config sparsity_config/Vit_imnet_config_base.json --model deit_dist_base_p16_224_imnet_0416_wo_fc/checkpoint.pth --param-limits 16.5 --min-param-limits 13
#python evolution_search.py --data-path /work/shadowpa0327/imagenet --output_dir deit_small_nxm_ea_124 --sparsity-config configs/deit_small_nxm_ea124.yml --model Sparse_deit_small_patch16_224 --param-limits 13.2 --min-param-limits 8