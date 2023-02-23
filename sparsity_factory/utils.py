import torch
import torch.nn as nn
from copy import deepcopy

# Preliminaries. Not to be exported.

exclude_module_name = ['head']

def _is_prunable_module(m, name = None):
    if name is not None and name in exclude_module_name:
        return False
    return isinstance(m, nn.Linear)

def _get_sparsity(tsr):
    total = tsr.numel()
    nnz = tsr.nonzero().size(0)
    return nnz/total

def _get_nnz(tsr):
    return tsr.nonzero().size(0)

# Modules

def get_weights(model):
    weights = []
    for name, m in model.named_modules():
        if _is_prunable_module(m, name):
            weights.append(m.weight)
    return weights

def get_convweights(model):
    weights = []
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            weights.append(m.weight)
    return weights

def get_modules(model):
    modules = []
    for name, m in model.named_modules():
        if _is_prunable_module(m, name):
            modules.append(m)
    return modules

def get_modules_with_name(model):
    modules = []
    for name, m in model.named_modules():
        if _is_prunable_module(m, name):
            modules.append((name, m))
    return modules


def get_convmodules(model):
    modules = []
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            modules.append(m)
    return modules

def get_copied_modules(model):
    modules = []
    for m in model.modules():
        if _is_prunable_module(m):
            modules.append(deepcopy(m).cpu())
    return modules

def get_model_sparsity(model):
    prunables = 0
    nnzs = 0
    for m in model.modules():
        if _is_prunable_module(m):
            prunables += m.weight.data.numel()
            nnzs += m.weight.data.nonzero().size(0)
    return nnzs/prunables

def get_sparsities(model):
    return [_get_sparsity(m.weight.data) for m in model.modules() if _is_prunable_module(m)]

def get_nnzs(model):
    return [_get_nnz(m.weight.data) for m in model.modules() if _is_prunable_module(m)]
