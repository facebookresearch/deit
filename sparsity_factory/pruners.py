import torch
from torch.nn.utils import prune
from .utils import get_weights, get_modules, get_modules_with_name
import numpy as np

ALL_PRUNERS = ['lamp', 'glob', 'unif', 'unifplus', 'erk', 'custom']
def check_valid_pruner(name):
    return name in ALL_PRUNERS

def weight_pruner_loader(pruner_string):
    """
    Gives you the pruning methods: LAMP, Glob, Unif, Unif+, ERK, nxm
    """
    if pruner_string == 'lamp':
        return prune_weights_lamp
    elif pruner_string == 'glob':
        return prune_weights_global
    elif pruner_string == 'unif':
        return prune_weights_uniform
    elif pruner_string == 'unifplus':
        return prune_weights_unifplus
    elif pruner_string == 'erk':
        return prune_weights_erk
    elif pruner_string == 'custom':
        return prune_weights_from_config
    else:
        raise ValueError('Unknown pruner')
    
"""
prune_weights_reparam: Allocate identity mask to every weight tensors.
prune_weights_l1predefined: Perform layerwise pruning w.r.t. given amounts.
prune_weights_semistructured: Perform layerwise N:M pruning w.r.t. given configuration (keep N weights out of M consecutive ones)
"""

def prune_weights_semistructured(module, configs=None):
    """
    Remove the weight by the defined N:M configs.
    The configs will be a 2D lists with the following format:
    configs = [[N1:M1], [N2:M2] ...] 
    Please make sure that the len(configs) == number_of_pruning_layers
    """
    def compute_mask(t, N, M):
        out_channel, in_channel = t.shape
        percentile = N / M
        t_reshaped = t.reshape(out_channel, -1, M)
        #print(t_reshaped.shape)
        mask = torch.ones_like(t)
        mask_reshaped = mask.reshape(out_channel, -1, M)
        
        nparams_topprune = int(M * percentile) 
        if nparams_topprune != 0:
            topk = torch.topk(torch.abs(t_reshaped), k=nparams_topprune, largest=False, dim = -1)
            #print(topk.indices)
            mask_reshaped = mask_reshaped.scatter(dim = -1, index = topk.indices, value = 0)
        
        return mask_reshaped.reshape(out_channel, in_channel)
    
    if configs == None:
        raise ValueError("Currently nxm pruning only support from manual config. \
                         Please provide config of the sparsity level for earch pruning target")
    
    mlist = get_modules_with_name(module)
    for idx, (name, m) in enumerate(mlist):
        weight_tensor = m.weight
        config = configs[idx]
        N, M = config[0], config[1]
        print(f"module: {name}, N:M = ({N}, {M})")
        mask = compute_mask(weight_tensor, N, M)
        prune.custom_from_mask(m, name = 'weight', mask = mask)

def prune_weights_reparam(model):
    module_list = get_modules(model)
    for m in module_list:
        prune.identity(m,name="weight")

def prune_weights_l1predefined(model,amounts):
    mlist = get_modules_with_name(model)
    for idx,(name, m) in enumerate(mlist):
        print(f"module: {name}, amounts of removed weight: {float(amounts[idx])}")
        prune.l1_unstructured(m,name="weight",amount=float(amounts[idx]))

"""
Methods: All weights
"""

def prune_weights_global(model,amount):
    parameters_to_prune = _extract_weight_tuples(model)
    prune.global_unstructured(parameters_to_prune,pruning_method = prune.L1Unstructured,amount=amount)

def prune_weights_lamp(model,amount):
    assert amount <= 1
    amounts = _compute_lamp_amounts(model,amount)
    prune_weights_l1predefined(model,amounts)

def prune_weights_uniform(model,amount):
    module_list = get_modules(model)
    assert amount <= 1 # Can be updated later to handle > 1.
    for m in module_list:
        print("module:", m, " remove amount:", amount)
        prune.l1_unstructured(m,name="weight",amount=amount)

def prune_weights_unifplus(model,amount):
    assert amount <= 1
    amounts = _compute_unifplus_amounts(model,amount)
    prune_weights_l1predefined(model,amounts)

def prune_weights_erk(model,amount):
    assert amount <= 1
    amounts = _compute_erk_amounts(model,amount)
    prune_weights_l1predefined(model,amounts)


def prune_weights_from_config(model, mode, configs):
    if mode == 'nxm':
        prune_weights_semistructured(model, configs)
    elif mode == 'unstructured':
        prune_weights_l1predefined(model, amounts=configs)

"""
These are not intended to be exported.
"""

def _extract_weight_tuples(model):
    """
    Gives you well-packed weight tensors for global pruning.
    """
    mlist = get_modules(model)
    return tuple([(m,'weight') for m in mlist])

def _compute_unifplus_amounts(model,amount):
    """
    Compute # of weights to prune in each layer.
    """
    amounts = []
    wlist = get_weights(model)
    unmaskeds = _count_unmasked_weights(model)
    totals = _count_total_weights(model)

    last_layer_minimum = np.round(totals[-1]*0.2) # Minimum number of last-layer weights to keep
    total_to_prune = np.round(unmaskeds.sum()*amount)

    if wlist[0].dim() == 4:
        amounts.append(0) # Leave the first layer unpruned.
        frac_to_prune = (total_to_prune*1.0)/(unmaskeds[1:].sum())
        if frac_to_prune > 1.0:
            raise ValueError("Cannot be pruned further by the Unif+ scheme! (first layer exception)")
        last_layer_to_surv_planned = np.round((1.0-frac_to_prune)*unmaskeds[-1])
        if last_layer_to_surv_planned < last_layer_minimum:
            last_layer_to_prune = unmaskeds[-1] - last_layer_minimum
            frac_to_prune_middle = ((total_to_prune-last_layer_to_prune)*1.0)/(unmaskeds[1:-1].sum())
            if frac_to_prune_middle > 1.0:
                raise ValueError("Cannot be pruned further by the Unif+ scheme! (first+last layer exception)")
            amounts.extend([frac_to_prune_middle]*(unmaskeds.size(0)-2))
            amounts.append((last_layer_to_prune*1.0)/unmaskeds[-1])
        else:
            amounts.extend([frac_to_prune]*(unmaskeds.size(0)-1))
    else:
        frac_to_prune = (total_to_prune*1.0)/(unmaskeds.sum())
        last_layer_to_surv_planned = np.round((1.0-frac_to_prune)*unmaskeds[-1])
        if last_layer_to_surv_planned < last_layer_minimum:
            last_layer_to_prune = unmaskeds[-1] - last_layer_minimum
            frac_to_prune_middle = ((total_to_prune-last_layer_to_prune)*1.0)/(unmaskeds[:-1].sum())
            if frac_to_prune_middle > 1.0:
                raise ValueError("Cannot be pruned further by the Unif+ scheme! (last layer exception)")
            amounts.extend([frac_to_prune_middle]*(unmaskeds.size(0)-1))
            amounts.append((last_layer_to_prune*1.0)/unmaskeds[-1])
        else:
            amounts.extend([frac_to_prune]*(unmaskeds.size(0)))
    return amounts

def _compute_erk_amounts(model,amount):
    unmaskeds = _count_unmasked_weights(model)
    erks = _compute_erks(model)

    return _amounts_from_eps(unmaskeds,erks,amount)

def _amounts_from_eps(unmaskeds,ers,amount):
    num_layers = ers.size(0)
    layers_to_keep_dense = torch.zeros(num_layers)
    total_to_survive = (1.0-amount)*unmaskeds.sum() # Total to keep.

    # Determine some layers to keep dense.
    is_eps_invalid = True
    while is_eps_invalid:
        unmasked_among_prunables = (unmaskeds*(1-layers_to_keep_dense)).sum()
        to_survive_among_prunables = total_to_survive - (layers_to_keep_dense*unmaskeds).sum()

        ers_of_prunables = ers*(1.0-layers_to_keep_dense)
        survs_of_prunables = torch.round(to_survive_among_prunables*ers_of_prunables/ers_of_prunables.sum())

        layer_to_make_dense = -1
        max_ratio = 1.0
        for idx in range(num_layers):
            if layers_to_keep_dense[idx] == 0:
                if survs_of_prunables[idx]/unmaskeds[idx] > max_ratio:
                    layer_to_make_dense = idx
                    max_ratio = survs_of_prunables[idx]/unmaskeds[idx]

        if layer_to_make_dense == -1:
            is_eps_invalid = False
        else:
            layers_to_keep_dense[layer_to_make_dense] = 1

    amounts = torch.zeros(num_layers)

    for idx in range(num_layers):
        if layers_to_keep_dense[idx] == 1:
            amounts[idx] = 0.0
        else:
            amounts[idx] = 1.0 - (survs_of_prunables[idx]/unmaskeds[idx])
    return amounts

def _compute_lamp_amounts(model,amount):
    """
    Compute normalization schemes.
    """
    unmaskeds = _count_unmasked_weights(model)
    num_surv = int(np.round(unmaskeds.sum()*(1.0-amount)))

    flattened_scores = [_normalize_scores(w**2).view(-1) for w in get_weights(model)]
    concat_scores = torch.cat(flattened_scores,dim=0)
    topks,_ = torch.topk(concat_scores,num_surv)
    threshold = topks[-1]

    # We don't care much about tiebreakers, for now.
    final_survs = [torch.ge(score,threshold*torch.ones(score.size()).to(score.device)).sum() for score in flattened_scores]
    amounts = []
    for idx,final_surv in enumerate(final_survs):
        amounts.append(1.0 - (final_surv/unmaskeds[idx]))

    return amounts

def _compute_erks(model):
    wlist = get_weights(model)
    erks = torch.zeros(len(wlist))
    for idx,w in enumerate(wlist):
        if w.dim() == 4:
            erks[idx] = w.size(0)+w.size(1)+w.size(2)+w.size(3)
        else:
            erks[idx] = w.size(0)+w.size(1)
    return erks

def _count_unmasked_weights(model):
    """
    Return a 1-dimensional tensor of #unmasked weights.
    """
    mlist = get_modules(model)
    unmaskeds = []
    for m in mlist:
        unmaskeds.append(m.weight_mask.sum())
    return torch.FloatTensor(unmaskeds)

def _count_total_weights(model):
    """
    Return a 1-dimensional tensor of #total weights.
    """
    wlist = get_weights(model)
    numels = []
    for w in wlist:
        numels.append(w.numel())
    return torch.FloatTensor(numels)

def _normalize_scores(scores):
    """
    Normalizing scheme for LAMP.
    """
    # sort scores in an ascending order
    sorted_scores,sorted_idx = scores.view(-1).sort(descending=False)
    # compute cumulative sum
    scores_cumsum_temp = sorted_scores.cumsum(dim=0)
    scores_cumsum = torch.zeros(scores_cumsum_temp.shape,device=scores.device)
    scores_cumsum[1:] = scores_cumsum_temp[:len(scores_cumsum_temp)-1]
    # normalize by cumulative sum
    sorted_scores /= (scores.sum() - scores_cumsum)
    # tidy up and output
    new_scores = torch.zeros(scores_cumsum.shape,device=scores.device)
    new_scores[sorted_idx] = sorted_scores

    return new_scores.view(scores.shape)
