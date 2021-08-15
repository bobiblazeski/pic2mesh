# pyright: reportMissingImports=false
import copy

import torch
import torch.nn.functional as F


def GeoAugment(x, policy='', channels_first=True, ts=None):
    ts = copy.deepcopy(ts) if ts is not None else {}
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            f = AUGMENT_FNS[p]
            change =ts[p] if p in ts else None
            x, t = f(x, change=change)
            ts[p] = t
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x, ts

def rand_translation(x, ratio=0.25, change=None):
    if change is None:
        if len(x.shape) == 4:
            bs = x.size(0)
            change = torch.FloatTensor(bs, 3).uniform_(-ratio, ratio)
            change = change.reshape(bs, 3, 1, 1)
        elif len(x.shape) == 3:
            change = torch.FloatTensor(3).uniform_(-ratio, ratio)
            change = change.reshape(3, 1, 1)
        else:
            change = torch.FloatTensor(3).uniform_(-ratio, ratio)
    return x + change.to(x.device), change

def rand_scaling(x, ratio=0.25, change=None):
    if change is None:
        if len(x.shape) == 4:    
            bs = x.size(0)
            change = torch.FloatTensor(bs, 3).uniform_(1-ratio, 1+ratio)
            change = change.reshape(bs, 3, 1, 1).to(x.device)
        elif len(x.shape) == 3:
            change = torch.FloatTensor(3).uniform_(1-ratio, 1+ratio)
            change = change.reshape(3, 1, 1).to(x.device)
        else:
            change = torch.FloatTensor(3).uniform_(1-ratio, 1+ratio)
    return x * change.to(x.device), change


AUGMENT_FNS = {
    'scaling': rand_scaling,
    'translation': rand_translation,
}