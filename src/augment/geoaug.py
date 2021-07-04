# pyright: reportMissingImports=false
import torch
import torch.nn.functional as F


def GeoAugment(x, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x

def rand_translation(x, ratio=0.125):
    if len(x.shape) == 4:
        bs = x.size(0)
        translations = torch.FloatTensor(bs, 3).uniform_(-ratio, ratio)    
        translations = translations.reshape(bs, 3, 1, 1)
    else:
        translations = torch.FloatTensor(3).uniform_(-ratio, ratio)    
        translations = translations.reshape(3, 1, 1)
    x += translations.to(x.device)
    return x

def rand_scaling(x, ratio=0.2):
    if len(x.shape) == 4:    
        bs = x.size(0)
        scalings = torch.FloatTensor(bs, 3).uniform_(1-ratio, 1+ratio)
        scalings = scalings.reshape(bs, 3, 1, 1).to(x.device)
    else:
        scalings = torch.FloatTensor(3).uniform_(1-ratio, 1+ratio)
        scalings = scalings.reshape(3, 1, 1).to(x.device)

    x *= scalings.to(x.device)
    return x


AUGMENT_FNS = {
    'scaling': [rand_scaling],
    'translation': [rand_translation],    
}