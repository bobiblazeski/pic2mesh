# pyright: reportMissingImports=false
import torch
import torch.nn as nn

def make_laplacian(n):
    if n == 3:
        hood = [[0.125, 0.125, 0.125],
                [0.125, 0.000, 0.125],
                [0.125, 0.125, 0.125],]

        zeros = [[0.000, 0.000, 0.000],
                [0.000, 0.000, 0.000],
                [0.000, 0.000, 0.000],]

        weights = torch.tensor([
            [hood, zeros, zeros],
            [zeros, hood, zeros],
            [zeros, zeros, hood],
        ])
        res = nn.Conv2d(3, 3, 3, stride=1, padding=1, 
            bias=False, padding_mode='replicate')
    elif n == 5:
        a = 0.03125
        b =  0.0625   
        hood = [[a, a, a, a, a],
                [a, b, b, b, a],
                [a, b, 0, b, a],
                [a, b, b, b, a],
                [a, a, a, a, a]]

        zero = [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]

        weights = torch.tensor([
            [hood, zero, zero],
            [zero, hood, zero],
            [zero, zero, hood],
        ])
        res = nn.Conv2d(3, 3, 5, stride=1, padding=2, 
            bias=False, padding_mode='replicate')        
    else:
        raise f'Only implemented for n 3 & 5 give{n}'
        
    res.requires_grad_(False)
    res.weight.data = weights
    return res