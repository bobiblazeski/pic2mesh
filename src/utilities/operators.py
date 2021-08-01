# pyright: reportMissingImports=false
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    elif n == 7:          
        hood = [[0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125],
                [0.0125, 0.0250, 0.0250, 0.0250, 0.0250, 0.0250, 0.0125],
                [0.0125, 0.0250, 0.0375, 0.0375, 0.0375, 0.0250, 0.0125],
                [0.0125, 0.0250, 0.0375, 0.0000, 0.0375, 0.0250, 0.0125],
                [0.0125, 0.0250, 0.0375, 0.0375, 0.0375, 0.0250, 0.0125],
                [0.0125, 0.0250, 0.0250, 0.0250, 0.0250, 0.0250, 0.0125],
                [0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125]]

        zero = [[0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]]

        weights = torch.tensor([
            [hood, zero, zero],
            [zero, hood, zero],
            [zero, zero, hood],
        ])
        res = nn.Conv2d(3, 3, 7, stride=1, padding=3, 
            bias=False, padding_mode='replicate')        
    else:
        raise f'Only implemented for n 3 & 5 give{n}'
        
    res.requires_grad_(False)
    res.weight.data = weights
    return res

def mean_distance(pts):    
    def get_distance(pts, hood):
        zeros = [[0., 0.],
                 [0., 0.],]

        weights = torch.tensor([
            [hood, zeros, zeros],
            [zeros, hood, zeros],
            [zeros, zeros, hood],
        ]).to(pts.device)
        dist = F.conv2d(pts, weights) ** 2
        dist = dist.sum(dim=1)    
        dist = torch.sqrt(dist)
        dist = dist.reshape(pts.size(0), -1)
        return dist.mean(1)
    
    v_hood = [[+1., +0.],
              [-1., +0.]]
    h_hood = [[+1., -1.],
              [+0., +0.]]
    v_dist = get_distance(pts, v_hood)
    h_dist = get_distance(pts, h_hood)
    return (v_dist + h_dist) / 2    