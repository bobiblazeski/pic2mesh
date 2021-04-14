from math import sqrt
from collections import OrderedDict

import torch
import torch.nn as nn
import  torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

from src.models.blocks import ConvBlock


class Discriminator(nn.Module):
    def __init__(self, opt):  
        super().__init__()

        in_ch, out_ch, ker_size, stride, padd_size = (opt.D_in_ch,
            opt.D_out_ch, opt.ker_size, opt.stride, opt.padd_size)
        self.model = nn.Sequential(OrderedDict([
            ('b0', ConvBlock(in_ch, out_ch, ker_size,stride, padd_size)),
            ('b1', ConvBlock(out_ch, out_ch, ker_size,stride, padd_size)),
            ('b2', ConvBlock(out_ch, out_ch, ker_size,stride, padd_size)),
            ('b3', ConvBlock(out_ch, out_ch, ker_size,stride, padd_size)),
            ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
            ('flatten', nn.Flatten()),
            ('linear', nn.Linear(out_ch, 1)),
            ('sigmoid', nn.Sigmoid()),
        ]))        
        
    def forward(self, x):        
        y = self.model(x)
        return y
        