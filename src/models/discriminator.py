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
        self.use_adaptive_reparam = opt.D_use_adaptive_reparam
        self.num_outcomes = opt.D_num_outcomes
        self.filters =  opt.D_filters        
        
        in_ch, out_ch, ker_size, stride, padd_size = (opt.D_in_ch,
            opt.D_out_ch, opt.ker_size, opt.stride, opt.padd_size)
        self.model = nn.Sequential(OrderedDict([
            ('b0', ConvBlock(in_ch, out_ch, ker_size,stride, padd_size)),
            ('b1', ConvBlock(out_ch, out_ch, ker_size,stride, padd_size)),
            ('b2', ConvBlock(out_ch, out_ch, ker_size,stride, padd_size)),
            ('b3', ConvBlock(out_ch, out_ch, ker_size,stride, padd_size)),
            ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
            ('flatten', nn.Flatten()),
            ('linear', nn.Linear(self.filters[-1], 1)),
            ('sigmoid',nn.Sigmoid()),
        ]))
        # self.fc = nn.Linear(self.filters[-1], self.num_outcomes)        
        # # resampling trick
        # self.reparam = nn.Linear(self.filters[-1], self.num_outcomes * 2, bias=False)
        
    def forward(self, x):        
        y = self.model(x)
        return y
        # output = self.fc(y)
        # if self.use_adaptive_reparam:
        #     stat_tuple = self.reparam(y).unsqueeze(2).unsqueeze(3)
        #     mu, logvar = stat_tuple.chunk(2, 1)
        #     std = logvar.mul(0.5).exp_()
        #     epsilon = torch.randn(x.shape[0], self.num_outcomes, 1, 1).to(stat_tuple)
        #     output = epsilon.mul(std).add_(mu).view(-1, self.num_outcomes)
        # return output