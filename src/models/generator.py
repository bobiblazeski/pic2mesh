from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.stylegan2.op import fused_leaky_relu
from src.stylegan2.Blocks import ModConvLayer
from src.models.blocks import ConvBlock
from src.util import grid_to_list


# Generator
# Takes a style and produces a high resolution model            
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()                      
        in_ch, out_ch, ker_size, stride, padd_size = (opt.G_in_ch,
            opt.G_out_ch, opt.ker_size, opt.stride, opt.padd_size)
        self.head =  ConvBlock(in_ch, out_ch, ker_size,stride, padd_size)
        self.body = nn.Sequential(OrderedDict([
            ('b1', ConvBlock(out_ch, out_ch, ker_size,stride, padd_size)),
            ('b2', ConvBlock(out_ch, out_ch, ker_size,stride, padd_size)),
            ('b3', ConvBlock(out_ch, out_ch, ker_size,stride, padd_size)),
        ]))        
        self.tail = nn.Sequential(
            nn.Conv2d(out_ch, 3, ker_size, stride, padd_size),
            nn.Tanh(),
        )
    
    def forward(self, points):                
        x = self.head(points)
        x = self.body(x)        
        x = self.tail(x)        
        x = grid_to_list(x)
        return x
    