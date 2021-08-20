# pyright: reportMissingImports=false
from collections import OrderedDict


import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from src.models.layers import (
    ConstantInput,    
    StyledConv2d,
)

from src.models.util import ConvBlock

class Stylist(nn.Sequential):
    def __init__(self, config):
        super(Stylist,self).__init__()
        channels = config.stylist_channels
        style_dim = config.style_dim
        for i, (in_ch, out_ch) in enumerate(zip(channels, channels[1:])):
            self.add_module(f'conv{i}', ConvBlock(in_ch, out_ch))            
        self.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
        self.add_module('flatten', nn.Flatten())
        self.add_module('linear', nn.Linear(channels[-1], style_dim))

class Synthesis(nn.Module):
    def __init__(self, config):        
        super(Synthesis,self).__init__()
        channels = config.synthesis_channels        
        self.input = ConstantInput(config.const_input_file, config.grid_size)
        self.head = StyledConv2d(3, channels[0], config.style_dim, 3)
        self.trunk = nn.Sequential(OrderedDict([
                ('b'+str(i), ConvBlock(in_ch, out_ch))
                    for i, (in_ch, out_ch) in 
                    enumerate(zip(channels, channels[1:]))]))
        self.tail = nn.Sequential(
            spectral_norm(nn.Conv2d(channels[-1], 3, 3, 1, 1, bias=False)),            
            nn.Tanh(),)

    def forward(self, style):
        x = self.input(style)        
        x = self.head(x, style)        
        x = self.trunk(x)        
        x = self.tail(x)        
        return x

class StyleGenerator(nn.Module):
    def __init__(self, config):        
        super(StyleGenerator,self).__init__()
        self.stylist = Stylist(config)
        self.synthesis = Synthesis(config)    
        
    def forward(self, image):
        style = self.stylist(image) 
        points = self.synthesis(style)
        return points
