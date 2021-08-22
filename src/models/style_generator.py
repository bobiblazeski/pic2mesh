# pyright: reportMissingImports=false
from collections import OrderedDict


import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from src.models.layers import ( 
    ModulatedConv2d, 
    Slices2D,    
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
        self.input = Slices2D(config.initial_input_file, config.grid_full_size)
        self.head = ModulatedConv2d(3, channels[0], config.style_dim, 3)              
        self.trunk = nn.ModuleList([
            ModulatedConv2d(in_ch, out_ch, config.style_dim, 3)
            for i, (in_ch, out_ch) in
            enumerate(zip(channels, channels[1:]))
        ])
        self.tail = nn.Sequential(
            spectral_norm(nn.Conv2d(channels[-1], 3, 3, 1, 1, bias=False)),            
            nn.Tanh(),)

    def forward(self, style, slice_idx, size):
        x = self.input(slice_idx, size)
        x = self.head(x, style) 
        for layer in self.trunk:
            x = layer(x, style)        
        x = self.tail(x)        
        return x        

class StyleGenerator(nn.Module):
    def __init__(self, config):        
        super(StyleGenerator,self).__init__()
        self.stylist = Stylist(config)
        self.synthesis = Synthesis(config)    
        
    def forward(self, image, slice_idx, size):
        style = self.stylist(image)
        points = self.synthesis(style, slice_idx, size)
        return points
