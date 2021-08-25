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


class ConvPoolBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super(ConvPoolBlock,self).__init__()        
        conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.add_module('conv', conv)
        #self.add_module('norm', nn.BatchNorm2d(out_ch))
        #self.add_module('swish', nn.SiLU())
        self.add_module('lrelu', nn.LeakyReLU(0.2))
        self.add_module('pool', nn.MaxPool2d(kernel_size=2))

class Stylist(nn.Sequential):
    def __init__(self, config):
        super(Stylist,self).__init__()
        channels = config.stylist_channels
        style_dim = config.style_dim
        for i, (in_ch, out_ch) in enumerate(zip(channels, channels[1:])):
            self.add_module(f'conv{i}', ConvPoolBlock(in_ch, out_ch))            
        self.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
        self.add_module('flatten', nn.Flatten())
        self.add_module('linear', nn.Linear(channels[-1], style_dim))

class Synthesis(nn.Module):
    def __init__(self, config):        
        super(Synthesis,self).__init__()        
        channels = config.synthesis_channels        
        self.input = Slices2D(config.initial_input_file, config.grid_full_size)
        #self.head = ModulatedConv2d(3, channels[0], config.style_dim, 3)
        self.head = StyledConv2d(3, channels[0], config.style_dim, 3)
        self.trunk = nn.ModuleList([
            #ModulatedConv2d(in_ch, out_ch, config.style_dim, 3)
            StyledConv2d(in_ch, out_ch, config.style_dim, 3)
            #ConvBlock(in_ch, out_ch)
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
            #x = layer(x)
        x = self.tail(x)       
        return x



class UpBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super(UpBlock,self).__init__()        
        self.add_module('upsample', nn.UpsamplingBilinear2d(scale_factor=2))
        conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.add_module('conv', conv)        
        self.add_module('lrelu', nn.LeakyReLU(0.2))
        
class Decoder(nn.Module):    
    def __init__(self, config):        
        super(Decoder,self).__init__()
        style_dim = config.style_dim
        channels = config.stylist_channels.copy()
        #channels.reverse()
        no_layers = len(config.stylist_channels ) - 1
        #self.start_size = int(config.fast_image_size / (2 ** no_layers))
        self.start_size = int(config.grid_slice_size / (2 ** no_layers))
        self.linear = nn.Linear(style_dim, self.start_size ** 2)
        self.head = ConvBlock(1, channels[0])
        self.trunk = nn.Sequential(*[            
            UpBlock(in_ch, out_ch)
            for i, (in_ch, out_ch) in
            enumerate(zip(channels, channels[1:]))
        ])
        self.tail = nn.Sequential(
            spectral_norm(nn.Conv2d(channels[-1], channels[0], 3, 1, 1, bias=False)),
            nn.Tanh(),)
        
    def forward(self, style):
        x = self.linear(style)        
        x = x.reshape(style.size(0), 1, self.start_size, self.start_size)                
        x = self.head(x)        
        x = self.trunk(x)                  
        x = self.tail(x)
        return x

class StyleGenerator(nn.Module):
    def __init__(self, config):        
        super(StyleGenerator,self).__init__()
        self.stylist = Stylist(config)
        self.synthesis = Synthesis(config)
        self.decoder = Decoder(config)   
        
    def forward(self, image, slice_idx, size):        
        style = self.stylist(image)                
        points = self.synthesis(style, slice_idx, size)        
        reconstruction =  self.decoder(style)
        return points, reconstruction        