# pyright: reportMissingImports=false
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from src.models.util import ConvBlock

class SkipBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SkipBlock, self).__init__()                
        self.conv = ConvBlock(in_ch, out_ch)        
        self.to_points = ConvBlock(out_ch, 3)        
    
    def upscale(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear',
            align_corners=True)
    
    def forward(self, x, prev_vrt):
        x = self.upscale(x)        
        x = self.conv(x)        
        vrt = self.to_points(x) + self.upscale(prev_vrt)                                   
        return x, vrt

class SkipGenerator(nn.Module):
    def __init__(self, channels):
        super(SkipGenerator, self).__init__()                              
        self.head =  ConvBlock(3, channels[0])        
        self.blocks = nn.ModuleList([
           SkipBlock(in_ch, out_ch) for  (in_ch, out_ch)
            in zip(channels, channels[1:])])
    
    def forward(self, points):                        
        x = self.head(points)
        for block in self.blocks:
            x, points = block(x, points)        
        return points

class ConvBlock1(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock1,self).__init__()
        conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.add_module('conv', conv)
        self.add_module('norm', nn.BatchNorm2d(out_ch))
        #self.add_module('swish', nn.SiLU())
        self.add_module('lrelu', nn.LeakyReLU(0.2))

class SinGenerator(nn.Sequential):
    def __init__(self, channels):
        super(SinGenerator,self).__init__()      
        self.add_module('head', ConvBlock1(3, channels[0]))        
        self.add_module('main', nn.Sequential(OrderedDict([
            ('b'+str(i), ConvBlock1(in_ch, out_ch))
            for i, (in_ch, out_ch) in 
            enumerate(zip(channels, channels[1:]))])))
        self.add_module('tail', nn.Sequential(
            spectral_norm(nn.Conv2d(channels[-1], 3, 3, 1, 1, bias=False)),
            nn.Tanh(),
        ))


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()                      
        channels =  config.fast_generator_channels 
        self.points = SinGenerator(channels)
        #self.colors = SkipGenerator(channels)
    
    def forward(self, outline, ratio=0.05):
        res = self.points(outline) * ratio + outline * (1 - ratio)
        return res, torch.ones_like(res)
