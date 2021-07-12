# pyright: reportMissingImports=false
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class SinGenerator(nn.Sequential):
    def __init__(self, channels):
        super(SinGenerator,self).__init__()      
        self.add_module('head', ConvBlock(3, channels[0]))        
        self.add_module('main', nn.Sequential(OrderedDict([
            ('b'+str(i), ConvBlock(in_ch, out_ch))
            for i, (in_ch, out_ch) in 
            enumerate(zip(channels, channels[1:]))])))
        self.add_module('tail', ConvBlock(channels[-1], 3))


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()                      
        channels =  config.fast_generator_channels 
        self.points = SinGenerator(channels)
        #self.colors = SkipGenerator(channels)
    
    def forward(self, points):
        res = self.points(points)
        return res, torch.ones_like(res)
