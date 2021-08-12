# pyright: reportMissingImports=false

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from src.models.util import ConvBlock
from src.utilities.operators import mean_distance

class UpConvBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super(UpConvBlock,self).__init__()
        self.add_module('upsample', nn.Upsample(scale_factor=4,
            mode='bilinear', align_corners=True))
        conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.add_module('conv', conv)
        #self.add_module('norm', nn.BatchNorm2d(out_ch))
        #self.add_module('swish', nn.SiLU())
        self.add_module('lrelu', nn.LeakyReLU(0.2))
        
class SurfaceGenerator(nn.Module):
    def __init__(self, channels):
        super(SurfaceGenerator,self).__init__()
        self.trunk = nn.Sequential(OrderedDict([
            ('head', UpConvBlock(3, channels[0])),
            ('main', nn.Sequential(OrderedDict([
                ('b'+str(i), ConvBlock(in_ch, out_ch))
                    for i, (in_ch, out_ch) in 
                    enumerate(zip(channels, channels[1:]))])))
        ]))
        self.points = nn.Sequential(
            spectral_norm(nn.Conv2d(channels[-1], 3, 3, 1, 1, bias=False)),
            #nn.Sigmoid(),)
            nn.Tanh(),)

    def scale(self, t, size):
        return F.interpolate(t, size=size, mode='bilinear', align_corners=True)

    def forward(self, outline):
        trunk = self.trunk(outline)
        points = self.points(trunk)        
        return points


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()                      
        channels =  config.fast_generator_channels 
        self.points = SurfaceGenerator(channels)
        #self.colors = SkipGenerator(channels)
    
    def forward(self, outline):
        #print(baseline.size(-1),  outline.size(-1))       
        points = self.points(outline)         
        return points
