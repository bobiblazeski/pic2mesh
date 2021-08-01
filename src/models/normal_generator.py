# pyright: reportMissingImports=false

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from src.models.util import ConvBlock
from src.utilities.operators import mean_distance

class SinGenerator(nn.Module):
    def __init__(self, channels):
        super(SinGenerator,self).__init__()
        self.trunk = nn.Sequential(OrderedDict([
            ('head', ConvBlock(3, channels[0])),
            ('main', nn.Sequential(OrderedDict([
                ('b'+str(i), ConvBlock(in_ch, out_ch))
                for i, (in_ch, out_ch) in 
                enumerate(zip(channels, channels[1:]))])))
        ]))
        self.points = nn.Sequential(
            spectral_norm(nn.Conv2d(channels[-1], 1, 3, 1, 1, bias=False)),
            nn.Sigmoid(),)      

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
        self.points = SinGenerator(channels)
        #self.colors = SkipGenerator(channels)
    
    def get_means(self, t):        
        means = t.detach().reshape(t.size(0), -1).mean(1)
        return means.reshape(-1, 1, 1, 1)
        
    def forward(self, points, normals):
        #print(baseline.size(-1),  outline.size(-1))       
        dist = mean_distance(points)
        magnitudes = self.points(points)        
        magnitudes = magnitudes - self.get_means(magnitudes)
        magnitudes = magnitudes * dist.reshape(-1, 1, 1, 1)

        res = points + normals * magnitudes
        return res, torch.ones_like(res)