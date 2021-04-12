from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.stylegan2.op import fused_leaky_relu
from src.stylegan2.Blocks import ModConvLayer
from src.models.blocks import ConvBlock



# Generator
# Takes a style and produces a high resolution model            
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.noise_amp = opt.G_noise_amp        
        self.head = ModConvLayer(
            opt.dlatent_size, 
            opt.G_in_ch,
            opt.G_out_ch, 
            opt.ker_size, 
        )
        out_ch, out_ch, ker_size, stride, padd_size = (opt.G_out_ch,
            opt.G_out_ch, opt.ker_size, opt.stride, opt.padd_size)
        self.body = nn.Sequential(OrderedDict([
            ('b1', ConvBlock(out_ch, out_ch, ker_size,stride, padd_size)),
            ('b2', ConvBlock(out_ch, out_ch, ker_size,stride, padd_size)),
            ('b3', ConvBlock(out_ch, out_ch, ker_size,stride, padd_size)),
        ]))
        
        self.tail = nn.Sequential(
            nn.Conv2d(opt.G_out_ch, opt.G_out_ch, opt.ker_size, opt.stride, opt.padd_size),
            nn.Tanh(),
        )
    
    def forward(self, points, normals, style):
        #vertices = self.vertices.expand(style.size(0), -1, -1, -1)
        noise = torch.randn_like(points) * self.noise_amp                
        x = torch.cat((points + noise, normals), dim=1)
        x = self.head(x, style)        
        x = self.body(x)        
        x = self.tail(x)        
        return x
    