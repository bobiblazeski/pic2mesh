from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.stylegan2.op import fused_leaky_relu
from src.stylegan2.Blocks import ModConvLayer


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, G_out_ch, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel, G_out_ch, 
                                         kernel_size=ker_size,
                                         stride=stride,
                                         padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(G_out_ch)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('Norm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

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
        self.body = nn.Sequential(OrderedDict([
            ('b1', ConvBlock(opt.G_out_ch, opt.G_out_ch, opt.ker_size, opt.stride, opt.padd_size)),
            ('b2', ConvBlock(opt.G_out_ch, opt.G_out_ch, opt.ker_size, opt.stride, opt.padd_size)),
            ('b3', ConvBlock(opt.G_out_ch, opt.G_out_ch, opt.ker_size, opt.stride, opt.padd_size)),
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
    