from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.stylegan2.op import fused_leaky_relu
from src.stylegan2.Blocks import ModConvLayer
#from src.models.blocks import ConvBlock
from src.utilities.util import grid_to_list

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_ch, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel, out_ch, 
                                         kernel_size=ker_size,
                                         stride=stride,
                                         padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_ch)),
        #self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))
        self.add_module('Nonlinearity', nn.Hardswish(inplace=True))

    # def weights_init(m):
    #     classname = m.__class__.__name__
    #     if classname.find('Conv2d') != -1:
    #         m.weight.data.normal_(0.0, 0.02)
    #     elif classname.find('Norm') != -1:
    #         m.weight.data.normal_(1.0, 0.02)
    #         m.bias.data.fill_(0) 

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
            #nn.Tanh(),
        )
    
    def forward(self, points):                
        x = self.head(points)
        x = self.body(x)        
        x = self.tail(x)
        #x = x + points
        x = grid_to_list(x)
        return x
    