import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import ModulateConvBlock
#from src.models.blocks import ConvBlock
from src.utilities.util import grid_to_list

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_ch, ker_size, stride, padding):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel, out_ch, 
                                         kernel_size=ker_size,
                                         stride=stride,
                                         padding=padding)),
        self.add_module('norm',nn.BatchNorm2d(out_ch)),
        #self.add_module('norm',nn.InstanceNorm2d(out_ch)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))
        #self.add_module('Nonlinearity',nn.Tanh())
        #self.add_module('GELU',nn.GELU())
        #self.add_module('Nonlinearity', nn.Hardswish(inplace=True))

#     def weights_init(m):
#         classname = m.__class__.__name__
#         if classname.find('Conv2d') != -1:
#             m.weight.data.normal_(0.0, 0.02)
#         elif classname.find('Norm') != -1:
#             m.weight.data.normal_(1.0, 0.02)
#             m.bias.data.fill_(0) 

class GenBlock(nn.Module):
    def __init__(self, latent_size, in_ch, out_ch, kernel, stride, padding, pool=None):
        super(GenBlock, self).__init__()        
        self.mod_conv =  ModulateConvBlock(latent_size, in_ch, out_ch, kernel)
        #self.mod_conv = ConvBlock(in_ch, out_ch, kernel, stride=stride, padding=padding)
        self.conv1 = ConvBlock(out_ch, out_ch, kernel, stride=stride, padding=padding)
        self.conv2 = ConvBlock(out_ch, out_ch, kernel, stride=stride, padding=padding)
        self.to_points = nn.Conv2d(out_ch, 3, kernel, stride=stride, padding=padding)
        self.pool = nn.AvgPool2d(kernel_size=pool, stride=pool) if pool else None
    
    def upscale(self, x, scale_factor):
        return F.interpolate(x, scale_factor=scale_factor, mode='bilinear', 
                             align_corners=True)# if scale_factor else x     
    
    def forward(self, x, z, prev_vrt):
        x = self.mod_conv(x, z)
        x = self.conv1(x)
        x = self.conv2(x)
        vrt = self.to_points(self.pool(x)) if self.pool else self.to_points(x)                    
        scale_factor = prev_vrt.size(-1) // vrt.size(-1)        
        if scale_factor > 1:
            vrt = self.upscale(vrt, scale_factor)
        vrt = vrt + prev_vrt
        return x, vrt
        
        

    
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()                      
        latent_size, in_ch, out_ch, ker_size, stride, padding= (opt.latent_size,
            opt.G_in_ch, opt.G_out_ch, opt.ker_size, opt.stride, opt.padd_size)
        self.head =  ConvBlock(in_ch, out_ch, ker_size, stride=stride, padding=padding)
        self.pools = [4, 2, None]
        
        self.b1 = GenBlock(latent_size, out_ch, out_ch, ker_size, stride=stride, 
                      padding=padding, pool=self.pools[0])
        self.b2 = GenBlock(latent_size, out_ch, out_ch, ker_size, stride=stride, 
                      padding=padding, pool=self.pools[1])
        self.b3 = GenBlock(latent_size, out_ch, out_ch, ker_size, stride=stride, 
                           padding=padding, pool=self.pools[2])
    
    def forward(self, points, style):                        
        x = self.head(points)
        
        x, vrt = self.b1(x, style, points)
        x, vrt = self.b2(x, style, vrt)
        x, vrt = self.b3(x, style, vrt)
        
        #vrt = grid_to_list(vrt)
        return vrt