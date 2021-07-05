# pyright: reportMissingImports=false
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

from src.models.util import ConvBlock

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):  
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.main = nn.Sequential(OrderedDict([            
            ('pool', nn.AvgPool2d(2, 2)),
            ('conv', ConvBlock(out_ch, out_ch)),
        ]))
    
    def half(self, x):
        size= x.size(-1) // 2
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        res = self.conv(x)
        return self.main(res) + self.half(res)
    
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):  
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.main = ConvBlock(out_ch, out_ch)
    
    def double(self, x):        
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):        
        doubled = self.double(self.conv(x))
        return self.main(doubled) + doubled



class Encoder(nn.Sequential):
    def __init__(self, channels):
        super(Encoder,self).__init__()
        for i, (in_ch, out_ch) in enumerate(zip(channels, channels[1:])):
            self.add_module('b'+str(i), DownBlock(in_ch, out_ch))

class Decoder(nn.Sequential):
    def __init__(self, channels):
        super(Decoder,self).__init__()
        for i, (in_ch, out_ch) in enumerate(zip(channels, channels[1:])):
            self.add_module('b'+str(i), UpBlock(in_ch, out_ch))

class Classifier(nn.Sequential):
    def __init__(self, out_ch):
        super(Classifier,self).__init__()
        self.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
        self.add_module('flatten', nn.Flatten())
        self.add_module('linear', nn.Linear(out_ch, 1))

class Discriminator(nn.Module):
    def __init__(self, opt):  
        super().__init__()

        channels = opt.fast_discriminator_channels
        self.encoder = Encoder(channels)
        self.classifier = Classifier(channels[-1])        
        self.decoder = Decoder(list(reversed(channels)))
        
    def forward(self, images, real=False):
        encodings = self.encoder(images)
        labels = self.classifier(encodings)        
        decodings = self.decoder(encodings) if real else None
        return labels, decodings
        