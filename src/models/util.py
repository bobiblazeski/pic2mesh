# pyright: reportMissingImports=false
import torch.nn as nn
from torch.nn.utils import spectral_norm

def weights_init(m):
    classname = m.__class__.__name__    
    if classname.find('Conv2d') != -1:        
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:        
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock,self).__init__()        
        conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.add_module('conv', conv)
        self.add_module('norm', nn.BatchNorm2d(out_ch))
        #self.add_module('swish', nn.SiLU())
        self.add_module('lrelu', nn.LeakyReLU(0.2))


def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(4), 
                                    conv2d(ch_in, ch_out, 4, 1, 0, bias=False), nn.SiLU(),
                                    conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid() )

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)        
