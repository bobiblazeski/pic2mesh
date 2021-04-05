import torch
import torch.nn as nn
import torch.nn.functional as F

from src.stylegan2.op import fused_leaky_relu
from src.stylegan2.Blocks import ModConvLayer


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel, out_channel, 
                                         kernel_size=ker_size,
                                         stride=stride,
                                         padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
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
        # blueprint = torch.load(opt.blueprint)
        # self.register_buffer('vertices', blueprint['vertices'])
        # self.register_buffer('normals', blueprint['normals'])
        

        self.head = ModConvLayer(
            opt.dlatent_size, 
            opt.in_channel,
            opt.out_channel, 
            opt.ker_size, 
        )
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N, opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        
        self.tail = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc),opt.nc_im,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
            nn.Tanh()
        )
    
    def forward(self, points, normals, style):
        #vertices = self.vertices.expand(style.size(0), -1, -1, -1)
        noise = torch.randn_like(points) * self.noise_amp                
        x = torch.cat((points + noise, normals), dim=1)
        x = self.head(x, style)        
        x = self.body(x)        
        x = self.tail(x)        
        return x
    