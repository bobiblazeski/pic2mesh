from math import sqrt
from collections import OrderedDict

import torch
import torch.nn as nn
import  torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

from src.models.shared import make_chan_in_out
from src.swish import Swish

activations = {
    'Swish': Swish,
    'LeakyReLU': lambda :  torch.nn.LeakyReLU(0.2),
}

class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, act, use_spectral_norm, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters,  kernel_size=1, stride = (2 if downsample else 1))
        self.net = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(input_channels, filters, 3, padding=1),
            'act1': act(),
            'conv2': nn.Conv2d(filters, filters, 3, padding=1),
            'act2': act(),
        }))
        self.downsample = nn.Conv2d(filters, filters, 3, padding = 1, stride = 2) if downsample else None
        if use_spectral_norm:          
            self.conv_res = spectral_norm(self.conv_res)
            self.net.conv1 = spectral_norm(self.net.conv1)
            self.net.conv2 = spectral_norm(self.net.conv2)
            if downsample:
                self.downsample = spectral_norm(self.downsample)

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)        
        return (x + res)  / sqrt(2)

class Discriminator(nn.Module):
    def __init__(self, num_outcomes, filters, act_name='Swish',
          use_adaptive_reparam=True, use_spectral_norm=False):
        super().__init__()
        self.use_adaptive_reparam = use_adaptive_reparam
        self.num_outcomes = num_outcomes        
        chan_in_out = make_chan_in_out(filters)
        layers = OrderedDict([])
        act = activations[act_name]
        for i, (in_chan, out_chan) in enumerate(chan_in_out):            
            is_not_last = i != (len(chan_in_out) - 1)
            layers['b'+str(i)] = DiscriminatorBlock(in_chan, out_chan, act,  
                use_spectral_norm, downsample=is_not_last)
        layers['avgpool'] = nn.AdaptiveAvgPool2d((1, 1))
        layers['flatten'] = nn.Flatten()
        self.model = nn.Sequential(layers)                
        self.fc = nn.Linear(filters[-1], num_outcomes)        
        # resampling trick
        self.reparam = nn.Linear(filters[-1], num_outcomes * 2, bias=False)
        
    def forward(self, x):        
        y = self.model(x)
        output = self.fc(y)
        if self.use_adaptive_reparam:
            stat_tuple = self.reparam(y).unsqueeze(2).unsqueeze(3)
            mu, logvar = stat_tuple.chunk(2, 1)
            std = logvar.mul(0.5).exp_()
            epsilon = torch.randn(x.shape[0], self.num_outcomes, 1, 1).to(stat_tuple)
            output = epsilon.mul(std).add_(mu).view(-1, self.num_outcomes)
        return output