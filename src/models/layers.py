# pyright: reportMissingImports=false

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConstantInput(nn.Module):
    def __init__(self, file, size, fixed):                
        super().__init__()
        t = torch.load(file)
        t = F.interpolate(t, size=size, mode='bilinear', align_corners=True)        
        self.fixed = fixed
        if fixed:
            self.register_buffer('input', t)
        else:
            self.input = nn.Parameter(t)

    def forward(self, t):
        if self.fixed:
            return self.input.expand(t.shape[0], -1, -1, -1)
        return self.input.repeat(t.shape[0], 1, 1, 1)

class Slices2D(nn.Module):
    def __init__(self, file, size):                
        super().__init__()
        t = torch.load(file)
        t = F.interpolate(t, size=size, mode='bilinear', align_corners=True)[0]
        self.register_buffer('t', t)

    def forward(self, slice_idx, size):
        res = torch.empty(slice_idx.size(0), 3, size, size, device=slice_idx.device)
        for i, (r, c) in enumerate(slice_idx):
            res[i] = self.t[:, r:r+size, c:c+size]
        return res

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.trace_model = False

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        if not hasattr(self, "noise") and self.trace_model:
            self.register_buffer("noise", noise)
        if self.trace_model:
            noise = self.noise
        return image + self.weight * noise

class ModulatedConv2d(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size,
            demodulate=True
    ):
        super().__init__()
        # create conv
        self.weight = nn.Parameter(
            torch.randn(channels_out, channels_in, kernel_size, kernel_size)
        )
        # create modulation network
        self.modulation = nn.Linear(style_dim, channels_in, bias=True)
        self.modulation.bias.data.fill_(1.0)
        # create demodulation parameters
        self.demodulate = demodulate
        if self.demodulate:
            self.register_buffer("style_inv", torch.randn(1, 1, channels_in, 1, 1))
        # some service staff
        self.scale = 1.0 / math.sqrt(channels_in * kernel_size ** 2)
        self.padding = kernel_size // 2

    def forward(self, x, style):
        modulation = self.get_modulation(style)
        x = modulation * x
        x = F.conv2d(x, self.weight, padding=self.padding)
        if self.demodulate:
            demodulation = self.get_demodulation(style)
            x = demodulation * x
        return x

    def get_modulation(self, style):
        style = self.modulation(style).view(style.size(0), -1, 1, 1)
        modulation = self.scale * style
        return modulation

    def get_demodulation(self, style):
        w = self.weight.unsqueeze(0)
        norm = torch.rsqrt((self.scale * self.style_inv * w).pow(2).sum([2, 3, 4]) + 1e-8)
        demodulation = norm
        return demodulation.view(*demodulation.size(), 1, 1)

class StyledConv2d(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        style_dim,
        kernel_size,
        demodulate=True,
        conv_module=ModulatedConv2d
    ):
        super().__init__()

        self.conv = conv_module(
            channels_in,
            channels_out,
            style_dim,
            kernel_size,
            demodulate=demodulate
        )

        self.noise = NoiseInjection()
        self.bias = nn.Parameter(torch.zeros(1, channels_out, 1, 1))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.act(out + self.bias)
        return out
