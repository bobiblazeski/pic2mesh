
from math import pi

import torch
import torchvision.transforms as transforms


angles = {
    'center' : torch.tensor([0.,   0., pi/2]).reshape(1, 3, 1, 1),
    'norm':   torch.tensor([1., pi/2, pi/2]).reshape(1, 3, 1, 1),
}

center, norm = angles['center'].squeeze(0), angles['norm'].squeeze(0)

transform = transforms.Compose([
    transforms.Lambda(lambda x: (x - center).div(norm)),
])

revert_transform = transforms.Compose([
    transforms.Lambda(lambda x: (x * norm).add(center)),
])


def make_chan_in_out(filters):    
    return list(zip(filters[:-1], filters[1:]))   