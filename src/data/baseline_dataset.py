# pyright: reportMissingImports=false
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import (
    Compose,  
    Grayscale,  
    Normalize,
    Resize,    
    ToTensor,    
)

def pyramid_transform(img_size, mean=0, std=1):
    transform = {        
        'image': Compose([
            Resize([img_size, img_size]),            
            Grayscale(),
            ToTensor(),
            Normalize(mean=(mean), std=(std)),
        ]),
    }
    def final_transform(img):        
        return {
            'image': transform['image'](img),            
        }
    return final_transform

class BaselineDataset(torch.utils.data.Dataset):
    
    def __init__(self, config):
        
        self.num_workers = config.num_workers
        self.pin_memory = config.pin_memory                
        self.baseline_size = config.fast_baseline_size
        self.geoaug_policy = config.geoaug_policy

        self.image_root = config.fast_render_root        
        self.image_size = config.fast_image_size        
        self.image_mean = config.fast_image_mean
        self.image_std = config.fast_image_std
        self.G_noise_amp = config.G_noise_amp
        
        blueprint = np.load(os.path.join(config.data_dir, config.blueprint))
        points = torch.tensor(blueprint['points'])                
                       
        self.baseline = self.scale(points, self.baseline_size)     
        
    def scale(self, t, size):
        return F.interpolate(t, size=size, mode='bicubic', align_corners=True)
        
    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        baseline = self.baseline[idx % self.baseline.size(0)]
        noise_b =  torch.randn_like(baseline) * self.G_noise_amp
        return baseline + noise_b