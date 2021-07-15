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
    RandomHorizontalFlip,
    ToTensor,    
)
from torchvision.datasets import ImageFolder
from src.augment.geoaug import GeoAugment

def pyramid_transform(img_size, mean=0, std=1):
    transform = {        
        'image': Compose([
            Resize([img_size, img_size]),
            RandomHorizontalFlip(),
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

class RenderDataset(torch.utils.data.Dataset):
    
    def __init__(self, config):
        
        self.num_workers = config.num_workers
        self.pin_memory = config.pin_memory        
        self.outline_size = config.fast_outline_size
        self.baseline_size = config.fast_baseline_size
        self.geoaug_policy = config.geoaug_policy

        self.image_root = config.fast_render_root        
        self.image_size = config.fast_image_size        
        self.image_mean = config.fast_image_mean
        self.image_std = config.fast_image_std
        
        blueprint = np.load(os.path.join(config.data_dir, config.blueprint))
        points = torch.tensor(blueprint['points'])                
                       
        self.outline = self.scale(points, self.outline_size)
        self.baseline = self.scale(points, self.baseline_size)               
        
        self.transform = pyramid_transform(self.image_size, 
            self.image_mean, self.image_std)
        self.img_ds = ImageFolder(self.image_root)
        
    def scale(self, t, size):
        return F.interpolate(t, size=size, mode='bicubic', align_corners=True)
        
    def __len__(self):
        return len(self.img_ds)
    
    def __getitem__(self, idx):              
        baseline = self.baseline[idx % self.baseline.size(0)]
        outline = self.outline[idx % self.outline.size(0)]
        idx_img = idx % len(self.img_ds)
        image, label = self.img_ds[idx_img]        
        res = self.transform(image)
        res['label'] = label
        res['outline'] = GeoAugment(outline, policy=self.geoaug_policy)  
        res['baseline'] = GeoAugment(baseline, policy=self.geoaug_policy)        
        return res