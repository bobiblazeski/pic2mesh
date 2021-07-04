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
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from src.utilities.util import loader_generator
from src.augment.geoaug import GeoAugment

def pyramid_transform(img_size, mean=0, std=1):
    transform = {
        'head': Compose([
            RandomHorizontalFlip(),            
        ]),
        'big': Compose([
            Resize([img_size, img_size]),
            ToTensor(),
            Normalize(mean=(mean), std=(std)),
        ]),        
    }
    def final_transform(img):
        flipped = transform['head'](img)
        return {
            'big': transform['big'](flipped),            
        }
    return final_transform

class FastDataset(torch.utils.data.Dataset):
    
    def __init__(self, config):
        
        self.num_workers = config.num_workers
        self.pin_memory = config.pin_memory        
        self.blueprint_size =  config.adversarial_data_blueprint_size
        self.geoaug_policy = config.geoaug_policy

        self.image_root = config.fast_image_root                
        self.image_size = config.fast_image_size
        self.image_mean = config.fast_image_mean
        self.image_std = config.fast_image_std
        
        blueprint = np.load(os.path.join(config.data_dir, config.blueprint))
        points = torch.tensor(blueprint['points'])                
                       
        self.points = self.scale(points, self.blueprint_size)                
        
        fast_transform = pyramid_transform(self.image_size, self.image_mean, self.image_std)
        self.img_ds = ImageFolder(self.image_root, transform=fast_transform)      
        
    def scale(self, t, size):
        return F.interpolate(t, size=size, mode='bicubic', align_corners=True)
        
    def __len__(self):
        return len(self.img_ds)
    
    def __getitem__(self, idx):                
        points = self.points[idx % self.points.size(0)]
        
        res, _ = self.img_ds[idx % len(self.img_ds)]
        res['points'] = GeoAugment(points, policy=self.geoaug_policy)         
        return res

class FastDataProvider:
    
    def __init__(self, config):
        self.ds =  FastDataset(config)
        self.batch_size = config.fast_batch_size
        self.pin_memory = config.pin_memory
        self.num_workers = config.num_workers
        self.shuffle = config.shuffle        
        self.fast_mean = config.fast_image_mean
        self.fast_std = config.fast_image_std
        
        self.loader = DataLoader(self.ds, shuffle=self.shuffle,
            batch_size=self.batch_size, num_workers=self.num_workers,
            pin_memory=self.pin_memory)
        self.generator = loader_generator(self.loader)
                
    def __len__(self):
        return len(self.loader)    
    
    def next_batch(self, device=None):
        batch = next(self.generator)              
        if device is not None:
            for key in batch.keys():
                batch[key] = batch[key].to(device)
        return batch, (self.fast_mean, self.fast_std)
