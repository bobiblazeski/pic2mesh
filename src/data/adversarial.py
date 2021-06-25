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

def pyramid_transform(mean=0, std=1):
    transform = {
        'head': Compose([
            RandomHorizontalFlip(),
            Grayscale(),
        ]),
        'large': Compose([
            Resize([128, 128]),
            ToTensor(),
            Normalize(mean=(mean), std=(std)),
        ]),
        'medium': Compose([
            Resize([64, 64]),
            ToTensor(),
            Normalize(mean=(mean), std=(std)),
        ]),
        'small': Compose([
            Resize([32, 32]),
            ToTensor(),
            Normalize(mean=(mean), std=(std)),
        ]),
    }
    def final_transform(img):
        flipped = transform['head'](img)
        return {
            'large': transform['large'](flipped),
            # 'medium': transform['medium'](flipped),
            # 'small': transform['small'](flipped),
        }
    return final_transform

class AdversarialDataset(torch.utils.data.Dataset):
    
    def __init__(self, config):
        self.num_workers = config.num_workers
        self.pin_memory = config.pin_memory
        self.adv_mean = config.adversarial_image_mean
        self.adv_std = config.adversarial_image_std
        self.adv_image_root = config.adversarial_image_root        
        self.patch_size = config.adversarial_data_patch_size
        self.full_size =  config.adversarial_data_blueprint_size
                   
        blueprint = np.load(os.path.join(config.data_dir, config.blueprint))
        points = torch.tensor(blueprint['points'])                
        normals = torch.tensor(blueprint['normals'])                
        self.points = self.scale(points, self.full_size)
        self.normals = self.scale(normals, self.full_size)        
        self.entries = self.create_entries(self.points, self.patch_size)        
        self.wmax = self.points.size(-1)
        self.hmax = self.points.size(-2)
        self.channels = self.points.size(0) - 1
        
        adv_transform = pyramid_transform(self.adv_mean, self.adv_std)
        self.adv_ds = ImageFolder(self.adv_image_root, transform=adv_transform)        
        
    def scale(self, t, size):
        return F.interpolate(t, size=size, mode='bicubic', align_corners=True)
        
    def __len__(self):
        return max(len(self.entries), len(self.adv_ds))
    
    def __getitem__(self, idx):        
        ch, w, h = self.entries[idx % len(self.entries)]
        points = self.points[ch, :, w:w + self.patch_size, h:h + self.patch_size]      
        normals = self.normals[ch, :, w:w + self.patch_size, h:h + self.patch_size] 
                
        res, _ = self.adv_ds[idx % len(self.adv_ds)]
        res['points'] = points
        res['normals'] = normals
        return res
    
    def create_entries(self, points, patch_size):
        ps, _, ws, hs = points.shape
        res = []
        for p in range(ps):
            for w in range(ws - patch_size + 1):
                for h in range(hs - patch_size + 1):
                    res.append([p, w, h])
        return torch.tensor(res)

class AdversarialDataProvider:
    
    def __init__(self, config):
        self.ds =  AdversarialDataset(config)
        self.batch_size = config.adversarial_batch_size
        self.pin_memory = config.pin_memory
        self.num_workers = config.num_workers
        self.shuffle = config.shuffle
        self.adv_bs = config.adversarial_batch_size
        self.adv_mean = config.adversarial_image_mean
        self.adv_std = config.adversarial_image_std
        self.adv_real_label = config.adversarial_real_label
        self.adv_fake_label = config.adversarial_fake_label        
        
        self.loader = DataLoader(self.ds, shuffle=self.shuffle,
            batch_size=self.batch_size, num_workers=self.num_workers,
            pin_memory=self.pin_memory)
        self.generator = loader_generator(self.loader)
                
    def __len__(self):
        return len(self.loader)    
    
    def next_batch(self, labels=False, device=None):
        batch = next(self.generator)
        bs = batch['large'].size(0)
        if labels:
            batch['label_real'] = torch.full((bs, 1), self.adv_real_label)
            batch['label_fake'] = torch.full((bs, 1), self.adv_fake_label)
        if device is not None:
            for key in batch.keys():
                batch[key] = batch[key].to(device)
        return batch, (self.adv_mean, self.adv_std)
