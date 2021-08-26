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
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.utilities.util import (    
    list_to_grid,
)
def pyramid_transform(img_size, mean=0, std=1):
    transform = Compose([            
        Resize([img_size, img_size]),
        RandomHorizontalFlip(),
        Grayscale(),
        ToTensor(),
        Normalize(mean=(mean), std=(std)),
    ])
    def final_transform(img):        
        return transform(img)
    
    return final_transform

class SlicedRenderDataset(torch.utils.data.Dataset):
    
    def __init__(self, config):        
        self.image_root = config.data_renders_dir        
        self.image_mean = config.fast_image_mean
        self.image_std = config.fast_image_std
        self.image_size = config.fast_image_size
        self.full_size = config.grid_full_size
        self.slice_size = config.grid_slice_size
                
        self.transform = pyramid_transform(self.image_size, 
                                           self.image_mean, self.image_std)
        self.img_ds = ImageFolder(self.image_root, transform=self.transform)
        self.slice_indices = self.make_indices(self.full_size - self.slice_size + 1)
        self.grids =  self.get_all_grids()
        
    def __len__(self):
        return len(self.img_ds) * len(self.slice_indices)
    
    def make_indices(self, n):         
        t = torch.arange(n)
        return torch.stack(torch.torch.meshgrid(t, t), dim=-1).reshape(-1, 2)
    
    def scale(self, t, size):
        return F.interpolate(t, size=size, mode='bilinear', align_corners=True)    
    
    def get_all_grids(self):
        file = f'./data/scaled_{self.full_size}.pth'
        if os.path.exists(file):
           return torch.load(file)
        grids = torch.empty(len(self.img_ds.imgs), 3, self.full_size, self.full_size)
        for i, (f, _) in enumerate(self.img_ds.imgs):
            f = f.replace('renders', 'grid').replace('.png', '.pth')
            grid = list_to_grid(torch.load(f)[None])            
            grids[i] = self.scale(grid, self.full_size)[0]
        torch.save(grids, file)
        return grids
    
    def get_slice(self, idx):
        #grid = self.get_grid(idx % len(self.img_ds))
        grid = self.grids[idx % len(self.img_ds)]
        indices = self.slice_indices[idx % len(self.slice_indices)]
        r, c = indices
        return (grid[:, r:r+self.slice_size, c:c+self.slice_size],
                indices)
   
    def __getitem__(self, idx):              
        res = {}        
        #image, _ = self.img_ds[idx % len(self.img_ds)]
        #res['image'] =  image        
        slice_data, slice_idx = self.get_slice(idx)
        res['slice_data'] = slice_data
        res['slice_idx'] = slice_idx
        return res

class SlicedRenderDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.pin_memory = config.pin_memory
        self.train_ds = SlicedRenderDataset(self.config)         
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.batch_size, 
            num_workers=self.num_workers, pin_memory=self.pin_memory)     