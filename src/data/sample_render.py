# pyright: reportMissingImports=false
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

class SampleRenderDataset(torch.utils.data.Dataset):
    
    def __init__(self, config):        
        self.image_root = config.data_renders_dir        
        self.image_mean = config.fast_image_mean
        self.image_std = config.fast_image_std
        self.image_size = config.fast_image_size
                
        self.transform = pyramid_transform(self.image_size, 
                                           self.image_mean, self.image_std)
        self.img_ds = ImageFolder(self.image_root, transform=self.transform)                  
        
    def __len__(self):
        return len(self.img_ds)
    
    def get_samples(self, idx):
        f =  self.img_ds.imgs[idx][0]
        f = f.replace('renders', 'samples').replace('.png', '.pth')
        return torch.load(f)
   
    def __getitem__(self, idx):              
        res = {}
        image, label = self.img_ds[idx]
        res['image'] =  image
        res['label'] =  label
        res['samples'] =  self.get_samples(idx)        
        return res

class SampleRenderDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.fast_batch_size
        self.num_workers = config.num_workers
        self.pin_memory = config.pin_memory
        self.train_ds = SampleRenderDataset(self.config)         
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.batch_size, 
            num_workers=self.num_workers, pin_memory=self.pin_memory)        