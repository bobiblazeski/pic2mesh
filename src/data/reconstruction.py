import os
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from src.utilities.util import loader_generator

class ReconstructionDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.patch_size = config.reconstruction_data_patch_size
        self.full_size =  config.reconstruction_data_blueprint_size                      
           
        blueprint = np.load(os.path.join(config.data_dir, config.blueprint))
        points = torch.tensor(blueprint['points'])        
        print(points.shape)        
        
        points = F.interpolate(points, size=self.full_size,
                               mode='bicubic', align_corners=True)        
        self.entries = self.create_entries(points, self.patch_size)        
        
        self.points = points        
        self.wmax = self.points.size(-1)
        self.hmax = self.points.size(-2)
        self.channels = self.points.size(0) - 1
        
    def __len__(self):
        return len(self.entries)        
    
    def __getitem__(self, idx):        
        ch, w, h = self.entries[idx]
        points = self.points[ch, :, w:w + self.patch_size, h:h + self.patch_size]              
        return {                        
            'points': points,            
        }
    
    def create_entries(self, points, patch_size):
        ps, _, ws, hs = points.shape
        res = []
        for p in range(ps):
            for w in range(ws - patch_size + 1):
                for h in range(hs - patch_size + 1):
                    res.append([p, w, h])
        return torch.tensor(res)

class ReconstructionDataProvider:
    
    def __init__(self, config):
        self.ds =  ReconstructionDataset(config)
        self.batch_size = config.reconstruction_batch_size
        self.pin_memory = config.pin_memory
        self.num_workers = config.num_workers
        self.shuffle = config.shuffle
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
        return batch