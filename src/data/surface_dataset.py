# pyright: reportMissingImports=false
import os
import random
import torch
import torch.nn.functional as F
from src.augment.geoaug import GeoAugment


class SurfaceDataset(torch.utils.data.Dataset):
    
    def __init__(self, config):
        self.geoaug_policy = config.geoaug_policy
        self.outline_size =  config.fast_outline_size
        self.baseline_size = config.fast_baseline_size        
        self.blends_no = config.data_blends_no
        self.blends_total = config.data_blends_total
        self.data_grid_dir = config.data_grid_dir                
        
        self.grid_files = [os.path.join(self.data_grid_dir, f) 
                           for f in os.listdir(self.data_grid_dir)]
        self.grid_files.sort()
        self.grid_verts = [torch.load(f)['vertices']for f in  self.grid_files]
        
        self.device = torch.device('cpu')        
        
    def scale(self, t, size):
        return F.interpolate(t[None], size=size, mode='bilinear', align_corners=True)[0]
        
    def __len__(self):
        return len(self.grid_files)

    def get_blends(self, _):
        sources = torch.randint(0, len(self.grid_files), (self.blends_no,))
        baselines = torch.stack([ 
            self.scale(self.grid_verts[i], self.baseline_size) 
            for i in sources])
        outlines = torch.stack([self.scale(self.grid_verts[i], self.outline_size)
            for i in sources])
        q = F.normalize(torch.rand(self.blends_no), p=1, dim=0).reshape(-1, 1, 1, 1)        
        return {
            'baseline': (baselines * q).sum(dim=0),
            'outline':  (outlines * q).sum(dim=0),
        }

    def get_grids(self, idx):
        idx_grid = idx % len(self.grid_files)
        orig = GeoAugment(self.grid_verts[idx_grid], self.geoaug_policy)
        baseline = self.scale(orig, self.baseline_size)
        outline = self.scale(orig, self.outline_size)
        return {
            'baseline': baseline, 
            'outline':  outline,
        }
    
    def __getitem__(self, idx):              
        if random.random() < 0.25:
            return self.get_blends(idx)
        else:
            return self.get_grids(idx)
