# pyright: reportMissingImports=false
import os
import torch
import torch.nn.functional as F


class BlendDataset(torch.utils.data.Dataset):
    
    def __init__(self, config):                    
        self.outline_size =  config.fast_outline_size
        self.baseline_size = config.fast_baseline_size        
        self.blends_no = config.data_blends_no
        self.blends_total = config.data_blends_total
        self.data_grid_dir = config.data_grid_dir                
        
        self.grid_files = [os.path.join(self.data_grid_dir, f) 
                           for f in os.listdir(self.data_grid_dir)]
        self.grid_files.sort()
        grid_verts = [torch.load(f)['vertices']for f in  self.grid_files]
        self.grid_baselines = [
            self.scale(v, self.baseline_size) for v in grid_verts]
        self.grid_outlines = [
            self.scale(v, self.outline_size) for v in grid_verts]                
        self.device = torch.device('cpu')        
        
    def scale(self, t, size):
        return F.interpolate(t[None], size=size, mode='bilinear', align_corners=True)[0]
        
    def __len__(self):
        return self.blends_total
    
    def __getitem__(self, _):              
        sources = torch.randint(0, len(self.grid_files), (self.blends_no,))
        baselines = torch.stack([self.grid_baselines[i] for i in sources])
        outlines = torch.stack([self.grid_outlines[i] for i in sources])
        q = F.normalize(torch.rand(self.blends_no), p=1, dim=0).reshape(-1, 1, 1, 1)        
        return {
            'baseline': (baselines * q).sum(dim=0),
            'outline':  (outlines * q).sum(dim=0),
        }