# pyright: reportMissingImports=false
import os
import torch
import torch.nn.functional as F

class GridDataset(torch.utils.data.Dataset):
    
    def __init__(self, config):
        
        self.outline_size =  config.fast_outline_size
        self.baseline_size = config.fast_baseline_size        
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
        return len(self.grid_files)
    
    def __getitem__(self, idx):              
        idx_grid = idx % len(self.grid_files)
        return {
            'baseline': self.grid_baselines[idx_grid],             
            'outline':  self.grid_outlines[idx_grid],
        }