import os
from random import randint
import numpy as np
import torch
import torch.nn.functional as F

class BlueprintSampler(torch.nn.Module):
    
    def __init__(self, opt):
        super().__init__()                
        self.patch_size = opt.data_patch_size
        
        blueprint = np.load(os.path.join(opt.data_dir, opt.blueprint))        
        points = torch.tensor(blueprint['points'])
        normals = torch.tensor(blueprint['normals'])
        assert len(points.shape) == 4 and len(normals.shape) == 4
        points = F.interpolate(points, size=opt.data_blueprint_size,
                               mode='bicubic', align_corners=True)
        normals = F.interpolate(normals, size=opt.data_blueprint_size, 
                                mode='bicubic', align_corners=True)        
        normals = F.normalize(normals)        
        self.points = points[0]
        self.normals = normals[0]        
        self.wmax = self.points.size(-1)
        self.hmax = self.points.size(-2)
    
    def __call__(self, bs):
        points = torch.zeros(bs, 3, self.patch_size, self.patch_size)
        normals = torch.zeros(bs, 3, self.patch_size, self.patch_size)
        for i in range(bs):
            w = randint(0, self.wmax - self.patch_size)
            h = randint(0, self.hmax - self.patch_size)          
            points[i] = self.points[:, w:w + self.patch_size, h:h + self.patch_size]
            normals[i] = self.normals[:, w:w + self.patch_size, h:h + self.patch_size]
        return points, normals