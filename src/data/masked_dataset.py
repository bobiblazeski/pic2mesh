import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from random import randint
from PIL import Image

from src.utilities.util import make_faces



class MaskedDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.patch_size = config.data_patch_size
        self.full_size =  config.data_blueprint_size
        self.raster_patch_size = config.raster_patch_size        
        self.img_dir = config.data_image_dir
        self.mask_dir = config.data_mask_dir

        self.faces = torch.tensor(make_faces(self.patch_size, self.patch_size))
        # images =  set([x.replace('.png', '') for x 
        #                in os.listdir(self.img_dir) if x.endswith('.png')])        
        
        # masks = set([x.replace('.pt', '') for x 
        #              in os.listdir(self.mask_dir ) if x.endswith('.pt')])        

        # self.entries = list(images & masks)        
        # if len(self.entries) < len(images):
        #     print('Missing masks', images - masks)
            
        self.transform = {    
            "image_normed": transforms.Compose([
                transforms.ToTensor(),
                #transforms.Resize(config.data_image_size),
                transforms.Normalize(config.image_mean, config.image_std),
                transforms.Grayscale(),
            ]),
            "mask": transforms.Lambda(lambda x: 
                torch.nn.functional.interpolate(x.float(),
                    size=config.data_image_size, mode='nearest').squeeze(0)),            
            "img_patch": transforms.Compose([
                transforms.Resize(config.data_image_resized),
                transforms.RandomCrop(config.data_patch_size),
            ]),            
        }        
        blueprint = np.load(os.path.join(config.data_dir, config.blueprint))
        points = torch.tensor(blueprint['points'])
        
        print(points.shape)
        normals = torch.tensor(blueprint['normals'])
        points_coarse = F.interpolate(points, size=config.data_blueprint_coarse,
                                      mode='bicubic', align_corners=True)
        points_coarse = F.interpolate(points_coarse, size=config.data_blueprint_size,
                                      mode='bicubic', align_corners=True)
        
        points = F.interpolate(points, size=config.data_blueprint_size,
                               mode='bicubic', align_corners=True)
        normals = F.interpolate(normals, size=config.data_blueprint_size, 
                                mode='bicubic', align_corners=True)
        self.entries = self.create_entries(points, self.patch_size)        
        self.points_coarse = points_coarse
        self.points = points
        self.normals = normals
        self.wmax = self.points.size(-1)
        self.hmax = self.points.size(-2)
        self.channels = self.points.size(0) -1
        
    def __len__(self):
        return len(self.entries)        
    
    def __getitem__(self, idx):
        # entry_path = self.entries[idx]
        # img = Image.open(os.path.join(self.img_dir, entry_path + '.png'))
        # mask =  torch.load(os.path.join(self.mask_dir, entry_path + '.pt'))

        # img_normed  = self.transform['image_normed'](img)
        # mask_resized = self.transform['mask'](mask)
        # res_masked =  img_normed * mask_resized        
        # img_patch = self.transform['img_patch'](res_masked)
        
        # ch = randint(0, self.channels)
        # w = randint(0, self.wmax - self.patch_size)
        # h = randint(0, self.hmax - self.patch_size)
        ch, w, h = self.entries[idx]
        points = self.points[ch, :, w:w + self.patch_size, h:h + self.patch_size]
        #assert points.size(-1) == points.size(-2) and  points.size(-2) == self.patch_size, (ch, w, h)
        normals = self.normals[ch, :, w:w + self.patch_size, h:h + self.patch_size]
        points_coarse = self.points_coarse[ch, :, w:w + self.patch_size, h:h + self.patch_size]
        return {            
            #'img_patch': img_patch,
            'points': points,
            'normals': normals,
            'points_coarse': points_coarse,
            #'faces': self.faces,
        }
    
    def create_entries(self, points, patch_size):
        ps, _, ws, hs = points.shape
        res = []
        for p in range(ps):
            for w in range(ws - patch_size + 1):
                for h in range(hs - patch_size + 1):
                    res.append([p, w, h])
        return torch.tensor(res)
