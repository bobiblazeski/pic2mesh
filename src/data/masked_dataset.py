import os
import numpy as np
import torch
import torchvision.transforms as transforms

from random import randint
from PIL import Image

class MaskedDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.patch_size = config.data_patch_size
        self.img_dir = config.data_image_dir
        self.mask_dir = config.data_mask_dir        
        images =  set([x.replace('.png', '') for x 
                       in os.listdir(self.img_dir) if x.endswith('.png')])        
        
        masks = set([x.replace('.pt', '') for x 
                     in os.listdir(self.mask_dir ) if x.endswith('.pt')])        

        self.entries = list(images & masks)        
        if len(self.entries) < len(images):
            print('Missing masks', images - masks)
            
        self.transform = {    
            "image_normed": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.Grayscale(),
            ]),
            "mask": transforms.Lambda(lambda x: 
                torch.nn.functional.interpolate(x.float(), size=config.data_image_size, 
                                                mode='nearest').squeeze(0)),
            "style_img": transforms.Compose([
                transforms.Resize(config.data_style_img),
            ]),
            "img_patch": transforms.Compose([
                transforms.Resize(config.data_image_resized),
                transforms.RandomCrop(config.data_patch_size),
            ]),
            
        }
        
        
        blueprint =  np.load(config.blueprint)        
        self.points = torch.tensor(blueprint['points'])[0]
        self.normals = torch.tensor(blueprint['normals'])[0]
        
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry_path = self.entries[idx]
        img = Image.open(os.path.join(self.img_dir, entry_path + '.png'))
        mask =  torch.load(os.path.join(self.mask_dir, entry_path + '.pt'))

        img_normed  = self.transform['image_normed'](img)
        mask_resized = self.transform['mask'](mask)
        res_masked =  img_normed * mask_resized
        style_img = self.transform['style_img'](res_masked)
        img_patch = self.transform['img_patch'](res_masked)
        patch_size = self.patch_size
        w, h = randint(0, patch_size), randint(0, patch_size)
        points = self.points[:, w:w + patch_size, h:h + patch_size]
        normals = self.normals[:, w:w + patch_size, h:h + patch_size]
        
        return {
            'style_img': style_img,
            'img_patch': img_patch,
            'points': points,
            'normals': normals,
        }