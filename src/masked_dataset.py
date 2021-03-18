import os
from PIL import Image
import torch

class MaskedDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, transform):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        images =  set([x.replace('.png', '') for x 
                       in os.listdir(img_dir) if x.endswith('.png')])
        masks = set([x.replace('.pt', '') for x 
                     in os.listdir(mask_dir) if x.endswith('.pt')])        
        self.entries = list(images & masks)
        if len(self.entries) < len(images):
            print('Missing masks', images - masks)        
        
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry_path = self.entries[idx]
        img = Image.open(os.path.join(self.img_dir, entry_path + '.png'))
        mask =  torch.load(os.path.join(self.mask_dir, entry_path + '.pt'))
        
        img_normed  = self.transform['image_normed'](img)
        img_raw =  self.transform['image_raw'](img)                
        mask_2x = self.transform['mask'](mask)
        
        res_normed =  img_normed * mask_2x
        res_raw =  img_raw * mask_2x
        return {'res_raw': res_raw, 'res_normed': res_normed}
