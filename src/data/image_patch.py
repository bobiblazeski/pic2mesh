# pyright: reportMissingImports=false
import torch
from torchvision.transforms import (
    Compose,
    Grayscale,
    Lambda,
    Normalize,    
    ToTensor,    
)
import pytorch_lightning as pl
from PIL import Image


class ImagePatchDataset(torch.utils.data.Dataset):
    
    def __init__(self, config, img_file, mean=0.5, std=0.5):
        self.img = Image.open(img_file)
        self.size = config.fast_image_size
        self.rows = self.img.width - self.size + 1
        self.cols = self.img.height - self.size + 1
        self.length = self.rows * self.cols
        transform = Compose([            
            #Grayscale(),
            ToTensor(),
            Normalize((mean), (std)),
            Lambda(lambda t: t.expand(3, -1, -1)) 
        ])
        self.img_t = transform(self.img)
        
    def __len__(self):
        return self.length    
   
    def __getitem__(self, idx):              
        r, c  = idx // self.rows, idx % self.rows        
        return self.img_t[:, r:r + self.size, c:c + self.size]

class ImagePatchDataModule(pl.LightningDataModule):

    def __init__(self, config, img_file):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.pin_memory = config.pin_memory
        self.train_ds = ImagePatchDataset(self.config, img_file)         
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True,
            batch_size=self.batch_size, num_workers=self.num_workers, 
            pin_memory=self.pin_memory)
