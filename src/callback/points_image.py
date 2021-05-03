import os
from random import randint

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

from src.util import grid_to_list

class PointsImage(pl.callbacks.Callback):
    
    def __init__(self, opt):
        super().__init__()
        self.num_samples = opt.log_grid_samples
        self.nrow = opt.log_grid_rows
        self.padding = opt.log_grid_padding                
        self.pad_value = opt.log_pad_value
        self.patch_size = opt.data_patch_size
        self.log_batch_interval = opt.log_batch_interval
        
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
    
    def sample_blueprint(self, bs):
        points = torch.zeros(bs, 3, self.patch_size, self.patch_size)
        normals = torch.zeros(bs, 3, self.patch_size, self.patch_size)
        for i in range(bs):
            w = randint(0, self.wmax - self.patch_size)
            h = randint(0, self.hmax - self.patch_size)          
            points[i] = self.points[:, w:w + self.patch_size, h:h + self.patch_size]
            normals[i] = self.normals[:, w:w + self.patch_size, h:h + self.patch_size]
        return points, normals
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # show images only every 20 batches
        if (trainer.batch_idx + 1) % self.log_batch_interval != 0:  # type: ignore[attr-defined]
            return
        bs = self.num_samples
        points, normals = self.sample_blueprint(bs)
        points, normals = points.to(pl_module.device), normals.to(pl_module.device)        
        # generate images
        with torch.no_grad():
            pl_module.eval()
            vertices = pl_module.G(points)            
            images1 = pl_module.R(vertices).permute(0, 3, 1, 2)
            images1 =  images1[ :, :3, :, :]
            points = grid_to_list(points)
            images2 = pl_module.R(points).permute(0, 3, 1, 2) 
            images2 =  images2[ :, :3, :, :]          
            images = torch.cat((images1, images2))
            pl_module.train()

        if len(images.size()) == 2:
            img_dim = pl_module.img_dim
            images = images.view(self.num_samples, *img_dim)

        grid = torchvision.utils.make_grid(
            tensor=images,
            nrow=self.nrow,
            padding=self.padding,                        
            pad_value=self.pad_value,
        )
        str_title = f"{pl_module.__class__.__name__}_images"
        trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)
        
    # def on_epoch_end(self, trainer, pl_module):  
    # def on_train_start(self, trainer, pl_module):      
    #     bs = self.num_samples
    #     points, normals = self.sample_blueprint(bs)
    #     points, normals = points.to(pl_module.device), normals.to(pl_module.device)      
    #     # generate images
    #     with torch.no_grad():
    #         pl_module.eval()            
    #         images = pl_module.R(vertices, normals=normals).permute(0, 3, 1, 2)             
    #         pl_module.train()

    #     if len(images.size()) == 2:
    #         img_dim = pl_module.img_dim
    #         images = images.view(self.num_samples, *img_dim)

    #     grid = torchvision.utils.make_grid(
    #         tensor=images,
    #         nrow=self.nrow,
    #         padding=self.padding,                        
    #         pad_value=self.pad_value,
    #     )
    #     str_title = f"{pl_module.__class__.__name__}_images"
    #     trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)