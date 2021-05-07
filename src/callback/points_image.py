import os
from random import randint

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

from src.utilities.util import grid_to_list

class PointsImage(pl.callbacks.Callback):
    
    def __init__(self, opt):
        super().__init__()
        self.num_samples = opt.log_grid_samples
        self.nrow = opt.log_grid_rows
        self.padding = opt.log_grid_padding                
        self.pad_value = opt.log_pad_value
        self.patch_size = opt.data_patch_size
        self.log_batch_interval = opt.log_batch_interval    
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # show images only every log_batch_interval batches
        if (trainer.batch_idx + 1) % self.log_batch_interval != 0:  # type: ignore[attr-defined]
            return
        batch = next(iter(trainer.datamodule.train_dataloader()))
        points = batch['points']        
        pt_normals = batch['normals']        
        bs = points.size(0)
        normals = pt_normals.reshape(bs, 3, -1).permute(0, 2, 1)
        
        points, normals = points.to(pl_module.device), normals.to(pl_module.device)        
        # generate images
        with torch.no_grad():
            pl_module.eval()
            vertices = pl_module.G(points)            
            images1 = pl_module.R(vertices).permute(0, 3, 1, 2)
            images1 =  images1[ :, :3, :, :]
            points = grid_to_list(points)
            images2 = pl_module.R(points, normals).permute(0, 3, 1, 2)
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