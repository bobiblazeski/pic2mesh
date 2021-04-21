from collections import OrderedDict

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from src.models.discriminator import Discriminator
from src.models.generator import Generator
from src.render.renderer import Renderer

class GAN(pl.LightningModule):
    
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.automatic_optimization = False
        self.mean = sum(hparams.image_mean) / len(hparams.image_mean)
        self.std = sum(hparams.image_std) / len(hparams.image_std)
        
        self.G = Generator(hparams)
        self.D = Discriminator(hparams)        
        self.R = Renderer(hparams)
     
    def forward(self, shape, style):
        return self.G(shape, style)
    
    def adversarial_loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)
    
    def training_step(self, batch, batch_idx, optimizer_idx):        
        img_patch = batch['img_patch']
        points =  batch['points']
        normals = batch['normals']            
        bs = img_patch.size(0)
        
        self.R.setup(points.device)
        
        # train generator
        if optimizer_idx == 0:            
            vertices = self.G(points)            
            renders =  self.R(vertices).permute(0, 3, 1, 2)             
            renders = (renders - self.mean) / self.std

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(bs, 1).type_as(points)            
           
            g_loss = self.adversarial_loss(self.D(renders), valid)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(bs, 1).type_as(points)         

            real_loss = self.adversarial_loss(self.D(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1).type_as(points)            
                        
            vertices = self.G(points)            
            renders =  self.R(vertices).permute(0, 3, 1, 2)           
            renders = (renders - self.mean) / self.std

            fake_loss = self.adversarial_loss(
                self.D(renders.detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    
    def configure_optimizers(self):
        lr_g = self.hparams.lr_g
        lr_d = self.hparams.lr_d
        b1 = self.hparams.beta1
        b2 = self.hparams.beta2      
        opt_g = torch.optim.Adam(self.G.parameters(), 
                                 lr=lr_g, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.D.parameters(), 
                                 lr=lr_d, betas=(b1, b2))
        return [opt_g, opt_d], []