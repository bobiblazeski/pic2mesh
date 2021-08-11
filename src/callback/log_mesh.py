# pyright: reportMissingImports=false
import os

import torch
import torch.nn.functional as F
import trimesh
import pytorch_lightning as pl

from src.utilities.util import (
    grid_to_list,
    make_faces,
)

class LogMesh(pl.callbacks.Callback):
    
    def __init__(self, opt):
        super().__init__()
        self.num_samples = opt.log_grid_samples
        self.nrow = opt.log_grid_rows
        self.padding = opt.log_grid_padding                
        self.pad_value = opt.log_pad_value        
        self.log_mesh_interval = opt.log_mesh_interval
        self.faces = None
        
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # show images only every log_batch_interval batches
        if (trainer.global_step % self.log_mesh_interval) != 0:  # type: ignore[attr-defined]
            return
        batch = next(iter(trainer.datamodule.train_dataloader()))
        
        
        # generate images
        with torch.no_grad():
            pl_module.eval()
            vertices = pl_module.G(batch['outline'].to(pl_module.device)) 
            pl_module.train()

            try:            
                if self.faces is None:
                    self.faces = make_faces(vertices.size(-2), vertices.size(-1))
                vertices = grid_to_list(vertices)[0].cpu().numpy()
                mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces)
                mesh_dir = os.path.join(trainer.log_dir, 'mesh')
                if not os.path.exists(mesh_dir):
                    os.makedirs(mesh_dir)
                file_path = os.path.join(mesh_dir, f'vertices_{trainer.current_epoch}_{trainer.global_step}.stl')
                mesh.export(file_path)
                
            except:
                print('Exception', vertices.shape)
                pass
        
    