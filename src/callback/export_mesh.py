import os
from random import randint

import numpy as np
import pytorch_lightning as pl
import torch
import trimesh
from src.utilities.util import make_faces, grid_to_list


class ExportMesh(pl.callbacks.Callback):
    
    def __init__(self, opt):
        super().__init__()
        self.num_samples = opt.log_grid_samples        
        self.log_mesh_interval = opt.log_mesh_interval
        self.faces = None        

        
    def on_epoch_end(self, trainer, pl_module):          
        if trainer.current_epoch % self.log_mesh_interval  != 0:
            return
        points = trainer.datamodule.train_dataloader().dataset.points.to(pl_module.device)
        if self.faces is None:
            self.faces = make_faces(points.size(-2), points.size(-1))
        # generate images
        with torch.no_grad():
            pl_module.eval()
            i = randint(0, points.size(0) -1)
            pts = points[i][None]            
            vertices = pl_module.G(pts)[0].cpu().numpy()
            mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces)
            mesh_dir = os.path.join(trainer.log_dir, 'mesh')
            if not os.path.exists(mesh_dir):
                os.makedirs(mesh_dir)
            file_path = os.path.join(mesh_dir, f'mesh_{trainer.current_epoch}_{i}.stl')
            mesh.export(file_path)
            
            if trainer.current_epoch == 0: 
                vertices = grid_to_list(pts)[0].cpu().numpy()
                mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces)
                mesh_dir = os.path.join(trainer.log_dir, 'mesh')
                if not os.path.exists(mesh_dir):
                    os.makedirs(mesh_dir)
                file_path = os.path.join(mesh_dir, f'mesh_{trainer.current_epoch}_{i}_orig.stl')
                mesh.export(file_path)

            pl_module.train()
