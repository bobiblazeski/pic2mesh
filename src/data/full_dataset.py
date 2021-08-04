# pyright: reportMissingImports=false
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import (
    Compose,
    Grayscale,
    Normalize,
    Resize,
    RandomHorizontalFlip,
    ToTensor,
    ToPILImage,
)
from pytorch3d.io import load_obj, save_obj
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

from torchvision.datasets import ImageFolder

from src.utilities.util import scale_geometry
from src.augment.geoaug import GeoAugment

def pyramid_transform(img_size, mask_size,  mean=0, std=1):
    transform = {
        'preprocess': Compose([
            Resize([mask_size, mask_size]),
            ToTensor(),            
        ]),
        'head': Compose([
            ToPILImage(),
            RandomHorizontalFlip(),            
        ]),
        'image': Compose([
            Resize([img_size, img_size]),
            Grayscale(),
            ToTensor(),
            Normalize(mean=(mean), std=(std)),

        ]),        
    }
    def final_transform(img, mask):
        img = transform['preprocess'](img)
        img = img * mask        
        flipped = transform['head'](img)
        return {
            'image': transform['image'](flipped),            
        }
    return final_transform

class FullDataset(torch.utils.data.Dataset):
    
    def __init__(self, config):
        
        self.num_workers = config.num_workers
        self.pin_memory = config.pin_memory        
        self.outline_size =  config.fast_outline_size
        self.baseline_size = config.fast_baseline_size
        self.stl_offset =  config.stl_offset
        self.geoaug_policy = config.geoaug_policy

        self.image_root = config.fast_image_root
        self.mask_root = config.mask_root            
        self.image_size = config.fast_image_size
        self.mask_size = config.mask_size
        self.image_mean = config.fast_image_mean
        self.image_std = config.fast_image_std
        
        self.data_grid_dir = config.data_grid_dir
        self.data_mesh_dir = config.data_mesh_dir
        
        
        blueprint = np.load(os.path.join(config.data_dir, config.blueprint))
        points = torch.tensor(blueprint['points'])                
                       
        self.points = self.scale(points, self.outline_size)                
        
        self.transform = pyramid_transform(self.image_size, self.mask_size, 
                                           self.image_mean, self.image_std)
        self.img_ds = ImageFolder(self.image_root)
        
        self.grid_files = [os.path.join(self.data_grid_dir, f) 
                           for f in os.listdir(self.data_grid_dir)]
        self.mesh_files = [os.path.join(self.data_mesh_dir, f) 
                           for f in os.listdir(self.data_mesh_dir)]
        self.device = torch.device('cpu')
        
    def scale(self, t, size):
        return F.interpolate(t, size=size, mode='bilinear', align_corners=True)
        
    def __len__(self):
        return len(self.img_ds)
    
    def get_samples(self, idx):
        idx_mesh = idx % len(self.mesh_files)
        mesh_file= self.mesh_files[idx_mesh]
        verts, faces = scale_geometry(mesh_file, self.device, offset=self.stl_offset)
        trg_mesh = Meshes(verts=[verts], faces=[faces])
        samples = sample_points_from_meshes(trg_mesh, self.baseline_size ** 2)[0]
        samples = samples.t().reshape(3, self.baseline_size, self.baseline_size)
        return samples.contiguous()
    
    def get_grid(self, idx):
        idx_grid = idx % len(self.grid_files)
        grid_file =  self.grid_files[idx_grid]
        grid = torch.load(grid_file)
        vertices, normals = grid['vertices'][None], grid['normals'][None]
        vertices = self.scale(vertices, self.baseline_size)
        normals = self.scale(normals, self.baseline_size)
        normals = F.normalize(normals, dim=1)
        return vertices[0], normals[0]
        
    
    def __getitem__(self, idx):              
        # Image
        idx_img = idx % len(self.img_ds)
        image, _ = self.img_ds[idx_img]
        mask_path =  self.img_ds.imgs[0][0].replace(self.image_root, self.mask_root)
        mask = torch.load(mask_path.replace('.png', '.pth'))
        res = self.transform(image, mask)        
        
        # Mesh        
        res['samples'] = self.get_samples(idx)
        
        vertices, normals = self.get_grid(idx)
        res['vertices'] = vertices
        res['normals'] = normals        
        #points = self.points[idx % self.points.size(0)]        
        #res['outline'] = GeoAugment(points, policy=self.geoaug_policy)                 
        return res