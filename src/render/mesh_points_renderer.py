# pyright: reportMissingImports=false
import os
import torch 
import torch.nn.functional as F 

from pytorch3d.structures import Pointclouds
import pytorch3d.transforms as T3
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    PointLights,
    DirectionalLights,
    RasterizationSettings,
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.renderer.blending import BlendParams
from src.utilities.util import make_faces
from src.utilities.util import  grid_to_list

class MeshPointsRenderer(torch.nn.Module):
    def __init__(self, opt):    
        super(MeshPointsRenderer, self).__init__()
        self.opt = opt
        self.viewpoint_distance = opt.viewpoint_distance, 
        self.viewpoint_elevation = opt.viewpoint_elevation, 
        self.viewpoint_azimuth = opt.viewpoint_azimuth

        self.max_brightness = opt.raster_max_brightness                
        self.renderer = None
    
    def setup(self, device):                    
        R, T = look_at_view_transform(
            self.viewpoint_distance, 
            self.viewpoint_elevation, 
            self.viewpoint_azimuth, 
            device=device)        
        cameras = FoVPerspectiveCameras(
            device=device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size= self.opt.fast_image_size, 
            blur_radius=self.opt.raster_blur_radius, 
            faces_per_pixel=self.opt.raster_faces_per_pixel,
        )
        rasterizer= MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )        
        lights = PointLights(device=device, 
                             location=[self.opt.lights_location])
        lights = DirectionalLights(device=device, 
                             direction=[self.opt.lights_direction])
        shader = SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights,
            blend_params=BlendParams(
              self.opt.blend_params_sigma,
              self.opt.blend_params_gamma,
              self.opt.blend_params_background_color,
            ),
        )        
        self.renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=shader,
        )
    
    def __call__(self, points, faces, colors=None, mean=None, std=None, grayscale=True):
        assert len(points.shape) == 4 and points.shape[1] == 3
        colors = colors if colors is not None else torch.ones_like(points)
        points, colors = grid_to_list(points), grid_to_list(colors)        
        if  self.renderer is None:
            self.setup(points.device)        
        textures = TexturesVertex(verts_features=colors)

        mesh = Meshes(verts=points, faces=faces, textures=textures)
        r_images = self.renderer(mesh)        
        r_images = r_images.permute(0, 3, 1, 2).contiguous()
        r_images = r_images[:, :3, :, :]
        if grayscale:
            r_images = r_images.mean(dim=1, keepdim=True)
        if mean and std:           
            r_images = (r_images - mean) / std
        return r_images