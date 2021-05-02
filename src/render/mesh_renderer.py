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
    RasterizationSettings,
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.renderer.blending import BlendParams

from src.util import make_faces

class MeshPointsRenderer(torch.nn.Module):
    def __init__(self, opt):    
        super(MeshPointsRenderer, self).__init__()
        self.opt = opt
        self.max_brightness = opt.raster_max_brightness
        size = opt.data_patch_size
        self.register_buffer('faces',  torch.tensor(make_faces(size, size))[None])
        self.renderer = None
    
    def setup(self, device):
        if  self.renderer is not None: return              
        R, T = look_at_view_transform(
            self.opt.viewpoint_distance, 
            self.opt.viewpoint_elevation, 
            self.opt.viewpoint_azimuth, 
            device=device)
        cameras = FoVPerspectiveCameras(
            device=device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size= self.opt.raster_image_size, 
            blur_radius=self.opt.raster_blur_radius, 
            faces_per_pixel=self.opt.raster_faces_per_pixel,
        )
        rasterizer= MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )        
        lights = PointLights(device=device, 
                             location=[self.opt.lights_location])
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
    
    def __call__(self, points, normals=None, translate=True):
        assert len(points.shape) == 3 and points.shape[-1] == 3
        bs = points.size(0)
        rgb = torch.ones((bs, points.size(1), 3), 
                         device=points.device) * self.max_brightness                 
        if translate:
            tm = points.mean(dim=-2, keepdim=False)
            T = T3.Translate(-tm, device=points.device)            
            points = T.transform_points(points)
            # There's error on normals
            # Probably not needed on just translation
            # normals = T.transform_normals(normals)
        
        faces = self.faces.expand(bs, -1, -1)        
        verts_rgb = torch.ones_like(points)
        textures = TexturesVertex(verts_features=verts_rgb.to(points.device))        
        mesh = Meshes(verts=points, faces=faces, textures=textures)
        return self.renderer(mesh)