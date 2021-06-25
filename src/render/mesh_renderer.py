# pylint: disable=unused-wildcard-import, method-hidden
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
from src.utilities.vertex_normals import VertexNormals
from src.utilities.alignment import align
from src.utilities.util import make_faces
from src.utilities.util import  grid_to_list

class MeshPointsRenderer(torch.nn.Module):
    def __init__(self, opt):    
        super(MeshPointsRenderer, self).__init__()
        self.opt = opt
        self.max_brightness = opt.raster_max_brightness
        self.vrt_nrm = VertexNormals(opt)
        size = opt.adversarial_data_patch_size
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
    
    def __call__(self, point_grid, normals=None, mean_std=None):
        assert len(point_grid.shape) == 4 and point_grid.shape[1] == 3
        points = grid_to_list(point_grid)
        bs, device = points.size(0), points.device
        
        faces = self.faces.expand(bs, -1, -1)        
        verts_rgb = torch.ones_like(points) * self.max_brightness
        textures = TexturesVertex(verts_features=verts_rgb.to(device))

        if normals is None:
            normals = self.vrt_nrm.vertex_normals_fast(points)   

        points, normals = align(points, normals)
        mesh = Meshes(verts=points, faces=faces, textures=textures)
        r_images = self.renderer(mesh)
        print(r_images.shape)
        r_images = r_images.permute(0, 3, 1, 2).contiguous()
        images = r_images[:, :, :, :3].mean(1, keepdim=True)        
        if mean_std is not None:
            mean, std = mean_std
            images = (images - mean) / std
        return images