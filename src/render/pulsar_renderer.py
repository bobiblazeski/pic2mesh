import os
import torch 
import torch.nn.functional as F 

from pytorch3d.structures import Pointclouds
import pytorch3d.transforms as T3

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    PointLights,
    PointsRasterizer,
    PointsRasterizationSettings,
    PulsarPointsRenderer,
)

class PulsarRenderer(torch.nn.Module):
    def __init__(self, opt):    
        super(PulsarRenderer, self).__init__()
        self.opt = opt
        self.max_brightness = opt.raster_max_brightness        
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
        raster_settings = PointsRasterizationSettings(
            image_size= self.opt.raster_image_size, 
            radius = self.opt.raster_radius,
            points_per_pixel = self.opt.raster_points_per_pixel,
        )
        rasterizer = PointsRasterizer(
            cameras= cameras, 
            raster_settings=raster_settings
        )
        lights = PointLights(device=device, 
                             location=[self.opt.lights_location])        
        self.renderer = PulsarPointsRenderer(
            rasterizer=rasterizer,
            n_channels=3
        ).to(device)

    
    def __call__(self, points, translate=True):
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
        cloud = Pointclouds(points=points, features=rgb)
        gamma = [self.opt.pulsar_gamma for _ in range(bs)]
        cameras = self.renderer.rasterizer.cameras
        R = cameras.R.expand(bs, -1, -1)
        T = cameras.T.expand(bs, -1)
        znear = cameras.znear.expand(bs)
        zfar = cameras.zfar.expand(bs)
        fov = cameras.fov.expand(bs)
        aspect_ratio = cameras.aspect_ratio.expand(bs)
        return self.renderer(cloud, gamma=gamma, R=R, T=T,
            znear=znear, zfar=zfar, fov=fov, aspect_ratio=aspect_ratio)
