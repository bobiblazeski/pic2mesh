import torch 

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    PointLights,
    PointsRasterizer,
    PointsRasterizationSettings,
)

from src.ShadingPointsRenderer import (
    ShadingCompositor,
    ShadingPointsRenderer,
)

class Renderer(torch.nn.Module):
    def __init__(self, opt):    
        super(Renderer, self).__init__()
        self.opt = opt
    
    def to(self, device):
        new_self = super(Renderer, self).to(device)
        
        
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
        compositor = ShadingCompositor(
            device=device, 
            cameras=cameras,
            lights=lights
        )        
        new_self.renderer = ShadingPointsRenderer(
            rasterizer=rasterizer,
            compositor=compositor,
        ) 
        return new_self
    
    def __call__(self, point_cloud):
        return self.renderer(point_cloud)