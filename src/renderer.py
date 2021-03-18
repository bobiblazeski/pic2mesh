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

class Renderer():
    def __init__(self, device, viewpoint, lights, raster):        
        R, T = look_at_view_transform(
            viewpoint.distance, 
            viewpoint.elevation, 
            viewpoint.azimuth, 
            device=device)
        cameras = FoVPerspectiveCameras(
            device=device, R=R, T=T)
        raster_settings = PointsRasterizationSettings(
            image_size=raster.image_size, 
            radius = raster.radius,
            points_per_pixel = raster.points_per_pixel,
        )
        rasterizer = PointsRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )
        lights = PointLights(device=device, 
                             location=lights.location)
        compositor = ShadingCompositor(
            device=device, 
            cameras=cameras,
            lights=lights
        )        
        self.renderer = ShadingPointsRenderer(
            rasterizer=rasterizer,
            compositor=compositor,
        )   
    
    def __call__(self, point_cloud):
        return self.renderer(point_cloud)