import os
import torch 
import torch.nn.functional as F 

from pytorch3d.structures import Pointclouds
import pytorch3d.transforms as T3

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    PointLights,
    RasterizationSettings,
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
)

from src.render.ShadingPointsRenderer import (
    ShadingCompositor,
    ShadingPointsRenderer,
)

class PointsRenderer(torch.nn.Module):
    def __init__(self, opt):    
        super(PointsRenderer, self).__init__()
        self.opt = opt
        self.max_brightness = opt.raster_max_brightness        
        trimap =  torch.load(os.path.join(opt.data_dir, 
            'trimap_{}.pth'.format(opt.data_patch_size)))        
        self.register_buffer('faces',  trimap['faces'])
        self.register_buffer('vert_tri_indices', trimap['vert_tri_indices'])
        self.register_buffer('vert_tri_weights', trimap['vert_tri_weights'])
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
            blur_radius=0.0, 
            faces_per_pixel=1,
        )
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),        
        lights = PointLights(device=device, 
                             location=[self.opt.lights_location])
        compositor = SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )        
        self.renderer = ShadingPointsRenderer(
            rasterizer=rasterizer,
            compositor=compositor,
        )
    
    def get_face_normals(self, vrt):
        faces = self.faces
        v1 = vrt.index_select(1,faces[:, 1]) - vrt.index_select(1, faces[:, 0])
        v2 = vrt.index_select(1,faces[:, 2]) - vrt.index_select(1, faces[:, 0])
        face_normals = F.normalize(v1.cross(v2), p=2, dim=-1)  # [F, 3]
        return face_normals

    def get_vertex_normals(self, vrt):
        face_normals = self.get_face_normals(vrt)
        bs = face_normals.size(0)
        r, c = self.vert_tri_indices.shape
        fn_group = face_normals.index_select(1, 
            self.vert_tri_indices.flatten()).reshape(bs, r, c, 3)
        weighted_fn_group = fn_group * self.vert_tri_weights    
        vertex_normals = weighted_fn_group.sum(dim=-2)
        return F.normalize(vertex_normals, p=2, dim=-1)

    
    def __call__(self, points, faces, normals=None, translate=True):
        assert len(points.shape) == 3 and points.shape[-1] == 3
        bs = points.size(0)
        rgb = torch.ones((bs, points.size(1), 3), 
                         device=points.device) * self.max_brightness
        if normals is None:
            normals = self.get_vertex_normals(points)            
        if translate:
            tm = points.mean(dim=-2, keepdim=False)
            T = T3.Translate(-tm, device=points.device)            
            points = T.transform_points(points)
            # There's error on normals
            # Probably not needed on just translation
            # normals = T.transform_normals(normals)
        cloud = Pointclouds(points=points, normals=normals, features=rgb)
        return self.renderer(cloud)
