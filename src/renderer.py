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
)

from src.ShadingPointsRenderer import (
    ShadingCompositor,
    ShadingPointsRenderer,
)


class Renderer(torch.nn.Module):
    def __init__(self, opt):    
        super(Renderer, self).__init__()
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

    
    def __call__(self, points, normals=None, translate=True):
        bs = points.size(0) # (b, 3, w, h)
        points = points.reshape(bs, 3, -1).permute(0, 2, 1)
        rgb = torch.ones((bs, points.size(1), 3), 
                         device=points.device) * self.max_brightness
        if normals is not None:
            normals = normals.reshape(bs, 3, -1).permute(0, 2, 1)            
        else:            
            normals = self.get_vertex_normals(points)
                
        if translate:
            tm = points.mean(dim=-2, keepdim=False)
            T = T3.Translate(-tm, device=points.device)
            # print('points.shape, normals.shape, tm.shape', 
            #       points.shape, normals.shape, tm.shape)
            points = T.transform_points(points)
            # There's error on normals
            # Probably not needed on just translation
            #normals = T.transform_normals(normals)

        point_cloud = Pointclouds(points=points, 
                          normals=normals,
                          features=rgb)
        return self.renderer(point_cloud)