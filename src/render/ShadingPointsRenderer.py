import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d

from typing import List, Optional, Tuple, Union


from pytorch3d.renderer.points.compositor import _add_background_color_to_images

from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.materials import Materials
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.renderer.mesh.shading import _apply_lighting

class ShadingCompositor(nn.Module):
    """
    Accumulate points using a normalized weighted sum.
    """

    def __init__(
        self, background_color: Optional[Union[Tuple, List, torch.Tensor]] = None,
        device="cpu", cameras=None, lights=None, materials=None, blend_params=None
    ):
        super().__init__()
        self.background_color = background_color
        
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, ptclds, **kwargs) -> torch.Tensor:
        background_color = kwargs.get("background_color", self.background_color)
        images =  self._shade(fragments, ptclds)
        #images = norm_weighted_sum(fragments, alphas, ptclds)

        # images are of shape (N, C, H, W)
        # check for background color & feature size C (C=4 indicates rgba)
        if background_color is not None and images.shape[1] == 4:
            return _add_background_color_to_images(fragments, images, background_color)
        return images
    
    def _shade(self, fragments, ptclds, **kwargs) -> torch.Tensor:
        lights = self.lights
        materials = self.materials
        cameras = self.cameras
        
        N, H, W, K = fragments.idx.shape        
        
        pix_to_point  = fragments.idx
        # Replace empty pixels in pix_to_face with 0 in order to interpolate.
        mask = pix_to_point < 0
        pix_to_point = pix_to_point.clone()
        pix_to_point[mask] = 0
        
        idx = pix_to_point.view(N * H * W * K, 1).expand(N * H * W * K, 3)
        pixel_points_vals = ptclds.points_packed().gather(0, 
            idx.long()).view(N, H, W, K, 3)
        
        pixel_normals_vals = ptclds.normals_packed().gather(0, 
            idx.long()).view(N, H, W, K, 3)
        
        ambient, diffuse, specular = _apply_lighting(
            pixel_points_vals, pixel_normals_vals, lights, cameras, materials
        )
        
        colors = (ambient + diffuse)+ specular

        dist_mask = torch.where(fragments.dists != -1, 
                                fragments.dists,
                                torch.zeros_like(fragments.dists))

        dist_mask_normed = F.normalize(dist_mask, p=1, dim=-1)
        dist_mask.shape, dist_mask_normed.sum()

        images = (dist_mask_normed[..., None] * colors).sum(dim=-2)
        
        return images
    
class ShadingPointsRenderer(nn.Module):
    """
    A class for rendering a batch of points. The class should
    be initialized with a rasterizer and compositor class which each have a forward
    function.
    """
    
    def __init__(self, rasterizer, compositor):
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor
    
    
    def to(self, device):
        # Manually move to device rasterizer as the cameras
        # within the class are not of type nn.Module
        self.rasterizer = self.rasterizer.to(device)
        self.compositor = self.compositor.to(device)
        return self
    
    def forward(self, point_clouds, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
#         r = self.rasterizer.raster_settings.radius

#         dists2 = fragments.dists.permute(0, 3, 1, 2)
#         weights = 1 - dists2 / (r * r)
#         images = self.compositor(
#             fragments.idx.long().permute(0, 3, 1, 2),
#             weights,
#             point_clouds.features_packed().permute(1, 0),
#             **kwargs,
#         )

#         # permute so image comes at the end
#         images = images.permute(0, 2, 3, 1)

        images = self.compositor(fragments, point_clouds)
        return images
    
    
