
import numpy as np
import torch

from math import sin, cos, pi
from PIL import Image

from matplotlib import cm
from geomdl import BSpline, knotvector
from geomdl.visualization import VisPlotly

import trimesh
from trimesh.util import triangle_strips_to_faces
from sklearn.preprocessing import minmax_scale
# pylint: disable=maybe-no-member

angles = {
    'center': torch.tensor([0.,   0., pi/2]).reshape(1, 3, 1, 1),
    'norm':   torch.tensor([1., pi/2, pi/2]).reshape(1, 3, 1, 1),
}

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


def feature_loss(input, in_feat, target, out_feat, 
            loss_fn=torch.nn.functional.l1_loss, pixel_loss=False):                               
    losses = []
    if pixel_loss:
        losses += [self.loss_fn(input,target)] # Pixel loss
    losses += [self.loss_fn(in_feat, out_feat)]
    losses += [self.loss_fn(gram_matrix(in_feat), gram_matrix(out_feat)) * 5e3]

    return sum(losses) 
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def make_faces(w, h):
#     mesh_indices = []
#     for iw  in range(w-1):
#         for ih in range(h-1):
#             mesh_indices.append([iw*w+ih, iw*w+ih+1, (iw+1)*w+ih])            
#             mesh_indices.append([iw*w+ih+1, (iw+1)*w+ih+1, (iw+1)*w+ih])
#     return np.array(mesh_indices)   


### Rotation matrices    
def ax(phi):
    return torch.tensor([
        [1.,       0.,        0.],
        [0., cos(phi), -sin(phi)],
        [0., sin(phi),  cos(phi)]])

def ay(theta):
    return torch.tensor([
        [ cos(theta), 0., sin(theta)],
        [         0., 1.,          0],
        [-sin(theta), 0,  cos(theta)]])

def az(psi):
    return torch.tensor([
        [ cos(psi), -sin(psi), 0.],
        [ sin(psi),  cos(psi), 0.],
        [        0,         0, 1.]])

def axyz(phi, theta, psi):
    return torch.matmul(az(psi), torch.matmul(ay(theta), ax(phi)))  

def show_img(pixels):      
    return Image.fromarray(np.uint8(
        pixels.squeeze().detach().cpu().numpy() * 255) , 'L')

def render_spline(ctrl_pts, rows, cols):
    ctrl_pts = ctrl_pts.reshape(-1, 3).detach().cpu().numpy().tolist()
    pts_h = rows
    pts_v = cols
    surf = BSpline.Surface()
    surf.degree_u = 3
    surf.degree_v = 3
    surf.set_ctrlpts(ctrl_pts, pts_h, pts_v)
    surf.knotvector_u = knotvector.generate(3, pts_h, clamped=True)
    surf.knotvector_v = knotvector.generate(3, pts_v, clamped=True)
    surf.delta = 0.025
    surf.evaluate()
    surf.vis = VisPlotly.VisSurface()
    surf.render(colormap=cm.cool)

def sample_mesh(vertices, faces, num_samples: int, eps: float = 1e-10):
    r""" Uniformly samples the surface of a mesh.

        Args:
            num_samples (int): number of points to sample
            eps (float): a small number to prevent division by zero
                          for small surface areas.

        Returns:
            (torch.Tensor, torch.Tensor) uniformly sampled points and
                the face idexes which each point corresponds to.

        Example:
            >>> points, chosen_faces = mesh.sample(10)
            >>> points
            tensor([[ 0.0293,  0.2179,  0.2168],
                    [ 0.2003, -0.3367,  0.2187],
                    [ 0.2152, -0.0943,  0.1907],
                    [-0.1852,  0.1686, -0.0522],
                    [-0.2167,  0.3171,  0.0737],
                    [ 0.2219, -0.0289,  0.1531],
                    [ 0.2217, -0.0115,  0.1247],
                    [-0.1400,  0.0364, -0.1618],
                    [ 0.0658, -0.0310, -0.2198],
                    [ 0.1926, -0.1867, -0.2153]])
            >>> chosen_faces
            tensor([ 953,  38,  6, 3480,  563,  393,  395, 3309, 373, 271])
    """

    if vertices.is_cuda:
        dist_uni = torch.distributions.Uniform(
            torch.tensor([0.0]).cuda(), torch.tensor([1.0]).cuda())
    else:
        dist_uni = torch.distributions.Uniform(
            torch.tensor([0.0]), torch.tensor([1.0]))

    # calculate area of each face
    x1, x2, x3 = torch.split(torch.index_select(
        vertices, 0, faces[:, 0]) - torch.index_select(
        vertices, 0, faces[:, 1]), 1, dim=1)
    y1, y2, y3 = torch.split(torch.index_select(
        vertices, 0, faces[:, 1]) - torch.index_select(
        vertices, 0, faces[:, 2]), 1, dim=1)
    a = (x2 * y3 - x3 * y2)**2
    b = (x3 * y1 - x1 * y3)**2
    c = (x1 * y2 - x2 * y1)**2
    Areas = torch.sqrt(a + b + c) / 2
    # percentage of each face w.r.t. full surface area
    Areas = Areas / (torch.sum(Areas) + eps)

    # define descrete distribution w.r.t. face area ratios caluclated
    cat_dist = torch.distributions.Categorical(Areas.view(-1))
    face_choices = cat_dist.sample([num_samples])

    # from each face sample a point
    select_faces = faces[face_choices]
    v0 = torch.index_select(vertices, 0, select_faces[:, 0])
    v1 = torch.index_select(vertices, 0, select_faces[:, 1])
    v2 = torch.index_select(vertices, 0, select_faces[:, 2])
    u = torch.sqrt(dist_uni.sample([num_samples]))
    v = dist_uni.sample([num_samples])
    points = (1 - u) * v0 + (u * (1 - v)) * v1 + u * v * v2

    return points#, face_choices

def get_geometry(stl_path, device=None, offset=0.1, start=0, end=1, scale=True):
    mesh = trimesh.load(stl_path)    
    # Center
    vertices = mesh.vertices
    if scale:
        vertices = minmax_scale(mesh.vertices.flatten(),
            feature_range=(start+offset, end-offset)) # Must be positive?! 
    vertices = torch.from_numpy(vertices).float().reshape(-1, 3)
    faces = torch.from_numpy(mesh.faces).long() # float()
    if device:
        vertices, faces = vertices.to(device), faces.to(device)
    return vertices, faces    

def scale_geometry(stl_path, device, offset=0.1):
    mesh = trimesh.load(stl_path)    
    # Scale 
    vertices = torch.from_numpy(mesh.vertices).float()
    vertices = vertices / vertices.abs().max() * (1.-offset)
    faces = torch.from_numpy(mesh.faces).long() # float()    
    return vertices.to(device), faces.to(device)

def get_nearest(points, stl_path, offset=0.1):
    mesh = trimesh.load(stl_path)
    vertices = minmax_scale(
        mesh.vertices.flatten(),
        feature_range=(offset, 1-offset)# Must be positive?!
    ).reshape(-1, 3)   
    mesh = trimesh.Trimesh(vertices=vertices, 
                           faces=mesh.faces)
    (closest_points,
     distances,
     triangle_id) = mesh.nearest.on_surface(points)
    return closest_points

def to_np(t):
    return t.cpu().detach().numpy()

def np_flat(arr, dtype=np.float32):
    return arr.flatten().astype(dtype)
    
def interleave(radius_c, radius_f, r_shape_c, r_shape_f):
    assert radius_c.size(0) == radius_f.size(0)  
    bsz = radius_c.size(0)
    row_env_no = (r_shape_f[0]-r_shape_c[0])*r_shape_f[1]    
    row_envl = radius_f[:, :row_env_no].reshape(bsz, 
      r_shape_f[0]-r_shape_c[0], r_shape_f[1])
    row_insd = radius_f[:, row_env_no:]
    radius_c_l = radius_c[:, r_shape_c[0]:]
    radius_c_r = radius_c[:, :r_shape_c[0]].reshape(bsz, -1, 1)
    interleave_l = torch.stack((radius_c_l, row_insd), 
      dim=1).permute(0, 2, 1) .contiguous().view(bsz, r_shape_c[0], -1)
    interleave = torch.cat((interleave_l, radius_c_r), dim=2)
    btw = torch.cat((interleave, row_envl[:,1:, :]), 
      dim=2).reshape(bsz, r_shape_f[0]-1, r_shape_f[1])
    final = torch.cat((row_envl[:,:1, :], btw), dim=1)
    return final

def format_image(t, resolution):
    return t.reshape(1, resolution, resolution, 1).permute(0, 3, 1, 2).expand(-1, 3, -1, -1)    

def normalize_matrix(m):
    min_v = m.min(dim=1).values.unsqueeze(1)
    range_v = m.max(dim=1).values.unsqueeze(1) - min_v
    return (m - min_v) / range_v


def tween(v, s, e):
    return s <= v and v < e

def get_nearest_points(r, c, n, r_max, c_max):
    res = [(r-n, x) for x in range(-n, n+1)]
    for ri in range(r-n+1, r+n):
        res += [(ri, c-n), (ri, c+n)]
    res += [(r+n, x) for x in range(-n, n+1)]
    return [x for x in res 
      if tween(x[0], 0, r_max) and tween(x[1], 0, c_max)]

def mean_nearest(r, c, mesh, val):
    r_max, c_max, _ = mesh.shape
    for n in range(1, max(r_max, c_max)):        
        nearest = [mesh[r, c] for (r, c) 
                   in get_nearest_points(r, c, n, r_max, c_max) 
                   if (mesh[r, c] != val).any()]
        if nearest: return torch.stack(nearest).mean(dim=0)
    return mesh[r, c]
    
    
def fill_with_nearest(mesh, val, inplace=False):
    res = mesh if inplace else mesh.clone()
    rows, cols, _ = mesh.shape
    for r in range(rows):
        for c in range(cols):
            if (mesh[r, c] == val).any():                                
                res[r, c] = mean_nearest(r, c, mesh, val)
    return res

class Sine(torch.nn.Module):
    def forward(self, x): return torch.sin(x)


# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import time


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.


class Lambda(torch.nn.Module):
    "An easy way to create a pytorch layer for a simple `func`."
    def __init__(self, func):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func=func

    def forward(self, x): return self.func(x)

def vertex_tris(faces):
    res = [[] for _ in range(faces.max()+1)]
    for fid, face in enumerate(faces):        
        for vid in face:
            res[vid].append(fid)        
    return res

def vertex_tri_maps(faces):
    vts = vertex_tris(faces)
    r, c = len(vts), max([len(x) for  x in vts])
    vert_tri_indices = torch.zeros(r, c, dtype=torch.long)
    vert_tri_weights = torch.zeros(r, c)    
    for r, tris in enumerate(vts):        
        weight = 1. / len(tris)
        for c, tri_id in enumerate(tris):
            vert_tri_indices[r, c] = tri_id
            vert_tri_weights[r, c] = weight
    return vert_tri_indices, vert_tri_weights.unsqueeze(dim=-1)[None]

def grid_to_list(t):
    return t.reshape(t.size(0), 3, -1).permute(0, 2, 1)

# def make_faces(w, h):
#     mesh_indices = []
#     for iw  in range(w-1):
#         for ih in range(h-1):
#             mesh_indices.append([iw*w+ih, iw*w+ih+1, (iw+1)*w+ih])            
#             mesh_indices.append([iw*w+ih+1, (iw+1)*w+ih+1, (iw+1)*w+ih])
#     return np.array(mesh_indices)   

def create_strips(n, m):
    res = []
    for i in range(n-1):
        strip = []
        for j in range(m):            
            strip.append(j+(i+1)*m)
            strip.append(j+i*m)
            #strip.append(j+(i+1)*m)
        res.append(strip)
    return res

def make_faces(n, m):
    strips = create_strips(n, m)    
    return triangle_strips_to_faces(strips)

def gaussian_kernel(l=5, sig=1.):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    kernel = torch.tensor(kernel / np.sum(kernel)).float()
    return kernel[None][None].expand(3, 3, -1, -1)

def gaussian_conv2d(kernel_size, sigma, padding=1):
    conv = torch.nn.Conv2d(3, 3, kernel_size, padding=padding)
    conv.weight.data = gaussian_kernel(kernel_size, sigma)
    return conv

class __SwishFn__(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
    
class Swish(torch.nn.Module):
    def forward(self, x):
        return __SwishFn__.apply(x)   

def channel_matrix(m, ch_rows, ch_cols):
    (rows, cols) = m.shape
    (rn, cn) = (rows // ch_rows, cols // ch_cols)
    r = torch.zeros(ch_rows, ch_cols, rn, cn)
    assert r.numel() == m.numel()
    for chr_r in range(ch_rows):
        for chr_c in range(ch_cols):
            for i in range(rn):
                for j in range(cn):
                    r[chr_r, chr_c, i, j] = m[i * ch_rows + chr_r, j
                            * ch_cols + chr_c]
    return r.reshape(ch_rows * ch_cols, rn, cn)

def loader_generator(loader):
    current = 0
    iterator = iter(loader)
    while True:
        if current >= len(loader):
            current = 0
            iterator = iter(loader)
        yield next(iterator)
        current += 1  