import torch
import torch.nn.functional as F
import pytorch3d.transforms as T3

def make_kmat(v):
    kmat = torch.zeros(v.size(0), 3, 3, device=v.device)
    kmat[:, 0, 1] = -v[:, 2]
    kmat[:, 0, 2] =  v[:, 1]

    kmat[:, 1, 0] =  v[:, 2]
    kmat[:, 1, 2] = -v[:, 0]

    kmat[:, 2, 0] = -v[:, 1]
    kmat[:, 2, 1] =  v[:, 0]
    return kmat

def get_rotation_matrix(vec):
    bs, device = vec.size(0), vec.device
    a = torch.tensor([0., 0., 1.], device=device).expand_as(vec)
    b = F.normalize(vec, dim=-1)
    v = torch.cross(a, b, dim=-1)    
    c = (a * b).sum(dim=-1)    
    s = torch.norm(v, dim=-1)    
    kmat = make_kmat(v)    
    eye = torch.eye(3, device=device)[None].expand(bs, -1, -1)    
    mult = ((1- c) / (s ** 2)).reshape(bs, 1, 1)
    rotation_matrix = eye + kmat + torch.bmm(kmat, kmat) * mult
    return rotation_matrix

def get_min_max(points):
    return (points.min(dim=1).values,
            points.max(dim=1).values)


def align(points, normals, rotate=True):
    device = points.device
    if rotate:
      # Rotate points
      vec = F.normalize(normals.mean(dim=1), dim=-1)
      rot_mat = get_rotation_matrix(vec)
      RT = T3.Rotate(rot_mat, device=device)
      t_points = RT.transform_points(points)
      t_normals = RT.transform_normals(normals)
    else:
       t_points = points

    # Translate points
    tm = t_points.mean(dim=-2, keepdim=False)
    T = T3.Translate(-tm, device=points.device)            
    t_points = T.transform_points(t_points)

    # Scale points
    min_vals, max_vals = get_min_max(t_points)
    tm = (min_vals + max_vals) / 2
    sm = torch.zeros_like(tm) + (1. / (max_vals - min_vals)[:, :2]).min()        
    ST = T3.Scale(sm, device=points.device)
    t_points = ST.transform_points(t_points)
    t_normals = ST.transform_normals(normals)

    return t_points, t_normals