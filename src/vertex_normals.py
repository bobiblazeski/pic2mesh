import os
from collections import OrderedDict
import torch
import torch.nn.functional as F
from src.util import (
    make_faces,    
)    

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
        weight = 1. #/ len(tris)
        for c, tri_id in enumerate(tris):
            vert_tri_indices[r, c] = tri_id
            vert_tri_weights[r, c] = weight
    return vert_tri_indices, vert_tri_weights.unsqueeze(dim=-1)[None]

class VertexNormals(torch.nn.Module):
    
    def __init__(self, opt, load=True):
        super().__init__()
        self.size = opt.data_patch_size
        self.path = os.path.join(opt.data_dir, 
            'trimap_{}.pth'.format(opt.data_patch_size))
        if load and os.path.exists(self.path):
            trimap = torch.load(self.path)
        else:
            trimap = self.make_trimap(opt.data_patch_size)
            torch.save(trimap, self.path)
        self.assign_trimap(trimap)
    
    def assign_trimap(self,  trimap):
        self.register_buffer('faces',  trimap['faces'])
        self.register_buffer('vert_tri_indices', trimap['vert_tri_indices'])
        self.register_buffer('vert_tri_weights', trimap['vert_tri_weights'])

    def vertex_normals_mean(self, vrt):
        face_normals = self.get_face_normals(vrt)
        bs = face_normals.size(0)
        r, c = self.vert_tri_indices.shape
        fn_group = face_normals.index_select(1, 
            self.vert_tri_indices.flatten()).reshape(bs, r, c, 3)
        weighted_fn_group = fn_group * self.vert_tri_weights    
        vertex_normals = weighted_fn_group.sum(dim=-2)
        return F.normalize(vertex_normals, p=2, dim=-1)
    
    def vertex_normals_weighted_area(self, vrt):
        face_normals = self.get_face_normals(vrt)
        face_areas = self.get_face_areas(vrt)
        bs = face_normals.size(0)
        r, c = self.vert_tri_indices.shape
        fn_group = face_normals.index_select(1, 
            self.vert_tri_indices.flatten()).reshape(bs, r, c, 3)
        
        fa_group = face_areas.index_select(1, 
            self.vert_tri_indices.flatten()).reshape(bs, r, c, 1)
        weighted_fa_group = fa_group * self.vert_tri_weights        
        
        weighted_fn_group = fn_group * fa_group   
        vertex_normals = weighted_fn_group.sum(dim=-2)
        return F.normalize(vertex_normals, p=2, dim=-1)
    
    def get_face_normals(self, vrt):
        faces = self.faces
        v1 = vrt.index_select(1,faces[:, 1]) - vrt.index_select(1, faces[:, 0])
        v2 = vrt.index_select(1,faces[:, 2]) - vrt.index_select(1, faces[:, 0])
        face_normals = F.normalize(v1.cross(v2), p=2, dim=-1)  # [F, 3]
        return face_normals
 
    
    def get_face_areas(self, vrt):
        faces = self.faces

        v0 = vrt.index_select(1, faces[:, 0])
        v1 = vrt.index_select(1, faces[:, 1])
        v2 = vrt.index_select(1, faces[:, 2])

        a = torch.norm(v1 - v0, dim=-1)
        b = torch.norm(v2 - v0, dim=-1)
        c = torch.norm(v2 - v1, dim=-1)

        s = (a + b + c) / 2
        return torch.sqrt(s*(s-a)*(s-b)*(s-c)).unsqueeze(dim=-1)
        
    def __repr__(self):
        return f'VertexNormals: size: {self.size} path: {self.path}'
    
    def make_trimap(self, size):
        faces = torch.tensor(make_faces(size, size))
        vert_tri_indices, vert_tri_weights = vertex_tri_maps(faces)
        return OrderedDict(OrderedDict([
          ('vert_tri_indices', vert_tri_indices),
          ('vert_tri_weights', vert_tri_weights),
          ('faces', faces),
        ]))