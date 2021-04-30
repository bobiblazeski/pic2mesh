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

def vertex_angle_maps(faces):
    vrt_no =  faces.max() + 1 
    angle_vrt_idx = torch.zeros(vrt_no, 6, 3, dtype=torch.long) -1
    #angle_vrt_idx = 
    for face in faces:
        v0, v1, v2 = face
        for i, m in enumerate(angle_vrt_idx[v0]):
            if m[0].item() == -1:
                angle_vrt_idx[v0, i, 0] = v1
                angle_vrt_idx[v0, i, 1] = v0
                angle_vrt_idx[v0, i, 2] = v2
                break
        for i, m in enumerate(angle_vrt_idx[v1]):
            if m[0].item() == -1:
                angle_vrt_idx[v1, i, 0] = v0
                angle_vrt_idx[v1, i, 1] = v1
                angle_vrt_idx[v1, i, 2] = v2
                break
        for i, m in enumerate(angle_vrt_idx[v2]):
            if m[0].item() == -1:
                angle_vrt_idx[v2, i, 0] = v0
                angle_vrt_idx[v2, i, 1] = v2
                angle_vrt_idx[v2, i, 2] = v1
                break
    angle_vrt_wt = torch.where(angle_vrt_idx.sum(dim=-1) != -3, 1., 0.)
    angle_vrt_wt = angle_vrt_wt[None].unsqueeze(-1)
    angle_vrt_idx = torch.where(angle_vrt_idx > 0, angle_vrt_idx, 0)
    return angle_vrt_idx, angle_vrt_wt


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
        self.register_buffer('angle_vrt_idx', trimap['angle_vrt_idx'])
        self.register_buffer('angle_vrt_wt', trimap['angle_vrt_wt'])

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

    def vertex_normals_fast(self, vrt):
        face_normals = self.get_face_normals(vrt, normalized=False)        
        bs = face_normals.size(0)
        r, c = self.vert_tri_indices.shape
        fn_group = face_normals.index_select(1, 
            self.vert_tri_indices.flatten()).reshape(bs, r, c, 3)
        weighted_fn_group = fn_group * self.vert_tri_weights    
        vertex_normals = weighted_fn_group.sum(dim=-2)
        return F.normalize(vertex_normals, p=2, dim=-1)
    
    def vertex_normals_weighted_angles(self, vrt):
        face_normals = self.get_face_normals(vrt)
        vertex_angles = self.get_vertex_angles(vrt)
        bs = face_normals.size(0)
        r, c = self.vert_tri_indices.shape
        fn_group = face_normals.index_select(1, 
            self.vert_tri_indices.flatten()).reshape(bs, r, c, 3)

        weighted_fn_group = fn_group * vertex_angles   
        vertex_normals = weighted_fn_group.sum(dim=-2)
        return F.normalize(vertex_normals, p=2, dim=-1)
    
    def get_face_normals(self, vrt, normalized=True):
        faces = self.faces
        v1 = vrt.index_select(1,faces[:, 1]) - vrt.index_select(1, faces[:, 0])
        v2 = vrt.index_select(1,faces[:, 2]) - vrt.index_select(1, faces[:, 0])
        face_normals = v1.cross(v2) # [F, 3]
        if normalized:
            face_normals = F.normalize(face_normals, p=2, dim=-1) # [F, 3]
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
    
    def get_vertex_angles(self, vrt):        
        bs = vrt.size(0)
        angle_pts = vrt.index_select(1, self.angle_vrt_idx.view(-1))
        angle_pts = angle_pts.reshape(bs, -1, 6, 3, 3)
        a = angle_pts[:, :, :, 0]
        b = angle_pts[:, :, :, 1]
        c = angle_pts[:, :, :, 2]

        ba = a - b
        bc = c - b

        ba_nrm = torch.norm(ba, dim=-1).unsqueeze(-1)
        bc_nrm = torch.norm(bc, dim=-1).unsqueeze(-1)
        one = torch.tensor(1.).to(vrt.device)
        ba_nrm = torch.where(ba_nrm > 0, ba_nrm, one)
        bc_nrm = torch.where(bc_nrm > 0, bc_nrm, one)

        ba_normed = ba / ba_nrm
        bc_normed = bc / bc_nrm
        dot_bac = (ba_normed * bc_normed).sum(dim=-1).unsqueeze(-1)
        angles = torch.arccos(dot_bac) * self.angle_vrt_wt
        return angles
        
    def __repr__(self):
        return f'VertexNormals: size: {self.size} path: {self.path}'
    
    def make_trimap(self, size):
        faces = torch.tensor(make_faces(size, size))
        vert_tri_indices, vert_tri_weights = vertex_tri_maps(faces)
        angle_vrt_idx, angle_vrt_wt = vertex_angle_maps(faces)
        return OrderedDict(OrderedDict([
          ('vert_tri_indices', vert_tri_indices),
          ('vert_tri_weights', vert_tri_weights),
          ('faces', faces),
          ('angle_vrt_idx', angle_vrt_idx),
          ('angle_vrt_wt', angle_vrt_wt),
        ]))
    