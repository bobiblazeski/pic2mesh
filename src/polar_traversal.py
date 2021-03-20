import numpy as np
import trimesh
import open3d as o3d
import torch
import torch.nn.functional as F

import math

from shapely.geometry import LineString, Point

# np.set_printoptions(precision=4, suppress=True)
# torch.set_printoptions(precision=4, sci_mode=False)

def scale_mesh(stl_path, offset=0.02):
    mesh = trimesh.load(stl_path)    
    vertices = torch.from_numpy(mesh.vertices).float()
    vertices = vertices / vertices.abs().max() * (1.-offset)        
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

def flip_mesh(mesh):
    vertices = np.array(mesh.vertices)
    switched = np.vstack([vertices[:, 1], vertices[:, 2],  vertices[:, 0]]).transpose()
    return trimesh.Trimesh(vertices=switched, faces=np.array(mesh.faces))

def get_start(mesh):
    ray_origins = np.array([[0, 0, 2],])
    ray_directions = np.array([[0, 0, -1]])

    locations, _, index_tri = mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions)
    top_face_idx = locations[:, 2].argmax()
    start_face = index_tri[top_face_idx] 
    start_point =  locations[top_face_idx]
    start_normal = np.array([0, 0, +1])
    assert start_point.shape == start_normal.shape, (start_point, start_normal)
    return start_face, start_point, start_normal

class Bridge:

    def __init__(self, path):
        self.path = path
        self.mesh = flip_mesh(scale_mesh(path))
        self.mesh.fix_normals()
        self.faces = self.make_faces()
        self.polars = math.pi + np.array(
            np.arctan2(self.mesh.vertices[:, 1], 
                       self.mesh.vertices[:, 0]))
        self.edge_to_face = self.make_edge_to_face()

    def make_faces(self):
        faces = np.array(self.mesh.faces)
        faces.sort(axis=1)
        return faces
    
    def make_edge_to_face(self):        
        edge_to_face = {}
        for i, face in enumerate(self.faces):
            edges = self.get_face_edges(face)
            for edge in edges:
                if edge in edge_to_face:
                    edge_to_face[edge].append(i)
                else:
                    edge_to_face[edge] = [i]
        return edge_to_face
    
    def get_face_edges(self, face):
        return [(face[0], face[1]),
                (face[0], face[2]),
                (face[1], face[2]),]
        
    def get_face_ids(self, edge, past_face_ids):        
        if edge in self.edge_to_face:
            face_ids = self.edge_to_face[edge]
            
            res =  [f for f in face_ids if f not in past_face_ids]
            #print('get_face_ids', face_ids, past_face_ids, res)
            return res
        return []    
    
    def next_edges_face_id(self, edge, past_face_ids, past_edges):
        face_ids = self.get_face_ids(edge, past_face_ids)
        face_id, edges = None, []
        if face_ids:
            face_id = face_ids[0] # Pick first face             
            face =  self.faces[face_id]
            face_edges = self.get_face_edges(face)
            for face_edge in face_edges:
                if face_edge not in past_edges:
                    edges.append(face_edge)                
        return face_id, edges
            
        
    def __repr__(self):
        return f'Path: {self.path}'
        
    def scale_mesh(self, stl_path, offset=0.1):
        mesh = trimesh.load(stl_path)    
        vertices = torch.from_numpy(mesh.vertices).float()
        vertices = vertices / vertices.abs().max() * (1.-offset)        
        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
    
def get_intersection_point(latitude, edge_vertex, edge_vertex_normals):
    ev_2d  = edge_vertex[:, :2]
    e1_pt, e2_pt = Point(ev_2d[0]),  Point(ev_2d[1])
    
    l1 = LineString(latitude)
    l2 = LineString(ev_2d)
    if l1.intersects(l2):
        pt = l1.intersection(l2)        
        xy = np.array(pt.xy)
        d1 = e1_pt.distance(pt)
        d2 = e1_pt.distance(e2_pt)
        assert d1 <= d2,  ('d1 > d2',  pt, d1, d2)
        ratio = d1 / d2 if d2 != 0 else 0.5
        xyz = ratio * edge_vertex[0] + (1-ratio) *  edge_vertex[1]
        normal =  ratio * edge_vertex_normals[0] + (1-ratio) *  edge_vertex_normals[1]
        point, normal =  np.array(xyz), np.array(normal)
        assert point.shape == normal.shape, (edge_vertex, edge_vertex_normals)
        return point, normal
        
        
    return None, None
   
def it_intersects(edge, latitude, bridge):
    edge_sel = np.array(edge)    
    edge_vertex =  bridge.mesh.vertices[edge_sel]        
    edge_vertex_normals =  bridge.mesh.vertex_normals[edge_sel]            
    return get_intersection_point(latitude, edge_vertex, edge_vertex_normals)
    

def make_latitudes(n):
    r_angle = torch.stack((
        torch.zeros(n) + 2,
        torch.arange(0, n) * (2 * math.pi / n)))
    xy = torch.stack((r_angle[0]*torch.cos(r_angle[1]),
                      r_angle[0]*torch.sin(r_angle[1])))#.t()
    latitudes = torch.stack((torch.zeros_like(xy), xy))
    return latitudes.permute(2, 0, 1).numpy()    

def get_intersection(latitude, edges, bridge):
    #print('get_intersection', edges)
    for edge in edges:
        edge_line = bridge.mesh.vertices[np.array(edge)]                
        point, normal = it_intersects(edge, latitude, bridge)
        if point is not None:            
            return point, normal,  edge    
    return None, None, None
    
def gather_path(latitude, face_id, point, normal, bridge):    
    face = bridge.faces[face_id]
    edges = bridge.get_face_edges(face)
    path = { 'points': [point], 'normals': [normal]}    
    point, normal, edge = get_intersection(latitude, edges, bridge)        
    path['points'].append(point) 
    path['normals'].append(normal) 
    past_face_ids, past_edges = [face_id], edges.copy()
    while True:        
        face_id, edges = bridge.next_edges_face_id(edge, past_face_ids, past_edges)
        #print(past_edges)
        if face_id is not None:
            past_face_ids.append(face_id)
            past_edges = past_edges + edges
            point, normal, edge = get_intersection(latitude, edges, bridge)
            if point is not None:                
                path['points'].append(point) 
                path['normals'].append(normal) 
            else: 
                break
        else:
            break
    path['points'] =  np.array(path['points'])
    path['normals'] =  np.array(path['normals'])
    assert path['points'].shape == path['normals'].shape, path
    return path

def gather_all_paths(stl_file, latitudes_num):
    latitudes = make_latitudes(latitudes_num)
    bridge = Bridge(stl_file)  
    face_id, point, normal = get_start(bridge.mesh)
    #print('gather_all_paths', point, normal, point.shape, normal.shape)
    paths = []
    for latitude in latitudes:
        #print(latitude)
        path = gather_path(latitude, face_id, point, normal, bridge)
        paths.append(path)
    return paths

def cumulative_distances(path):
    cumulative = [0]
    for i in range(0, len(path)-1):        
        dist = np.linalg.norm(path[i] - path[i+1])
        cumulative.append(dist+cumulative[-1])
    cumulative = np.array(cumulative)
    return cumulative

def get_sample(pos, normed, path_points, path_normals):    
    i = 1
    while i < len(normed):
        d0, d1 = normed[i-1], normed[i]
        if pos <= d1:
            ratio = (pos - d0) / (d1 - d0) 
            pnt = ratio * path_points[i-1] + (1 - ratio) * path_points[i]
            nrm = ratio * path_normals[i-1] + (1 - ratio) * path_normals[i]
            return pnt, nrm, i
        i += 1
    raise (pos, normed, path)
    
def sample_path(path, samples_num):
    cumulative = cumulative_distances(path['points'])
    normed =  cumulative / cumulative[-1]
    step = 1. / samples_num
    position = np.arange(0, 1, step) + step
    samples = {'points': [], 'normals': []}
    start = 0
    for pos in position:
        pnt, nrm, start = get_sample(pos, normed[start:], path['points'][start:],
                               path['normals'][start:])
        samples['points'].append(pnt)
        samples['normals'].append(nrm)
        
    samples['points'] =  np.array(samples['points'])
    samples['normals'] =  np.array(samples['normals'])
    assert samples['points'].shape == samples['normals'].shape,  \
        "Points {} /== Normals {}".format(samples['points'].shape, samples['normals'].shape)
    return samples

def sample_all_paths(paths, samples_num):
    all_samples = []
    for path in paths:
        path_samples =  sample_path(path, samples_num)        
        #print(path_samples['points'].shape, path_samples['normals'].shape)
        all_samples.append(path_samples)
        
    #for p in  all_samples:
    #    print(p['points'].shape, p['normals'].shape)
    res_points = np.array([p['points'] for p in all_samples])
    res_normals = np.array([p['normals'] for p in all_samples])

    assert res_points.shape ==  res_normals.shape
    n, h = len(paths) // 2, len(res_points) // 2
    # # Concatenate opposite angles 0-180, 10:190, ...170:350
    return (
        np.concatenate((np.flip(res_points[h:], axis=1), res_points[:h]), axis=1),
        np.concatenate((np.flip(res_normals[h:], axis=1), res_normals[:h]), axis=1)
    )
