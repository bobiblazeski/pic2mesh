# pyright: reportMissingImports=false
from math import (atan2, pi, sqrt)    

from collections import OrderedDict

import os
import numpy as np
import torch
import torch.nn.functional as F

torch.set_printoptions(sci_mode=False)

from src.utilities.polar_traversal import (
    Bridge,
    make_latitudes,
    get_start,
    gather_path,
    cumulative_distances,
)


def get_dist(x, y):
    return sqrt((x-0.5) **2 + (y-0.5) ** 2)

def get_angle(x, y, square_no):
    side = (square_no + 1) * 2
    corr = (side -1) / 2    
    angle = atan2(y -corr, x -corr)
    return angle if angle >= 0 else angle + 2 * pi

def get_angles_series(n):
    return [x for x in range(0, n*2+1, 2)]


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
    raise (pos, normed, path_points, path_normals)

def single_path_sample(path, layer_no, total_layers):
    cumulative = cumulative_distances(path['points'])
    normed =  cumulative / cumulative[-1]
    pos = layer_no / total_layers    
    pnt, nrm, _ = get_sample(pos, normed, 
                             path['points'], path['normals'])
    return torch.tensor(pnt).float(), torch.tensor(nrm).float()

def square_indices(n):
    max_idx = 2 * (n+1)
    res = [[0, x] for x in range(max_idx -1, -1, -1)]\
        + [[x, 0] for x in range(1, max_idx)]\
        + [[max_idx -1, x] for x in range(1, max_idx)]\
        + [[x, max_idx -1] for x in range(max_idx -2, 0, -1)]
    return res

def shifted_square_indices(sq_no, square_no):    
    sq_ids =  square_indices(sq_no)
    ost = square_no - sq_no
    return [(i+ost, k+ost) for (i, k) in iter(sq_ids)]

def get_latitudes_angles(n):
    return (torch.arange(0, n) * (2 * pi / n)).tolist()

class Blueprint:
    
    def __init__(self, path, latitudes_num, offset=0.2):
        self.path = path
        self.latitudes_num = latitudes_num
        self.bridge = Bridge(path, offset=offset)
        paths_path =  f'./data/paths_{latitudes_num}.pth'        
        if os.path.exists(paths_path):
            self.paths = torch.load(paths_path) 
        else:
            self.paths = self.gather_paths(latitudes_num)       
    
    def gather_paths(self, latitudes_num=None):
        latitudes_num = latitudes_num or self.latitudes_num  
        latitude_angles = get_latitudes_angles(latitudes_num)        
        latitudes = make_latitudes(latitudes_num)
        bridge = self.bridge
        face_id, point, normal = get_start(bridge.mesh)
        paths = OrderedDict([])
        for i, (latitude, latitude_angle) in enumerate(zip(latitudes, latitude_angles)):
            print(i, latitudes_num)
            path = gather_path(latitude, face_id, point, normal, bridge)
            paths[latitude_angle] = path
        return paths
        
    def __repr__(self):
        return f'Blueprint: {self.path} latitudes_num: {self.latitudes_num}' 

    def get_border_paths(self, angle):        
        assert angle >=0 and angle < 2 * pi
        paths = self.paths
        angle_keys = list(paths.keys())
        for i, angle_key in enumerate(angle_keys):        
            if angle < angle_key:            
                return (paths[angle_keys[i % len(angle_keys)]],
                        paths[angle_keys[(i+1) % len(angle_keys)]])
        return (paths[angle_keys[0]], paths[angle_keys[-1]])

    def grid_sample(self, square_no, save_path=None):
        side = (square_no + 1) * 2
        res = self.load_result(save_path) \
            if save_path is not None and os.path.exists(save_path) \
            else self.create_result(side)
        
        face_id, point, normal = get_start(self.bridge.mesh)
        for sq_no in range(square_no+1):
            print(sq_no, res['sq_no'][0])
            if sq_no > res['sq_no'][0]:                                
                side = (sq_no + 1) * 2
                sq_ids = shifted_square_indices(sq_no, square_no)            
                angles_no = len(sq_ids)                          
                for (r, c) in sq_ids:
                    angle = get_angle(r, c, square_no)                    
                    path0, path1 = self.get_border_paths(angle)
                    pnt0, nrm0 = single_path_sample(path0, sq_no, square_no)
                    pnt1, nrm1 = single_path_sample(path1, sq_no, square_no)
                    pnt = (pnt0 + pnt1)/2
                    nrm = F.normalize((nrm0 + nrm1)/2, dim=0)
                    res['points'][0, :, c, r] = pnt
                    res['normals'][0, :, c, r] = nrm                
                    res['grid'][c, r] = float(sq_no)                
                res['sq_no'][0] = sq_no
                self.save_result(res, save_path)
        return res            
        
    def create_result(self, side):
        return {
            'points': torch.zeros(1, 3, side, side),
            'normals': torch.zeros(1, 3, side, side),
            'grid': torch.zeros(side, side) - 9,
            'sq_no': torch.tensor([-1]),
        }
    
    def load_result(self, path):
        loaded = np.load(path)
        
        return {
            'points': torch.tensor(loaded['points']).float(),
            'normals': torch.tensor(loaded['normals']).float(),
            'grid': torch.tensor(loaded['grid']).float(),
            'sq_no': torch.tensor(loaded['sq_no']).long(),
        }
    
    def save_result(self, res, path=None):
        #assert type('s')  == 'str'
        if path is not None:            
            print('path', path, res['sq_no'])
            np.savez_compressed(
                path,
                points=res['points'].numpy(),
                normals=res['normals'].numpy(),
                grid=res['grid'].numpy(),
                sq_no=res['sq_no'].numpy(),
            )
            