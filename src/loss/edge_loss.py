import torch

from pytorch3d.structures import Meshes
from src.utilities.util import make_faces

class EdgeLoss(torch.nn.Module):
    def __init__(self, config):
        super(EdgeLoss, self).__init__()
        self.patch_size = config.data_patch_size
        faces = torch.tensor(make_faces(self.patch_size, self.patch_size))
        vertices = torch.rand(self.patch_size ** 2, 3)
        meshes = Meshes(verts=[vertices], faces=[faces])
        self.no_edges = max(meshes.edges_packed().shape)
        edges_packed = meshes.edges_packed()       
        self.register_buffer('v0',  edges_packed[:, 0])
        self.register_buffer('v1',  edges_packed[:, 1])

    def forward(self, vertices, target_length=0):
        bs = vertices.size(0)
        no_edges = self.no_edges * bs        
        v0 = vertices.index_select(1, self.v0)
        v1 = vertices.index_select(1, self.v1)
        loss = ((v0 - v1).norm(dim=1, p=2) - target_length) ** 2.0
        return loss.sum() / no_edges