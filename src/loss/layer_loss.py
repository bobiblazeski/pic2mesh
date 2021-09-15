# pyright: reportMissingImports=false
import lpips
import torch

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)

class LayerLoss(torch.nn.Module):
    """ Initializes a layered loss
        Parameters (default listed first)
        ---------------------------------                
        net : str
            ['alex','vgg','squeeze'] are the base/trunk networks available
        layers : None or list
            None all layers will be used
            -alex ['relu1', 'relu2', 'relu3', 'relu4', 'relu5']
            -vgg ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
            -squeeze ['relu1', 'relu2', 'relu3', 'relu4', 'relu5', 'relu6', 'relu7']
        use_dropout : bool
            [True] to use dropout when training linear layers
            [False] for no dropout when training linear layers            
    """
    def __init__(self, net, layers=None, use_dropout=True):
        super(LayerLoss, self).__init__()
        assert net in ['alex','vgg','squeeze'] 
        self.net = lpips.LPIPS(pretrained=True, net=net, lpips=False, 
                               use_dropout=use_dropout).net
        self.layers = layers

    def forward(self, in0, in1,  layers=None):
        layers = layers or self.layers
        outs0, outs1 = self.net(in0), self.net(in1)
        feats0, feats1, diffs = {}, {}, {}

        for kk, key in enumerate(outs0._asdict().keys()):
            if layers is not None and key not in layers:
                continue
            feats0[kk] = lpips.normalize_tensor(outs0[kk])
            feats1[kk] = lpips.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk]-feats1[kk])**2

        res = torch.stack([
            spatial_average(diffs[kk].sum(dim=1,keepdim=True)).sum()
            for kk in diffs.keys()])        
        return res.mean()