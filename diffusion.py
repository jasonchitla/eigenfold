import torch
import numpy as np
from torch_geometric.transforms import BaseTransform

class ForwardDiffusionKernel(BaseTransform):
    def __init__(self):
        self.skew = 0.0
        self.Hf = 2.0
        self.rmsd_max = 0.0
        self.tmin = 0.01
        self.kmin = 5
        self.cutoff = 5.0
        self.center = True
        self.key = 'resi'
        
    def __call__(self, data):
        if data.skip: return data
        sde = data.sde
        rsde = data.resi_sde
        
        step = np.random.rand()
        if step > 2/(1+np.exp(abs(self.skew))):
            step = np.random.beta(2, 1) if self.skew > 0 else np.random.beta(1, 2)
            
        rmsd_max = np.interp(self.Hf, rsde.hs[::-1], rsde.rmsds[::-1])
        if self.rmsd_max > 0:
            rmsd_max = min(rmsd_max, self.rmsd_max)
            
        rmsd_min = rsde.rmsd(self.tmin)
        rmsd = rmsd_max * step + rmsd_min * (1-step)
        
        t = np.interp(rmsd, rsde.rmsds, rsde.ts)
        
        k = (sde.D * t < self.cutoff).sum() - 1
        k = min(sde.N-1, max(self.kmin, k))
        
        data.step, data.t, data.k = step, t, k
        data.rmsd = rmsd
        
        # centering and SO(3) alignment
        num_nodes = data[self.key].num_nodes
        data.score_norm = sde.score_norm(t, k, adj=True)
        data[self.key].node_t = torch.ones(num_nodes) * t
        
        pos = data[self.key].pos.numpy()
        
        pos, score = sde.sample(t, pos, center=self.center, score=True, k=k, adj=True)
        pos, score = torch.from_numpy(pos), torch.from_numpy(score)
        
        data[self.key].pos, data[self.key].score = pos, score
        data.score = score
        return data