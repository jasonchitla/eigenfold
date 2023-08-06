from functools import lru_cache
import numpy as np
import torch, os
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader
from .utils import get_logger
logger = get_logger(__name__)
from .pdb.pdb import pdb_to_npy
from .diffusion import ForwardDiffusionKernel

class ResidueDataset(Dataset):
    def __init__(self, split, inference_mode=False, **kwargs):
        super(ResidueDataset, self).__init__(**kwargs)
        self.inference_mode = inference_mode
        self.split = split
        
    @lru_cache(maxsize=None)
    def get_sde(self, i):
        # To enforce a RMS distance of 3.8 angstrom between adjacent alpha carbons
        # 3/(3.8**2)
        sde_a = 0.20775623268698062
        sde_b = 0.0
        sde = PolymerSDE(N=i, a=sde_a, b=sde_b)
        sde.make_schedule(Hf=2.0, step=0.5, tmin=0.01)
        return sde
        
    def len(self):
        return len(self.split)

    def null_data(self, data):
        data.skip = True
        return data
        
    def get(self, idx):
        if type(self.split) == list:
            subdf = self.split[idx][1]
            row = subdf.loc[np.random.choice(subdf.index)]
        else:
            row = self.split.iloc[idx]       
    
        data = HeteroData()
        data.skip = False
        data['resi'].num_nodes = row.seqlen

        atom_ids = np.arange(row.seqlen)
        src, dst = np.repeat(atom_ids, row.seqlen), np.tile(atom_ids, row.seqlen)
        mask = src != dst
        src, dst = src[mask], dst[mask]
        edge_idx = np.stack([src, dst])
        data['resi'].edge_index = torch.tensor(edge_idx)
        data.resi_sde = data.sde = self.get_sde(row.seqlen)
        data.path = pdb_path = os.path.join('./data/pdb_chains', row.name[:2], row.name); data.info = row
        
        
        ret = pdb_to_npy(pdb_path, seqres=row.seqres)
        if not self.inference_mode and ret is None:
            logger.warning(f"Error loading {pdb_path}")
            return self.null_data(data)
        elif ret is not None:
            pos, mask = ret
            pos[~mask,0] = data.sde.conditional(mask, pos[mask,0])
            data['resi'].pos = torch.tensor(pos[:,0]).float()
        
        embeddings_name = row.__getattr__('name' if self.inference_mode else 'reference')
        embeddings_path = os.path.join('./embeddings-limit256', embeddings_name[:2], embeddings_name) + '.omegafold_num_recycling.4.npz'
        if not os.path.exists(embeddings_path):
            logger.warning(f"No LM embeddings at {embeddings_path}")
            return self.null_data(data)
            
        try:
            embeddings_dict = dict(np.load(embeddings_path))
            node_repr, edge_repr = embeddings_dict['node_repr'], embeddings_dict['edge_repr']
        except:
            logger.warning(f"Error loading {embeddings_path}")
            return self.null_data(data)
        
        if node_repr.shape[0] != data['resi'].num_nodes:
            logger.warning(f"LM dim error at {embeddings_path}: expected {data['resi'].num_nodes} got {node_repr.shape} {edge_repr.shape}")
            return self.null_data(data)
            
        data['resi'].node_data = torch.tensor(node_repr)
        edge_repr = torch.tensor(edge_repr)
        src, dst = data['resi'].edge_index[0], data['resi'].edge_index[1]
        data['resi'].edge_data_extended = torch.cat([edge_repr[src, dst], edge_repr[dst, src]], -1)
        
        return data
    
def get_loader(splits, inference_mode=False, mode='train', shuffle=True):
    try:
        split = splits[splits.split == mode]
    except:
        split = splits
        logger.warning("Not splitting based on split")
    
    if 'seqlen' not in split.columns:
        split['seqlen'] = [len(s) for s in split.seqres]

    # max seqlen of 1500
    split = split[split.seqlen <= 1500]
    
    transform = ForwardDiffusionKernel()
    dataset = ResidueDataset(split=split, inference_mode=inference_mode, transform=transform)
        
    logger.info(f"Initialized {mode if mode else ''} loader with {len(dataset)} entries")
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=shuffle, pin_memory=False, num_workers=4)
    
    return loader
