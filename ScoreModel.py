import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter, scatter_max
from .utils import sinusoidal_embedding, GaussianSmearing
from e3nn import o3
from e3nn.nn import BatchNorm

""" performs message passing with attention """
class ConvLayer(torch.nn.Module):
    def __init__(self, in_irreps, in_tp_irreps, out_tp_irreps,
                 spherical_harmonics_irreps, out_irreps, n_edge_features):
        super(ConvLayer, self).__init__()
        
        self.input_linear = o3.Linear(in_irreps, in_tp_irreps, internal_weights=True)
        # In a fully connected tensor product, all paths that lead to any of the irreps specified in out_tp_irreps are created
        self.tensor_product_value = o3.FullyConnectedTensorProduct(in_tp_irreps, spherical_harmonics_irreps, out_tp_irreps, shared_weights=False)      
        self.output_linear = o3.Linear(out_tp_irreps, out_irreps, internal_weights=True)

        key_irreps = [(mul//2, ir) for mul, ir in in_tp_irreps]
        self.query_transform = o3.Linear(in_tp_irreps, key_irreps)
        self.tensor_product_key = o3.FullyConnectedTensorProduct(in_tp_irreps, spherical_harmonics_irreps, key_irreps, shared_weights=False)
        fc_dim = 128
        self.fc_key = nn.Sequential(
            nn.Linear(n_edge_features, fc_dim), nn.ReLU(), nn.Linear(fc_dim, fc_dim),
            nn.ReLU(), nn.Linear(fc_dim, self.tensor_product_key.weight_numel)
        )
        self.dot = o3.FullyConnectedTensorProduct(key_irreps, key_irreps, "0e")

        self.fc_value = nn.Sequential(
            nn.Linear(n_edge_features, fc_dim), nn.ReLU(), nn.Linear(fc_dim, fc_dim),
            nn.ReLU(), nn.Linear(fc_dim, self.tensor_product_value.weight_numel)
        )
        self.batch_norm = BatchNorm(out_irreps)

    
    def forward(self, node_data, edge_index, edge_data, edge_spherical_harmonics):
        transformed_node_data = self.input_linear(node_data)
        edge_source, edge_dest = edge_index
        out_nodes = node_data.shape[0]
        def ckpt_forward(node_data, edge_source, edge_dest, edge_spherical_harmonics, edge_data):
            q = self.query_transform(node_data)
            k = self.tensor_product_key(node_data[edge_source], edge_spherical_harmonics, self.fc_key(edge_data))
            v = self.tensor_product_value(node_data[edge_source], edge_spherical_harmonics, self.fc_value(edge_data))
            a = self.dot(q[edge_dest], k)

            #maximizes all values in a at edge_dst
            max_ = scatter_max(a, edge_dest, dim=0, dim_size=out_nodes)[0]
            a = (a - max_[edge_dest]).exp()
            z = scatter(a, edge_dest, dim=0, dim_size=out_nodes)
            a = a / z[edge_dest]
            return scatter(a * v, edge_dest, dim=0, dim_size=out_nodes)
        
        if self.training:        
            out = torch.utils.checkpoint.checkpoint(ckpt_forward,
                    transformed_node_data, edge_source, edge_dest, edge_spherical_harmonics, edge_data)
        else:
            out = ckpt_forward(transformed_node_data, edge_source, edge_dest, edge_spherical_harmonics, edge_data)
        
        out = self.output_linear(out)
        out = out + F.pad(node_data, (0, out.shape[-1] - node_data.shape[-1])) 
        out = self.batch_norm(out)
        return out

class ScoreModel(torch.nn.Module):
    def __init__(self, embed_dims, num_conv_layers, position_embed_dims):
        super(ScoreModel, self).__init__()

        self.position_embed_dims = position_embed_dims
        self.embed_node_func = (lambda x : sinusoidal_embedding(x, embed_dims, max_positions=10000))
        """ Each spherical harmonic is associated with a degree 'l' (non-negative integer)
            The degree 'l' gives you a sense of the complexity of the pattern on the sphere.
            A spherical harmonic of degree 0 is just a constant. A spherical harmonic of degree 1 is
            a simple dipole pattern, with one 'positive' region and one 'negative' region.
            We use degree 2, seen here below. """
        self.spherical_harmonics_irreps = o3.Irreps.spherical_harmonics(lmax=2)
        
        node_dims = 256
        bottleneck_dims = 32
        self.node_embedding_transform = nn.Sequential(
            nn.Linear(embed_dims + node_dims, bottleneck_dims),
            nn.ReLU(),
            nn.Linear(bottleneck_dims, bottleneck_dims),
            nn.ReLU(),
            nn.Linear(bottleneck_dims, bottleneck_dims)
        )
        edge_dims = 128
        num_gaussians = 50
        self.edge_embedding_transform = nn.Sequential(
            nn.Linear(embed_dims + num_gaussians + self.position_embed_dims + 2 * edge_dims, bottleneck_dims),
            nn.ReLU(),
            nn.Linear(bottleneck_dims, bottleneck_dims),
            nn.ReLU(),
            nn.Linear(bottleneck_dims, bottleneck_dims)
        )
        self.node_norm = nn.LayerNorm(node_dims)
        self.edge_norm = nn.LayerNorm(2 * edge_dims)

        # can alter stop max
        self.distance_expansion = GaussianSmearing(0.0, 50**0.5, num_gaussians)
        
        conv_layers = []
        irrep_seq = [
            [(0, 1)],
            [(0, 1), (1, -1)],
            [(0, 1), (1, -1), (1, 1)],
            [(0, 1), (1, -1), (1, 1), (0, -1)]
        ]
        def generate_irreps(ns, nv, irs):
            irreps = [(ns, (l, p)) if (l == 0 and p == 1) else [nv, (l, p)] for l, p in irs]
            return irreps
        
        # as we add more layers, more complex spherical harmonics are considered
        for i in range(num_conv_layers):
            ns = bottleneck_dims
            nv = 4
            ntps = 16
            ntpv = 4
            in_seq, out_seq = irrep_seq[min(i, len(irrep_seq) - 1)], irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            in_irreps = generate_irreps(ns, nv, in_seq)
            out_irreps = generate_irreps(ns, nv, out_seq)
            in_tp_irreps = generate_irreps(ntps, ntpv, in_seq)
            out_tp_irreps = generate_irreps(ntps, ntpv, out_seq)
            layer = ConvLayer(
                in_irreps=in_irreps,
                spherical_harmonics_irreps=self.spherical_harmonics_irreps,
                in_tp_irreps=in_tp_irreps,
                out_tp_irreps=out_tp_irreps,
                out_irreps=out_irreps,
                n_edge_features=3*ns
            )
            conv_layers.append(layer)
                
        self.conv_layers = nn.ModuleList(conv_layers)
        # need paths that lead to irreps specified in 1x1o + 1x1e
        self.final_tensor_product = o3.FullyConnectedTensorProduct(out_irreps, out_irreps, '1x1o + 1x1e', internal_weights=True)

