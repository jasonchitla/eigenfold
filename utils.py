import math, torch
from torch.nn import functional as F
import logging, socket
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict

def sinusoidal_embedding(timesteps, embedding_dim, max_positions=10000):
    """ from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py   """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2)).float()

def get_logger(name, level='info'):
    logger = logging.Logger(name)
    level = {
        'crititical': 50,
        'error': 40,
        'warning': 30,
        'info': 20,
        'debug': 10
    }[level]
    logger.setLevel(level)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(f'%(asctime)s [{socket.gethostname()}:%(process)d] [%(levelname)s] %(message)s') 
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def save_loss_plot(log, path):
    x = np.array(log['rmsd'])
    y1 = np.array(log['loss'])
    y2 = np.array(log['base_loss'])
    order = np.argsort(x)
    x, y1, y2 = x[order], y1[order], y2[order]
    plt.scatter(x, y1, c='blue')
    plt.scatter(x, y2, c='orange')
    plt.plot(x, gaussian_filter1d(y1, 1), c='blue')
    plt.plot(x, gaussian_filter1d(y2, 1), c='orange')
    plt.ylim(bottom=0)
    plt.savefig(path)
    plt.clf()

class ActivationStats(torch.nn.Module):
    def __init__(self):
        super(ActivationStats, self).__init__()
        self.stats = defaultdict(lambda: {"means": [], "stds": []})

    def hook_fn(self, module, input, output):
        if not module.training:
            self.stats[module]["means"].append(output.data.mean().cpu())
            self.stats[module]["stds"].append(output.data.std().cpu())

    def register_to(self, module):
        if isinstance(module, torch.nn.Sequential):
            for sub_module in module:
                sub_module.register_forward_hook(self.hook_fn)
        else:
            module.register_forward_hook(self.hook_fn)