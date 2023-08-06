import torch, wandb
import numpy as np
from .utils import get_logger
logger = get_logger(__name__)

def loss_func(data):
    loss = ((data.score - data.pred)**2 / data.score_norm[:,None]**2).mean()
    base_loss = (data.score**2 / data.score_norm[:,None]**2).mean()    
    return loss, base_loss

def get_scheduler(optimizer):
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=1.0, total_iters=10000)
    constant = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1., total_iters=100000)
    decay = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=1.0, total_iters=500000)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, constant, decay], milestones=[10000, 110000])
    
def epoch(model, loader, optimizer=None, scheduler=None, device='cpu', print_freq=1000):
    if optimizer is not None: model.train()
    else: model.eval()
    
    log = {'rmsd': [], 'step': [], 'loss': [], 'base_loss': []}
    for i, data in enumerate(loader):
        data = data.to(device)
        try:
            if data.skip:
                logger.warning(f"Skipping batch")
                continue
            data, loss, base_loss = iter_(model, data, optimizer)
            if scheduler: scheduler.step()
            
            with torch.no_grad():
                log['rmsd'].append(float(data.rmsd.cpu().numpy()))
                log['step'].append(float(data.step.cpu().numpy()))
                log['loss'].append(float(loss.cpu().numpy()))
                log['base_loss'].append(float(base_loss.cpu().numpy()))

        except RuntimeError as e:
            if 'out of memory' in str(e):
                path = [d.path for d in data] if type(data) is list else data.path
                logger.warning(f'CUDA OOM, skipping batch {path}')
                for p in model.parameters():
                    if p.grad is not None: del p.grad  
                torch.cuda.empty_cache()
                continue
            
            else:
                logger.error("Uncaught error " + str(e))
                #raise e
                
        if (i+1) % print_freq == 0:
            logger.info(f"Last {print_freq} iters: loss {np.mean(log['loss'][-print_freq:])} base {np.mean(log['base_loss'][-print_freq:])}")
            try: wandb.log({
                'iter_loss': np.mean(log['loss'][-print_freq:]),
                'iter_base_loss': np.mean(log['base_loss'][-print_freq:])
            })
            except: pass
        
    return log


def iter_(model, data, optimizer):
    if optimizer is not None:
        model.zero_grad()
        pred = model(data)
        loss, base_loss = loss_func(data)
        loss.backward()
        if not np.isfinite(loss.item()):
            logger.warning(f"Nonfinite loss {loss.item()}; skipping")  
        elif loss.item() > 10.0:
            logger.warning(f"Large loss {loss.item()}; skipping")
        else:
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            except:
                logger.warning("Nonfinite grad, skipping")
    else: 
        with torch.no_grad():
            pred = model(data)
            loss, base_loss = loss_func(data)
    return data, loss, base_loss
        

def get_optimizer(model):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0003)
    return optimizer