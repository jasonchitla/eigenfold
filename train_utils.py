import torch, wandb
import numpy as np
from tqdm import tqdm
from utils import get_logger
logger = get_logger(__name__)

def loss_func(data):
    loss = ((data.score - data.pred)**2 / data.score_norm[:,None]**2).mean()
    base_loss = (data.score**2 / data.score_norm[:,None]**2).mean()    
    return loss, base_loss

def epoch(model, loader, optimizer=None, scheduler=None, scaler=None, device='cpu', print_freq=1000):
    if optimizer is not None: model.train()
    else: model.eval()
    
    log = {'rmsd': [], 'step': [], 'loss': [], 'base_loss': []}
    for i, data in tqdm(enumerate(loader), total=len(loader)):
        data = data.to(device)
        try:
            if data.skip:
                logger.warning(f"Skipping batch")
                continue
            data, loss, base_loss = iter_(model, data, optimizer, scaler)
            
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
            try: 
                if optimizer is not None:
                    wandb.log({
                        'iter_loss': np.mean(log['loss'][-print_freq:]),
                        'iter_base_loss': np.mean(log['base_loss'][-print_freq:])
                    })
                else:
                    wandb.log({
                        'val_iter_loss': np.mean(log['loss'][-print_freq:]),
                        'val_iter_base_loss': np.mean(log['base_loss'][-print_freq:])
                    })
            except: pass
    if optimizer is None and scheduler is not None: scheduler.step()
        
    return log


def iter_(model, data, optimizer, scaler):
    if optimizer is not None:
        model.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            pred = model(data)
            loss, base_loss = loss_func(data)
        scaler.scale(loss).backward()
        if not np.isfinite(loss.item()):
            logger.warning(f"Nonfinite loss {loss.item()}; skipping")  
        elif loss.item() > 10.0:
            logger.warning(f"Large loss {loss.item()}; skipping")
        else:
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
            except:
                logger.warning("Nonfinite grad, skipping")
    else: 
        with torch.no_grad():
            pred = model(data)
            loss, base_loss = loss_func(data)
    return data, loss, base_loss
        

def get_optimizer(model, lr):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    return optimizer