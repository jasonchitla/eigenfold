import os, torch, wandb, time
import numpy as np
import pandas as pd
from score_model import ScoreModel
from data_utils import get_loader
from utils import get_logger, save_loss_plot
from train_utils import get_optimizer, epoch
from torch.optim.lr_scheduler import ReduceLROnPlateau
logger = get_logger(__name__)
    
def main(config=None):
    time_id = int(time.time()*1000)
    logger.info(f'Initializing run with ID {time_id}')
    with wandb.init(
        settings=wandb.Settings(start_method="fork"),
        project='harmonic-diffusion-antibodies',
        name=str(time_id),
        config=config
    ):
        # this config will be set by Sweep Controller
        config = wandb.config

        logger.info(f'Loading splits')
        splits = pd.read_csv('./limit256.csv')
        try: splits = splits.set_index('path')   
        except: splits = splits.set_index('name')   
        
        logger.info("Constructing model")
        # divide embed_dims by 2 to get position_embed_dims
        model = ScoreModel(embed_dims=config.embed_dims, num_conv_layers=config.num_conv_layers, position_embed_dims=config.embed_dims/2, tmin=0.001, tmax=1000000.0)
        
        total_params = sum([p.numel() for p in model.parameters()])
        logger.info(f"Model has {total_params} params")
        wandb.log({'total_params': total_params})

        model_dir = os.path.join('./workdir', str(time_id))
        if not os.path.exists(model_dir): os.mkdir(model_dir)
            
        device = torch.device('cuda')
        model = model.to(device)
        
        train_loader = get_loader(splits, inference_mode=False, mode='train', shuffle=True)
        val_loader = get_loader(splits, inference_mode=False, mode='val', shuffle=False)
        optimizer = get_optimizer(model, lr=config.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, patience=3)
        scaler = torch.cuda.amp.GradScaler()
        
        run_training(model, optimizer, scheduler, train_loader, val_loader, scaler, device, model_dir=model_dir)
        
def run_training(model, optimizer, scheduler, train_loader, val_loader, scaler, device, model_dir=None, 
                ep=1, best_val_loss = np.inf, best_epoch = 1):
    # 100 epochs
    while ep <= 100:
        logger.info(f"Starting training epoch {ep}")
        log = epoch(model, train_loader, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    device=device, print_freq=500)

        train_loss, train_base_loss = np.nanmean(log['loss']), np.nanmean(log['base_loss'])
        logger.info(f"Train epoch {ep}: len {len(log['loss'])} loss {train_loss}  base loss {train_base_loss}")
        
        logger.info(f"Starting validation epoch {ep}")
        log = epoch(model, val_loader, device=device, print_freq=500)
        
        val_loss, val_base_loss = np.nanmean(log['loss']), np.nanmean(log['base_loss'])
        logger.info(f"Val epoch {ep}: len {len(log['loss'])} loss {val_loss}  base loss {val_base_loss}")

        # save val loss plot
        png_path = os.path.join(model_dir, str(ep) + '.png')
        save_loss_plot(log, png_path)
        csv_path = os.path.join(model_dir, str(ep) + '.val.csv')
        pd.DataFrame(log).to_csv(csv_path)
        logger.info(f"Saved loss plot {png_path} and csv {csv_path}")

        # check if best epoch
        new_best = False
        if val_loss <= best_val_loss:
            best_val_loss = val_loss; best_epoch = ep
            logger.info(f"New best val epoch")
            new_best = True

        # save checkpoints
        state = {
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'epoch': ep,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
        }
        if new_best:
            path = os.path.join(model_dir, 'best_model.pt')
            logger.info(f"Saving best checkpoint {path}")
            torch.save(state, path)

        # save every 5 epochs
        # if ep % 5 == 0:
        #     path = os.path.join(model_dir, f'epoch_{ep}.pt')
        #     logger.info(f"Saving epoch checkpoint {path}")
        #     torch.save(state, path)
        
        update = {
            'train_loss': train_loss,
            'train_base_loss': train_base_loss,
            'val_loss': val_loss,
            'val_base_loss': val_base_loss,
            'current_lr': scheduler.get_last_lr()[0],
            'epoch': ep
        }
        logger.info(str(update))
        wandb.log(update)
            
        ep += 1
        
    logger.info("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))
    
    
if __name__ == '__main__':
    sweep_config = {
        'method': 'random'
    }
    sweep_config['metric'] = {
        'name': 'val_loss',
        'goal': 'minimize'   
    } 
    sweep_config['parameters'] = {
        'learning_rate': {
            'values': [0.0001, 0.0002, 0.0003, 0.0004]
        },
        'num_conv_layers': {
            'values': [4, 5, 6]
        },
        'embed_dims': {
            'values': [16, 24, 32]
        },
    }
    sweep_id = wandb.sweep(sweep_config, project="harmonic-diffusion-antibodies")
    wandb.agent(sweep_id, main, count=5)