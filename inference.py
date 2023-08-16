import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True)
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--num_conv_layers', type=int, required=True)
parser.add_argument('--splits', type=str, required=True)

parser.add_argument('--inf_mols', type=int, default=1000)
parser.add_argument('--elbo', action='store_true', default=False)
parser.add_argument('--alpha', type=float, default=0)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--inf_step', type=float, default=0.5)
parser.add_argument('--elbo_step', type=float, default=0.2)
parser.add_argument('--inf_type', type=str,
                        choices=['entropy', 'rate'], default='rate')

parser.add_argument('--embeddings_dir', type=str, default=None)

args = parser.parse_args()
args.inference_mode = True

import os, torch, wandb, time
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from utils import get_logger
logger = get_logger(__name__)
from data_utils import get_loader
import pandas as pd
from score_model import ScoreModel
from inference_utils import inference_epoch
    
def main():
    time_id = int(time.time()*1000)
    logger.info(f'Starting inference with ID {time_id}')
    wandb.init(
        settings=wandb.Settings(start_method="fork"),
        project='harmonic-diffusion-antibodies-inference',
        name=str(time_id)
    )

    logger.info(f'Loading splits {args.splits}')
    try: splits = pd.read_csv(args.splits).set_index('path')   
    except: splits = pd.read_csv(args.splits).set_index('name')   
    
    logger.info("Constructing model")
    model = ScoreModel(embed_dims=32, num_conv_layers=args.num_conv_layers, position_embed_dims=16, tmin=0.001, tmax=1000000.0, dropout=0.0).to(device)
    ckpt = os.path.join(args.model_dir, args.ckpt)
    
    logger.info(f'Loading weights from {ckpt}')
    state_dict = torch.load(ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['model'], strict=True)
    ep = state_dict['epoch']
    
    val_loader = get_loader(splits, inference_mode=True, mode='val', shuffle=False)
    samples, log, log_for_best_samples = inference_epoch(args, model, val_loader.dataset, device=device, pdbs=True, elbo=args.elbo)
    
    means = {key: np.mean(log[key]) for key in log if key != 'path'}
    logger.info(f"Inference epoch {ep}: len {len(log['rmsd'])} MEANS {means}")
    best_means = {key: np.mean(log_for_best_samples[key]) for key in log_for_best_samples if key != 'path'}
    logger.info(f"Best samples stats: len {len(log_for_best_samples['rmsd'])} MEANS {best_means}")
    
    inf_name = f"{args.splits.split('/')[-1]}.ep{ep}.num{args.num_samples}.step{args.inf_step}.alpha{args.alpha}.beta{args.beta}"
    if args.inf_step != args.elbo_step: inf_name += f".elbo{args.elbo_step}"
    csv_path = os.path.join(args.model_dir, f'{inf_name}.csv')
    pd.DataFrame(log).set_index('path').to_csv(csv_path)
    pd.DataFrame(log_for_best_samples).set_index('path').to_csv('log_for_best_samples.csv')
    logger.info(f"Saved inf csv {csv_path}")
    
    if not os.path.exists(os.path.join(args.model_dir, inf_name)): os.mkdir(os.path.join(args.model_dir, inf_name))
    for samp in samples:
        samp.pdb.write(os.path.join(args.model_dir, inf_name, samp.path.split('/')[-1] + f".{samp.copy}.anim.pdb"), reverse=True)
        samp.pdb.clear().add(samp.Y).write(
            os.path.join(args.model_dir, inf_name, samp.path.split('/')[-1] + f".{samp.copy}.pdb"))
        
if __name__ == '__main__':
    main()