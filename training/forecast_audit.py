"""Measure validation action prediction errors; not a substitute for rollouts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import RobotDataset
from .smolvla import Adapter,load_policy,seed_all


def audit(args):
    seed_all(42)
    policy=load_policy(str(args.checkpoint.resolve()),'mps',256).eval()
    stats=json.loads((args.checkpoint/'statistics.json').read_text());adapter=Adapter(policy,stats,'mps')
    dataset=RobotDataset(args.dataset,'validation',policy.config.chunk_size)
    first=[];near=[];all_errors=[];rows=[]
    for i in np.linspace(0,len(dataset)-1,args.samples,dtype=int):
        sample=dataset[int(i)];batch=adapter.batch([sample],training=False)
        noise=None if args.noise=='random' else torch.zeros((1,policy.config.chunk_size,policy.config.max_action_dim),device='mps')
        with torch.inference_mode():pred=adapter.decode(policy.predict_action_chunk(batch,noise=noise))[0].float().cpu().numpy()
        error=pred-sample['action'];valid=~sample['action_is_pad']
        first.append(error[0]);near.extend(error[:min(5,int(valid.sum()))]);all_errors.extend(error[valid])
        rows.append(dict(dataset_row=int(i),episode=dataset.rows[int(i)]['episode'],
            first_action_rmse_rad=float(np.sqrt(np.mean(error[0]**2)))))
    first=np.array(first);near=np.array(near);all_errors=np.array(all_errors)
    result=dict(split='validation',samples=args.samples,latent_noise=args.noise,
        first_action_rmse_rad=float(np.sqrt(np.mean(first**2))),
        first_five_actions_rmse_rad=float(np.sqrt(np.mean(near**2))),
        full_chunk_rmse_rad=float(np.sqrt(np.mean(all_errors**2))),
        first_action_joint_rmse_rad=dict(zip(stats['action']['names'],np.sqrt(np.mean(first**2,axis=0)).tolist())),
        rows=rows,scope='Offline error on teacher-state observations; actual physical policy rollouts are required to measure task success.')
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('rows','first_action_joint_rmse_rad')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=Path('runs/book5-g1-dataset-01'))
    p.add_argument('--samples',type=int,default=32);p.add_argument('--noise',choices=('random','zero'),default='zero')
    p.add_argument('--output',type=Path,required=True);audit(p.parse_args())
