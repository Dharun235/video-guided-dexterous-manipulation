"""Cache VLA proposals, then learn precise arm corrections from audited data."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .dataset import EpisodeDataset,RobotDataset
from .smolvla import Adapter,load_policy,seed_all


def cache(args):
    args.output.mkdir(parents=True,exist_ok=False);seed_all(42)
    policy=load_policy(str(args.checkpoint.resolve()),'mps',256).eval()
    stats=json.loads((args.checkpoint/'statistics.json').read_text());adapter=Adapter(policy,stats,'mps')
    output={};started=time.monotonic()
    for split in ('train','validation'):
        sources=[RobotDataset(args.dataset,split,policy.config.chunk_size)]
        if split=='train':sources.append(EpisodeDataset(args.nominal,policy.config.chunk_size))
        jobs=[]
        for source_id,dataset in enumerate(sources):
            start=0
            while start<len(dataset):
                episode=dataset.rows[start]['episode'];end=start+1
                while end<len(dataset) and dataset.rows[end]['episode']==episode:end+=1
                folder=dataset.root/episode;schema=json.loads((folder/'dataset.json').read_text())
                with np.load(folder/'episode.npz') as x:
                    velocities=x['observation_velocity'].copy();states=x['observation_state'].copy();actions=x['action'].copy()
                mapping=[schema['state_joint_names'].index(n) for n in schema['action_actuator_names']]
                for i in range(start,end,5):jobs.append((dataset,i,start,end,states,velocities[:,mapping],actions,source_id))
                start=end
        features=[];targets=[];nominal=[]
        for offset in range(0,len(jobs),args.batch_size):
            batch_jobs=jobs[offset:offset+args.batch_size]
            samples=[job[0][job[1]] for job in batch_jobs]
            batch=adapter.batch(samples,training=False)
            noise=torch.zeros((len(samples),policy.config.chunk_size,policy.config.max_action_dim),device='mps')
            with torch.inference_mode():pred=adapter.decode(policy.predict_action_chunk(batch,noise=noise)).float().cpu().numpy()
            for proposal,job in zip(pred,batch_jobs):
                _,i,start,end,states,velocities,actions,source_id=job
                for k in range(min(5,end-i)):
                    j=i-start+k
                    features.append(np.r_[states[j],velocities[j],proposal[k]])
                    targets.append(actions[j,:7]);nominal.append(source_id==1)
            if offset%100==0:print(f'{split}: {offset}/{len(jobs)} visual contexts',flush=True)
        output[split+'_features']=np.array(features,np.float32)
        output[split+'_targets']=np.array(targets,np.float32)
        output[split+'_nominal']=np.array(nominal,bool)
        print(f'{split}: cached {len(features)} feedback examples',flush=True)
    np.savez_compressed(args.output/'cache.npz',**output)
    metadata=dict(base_checkpoint=str(args.checkpoint.resolve()),
        base_weights_sha256=hashlib.sha256((args.checkpoint/'model.safetensors').read_bytes()).hexdigest(),
        dataset=str(args.dataset.resolve()),dataset_manifest_sha256=hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest(),
        nominal=str(args.nominal.resolve()),nominal_arrays_sha256=hashlib.sha256((args.nominal/'episode.npz').read_bytes()).hexdigest(),
        feature_contract='19 robot joint positions + 13 motor-joint velocities + 13 decoded VLA proposals',
        target='7 actual arm motor targets; fingers unchanged',source_recordings=1,
        validation_used_for_training=False,test_used=False,proposal_noise='zero',replan_every=5,
        elapsed_seconds=time.monotonic()-started)
    (args.output/'cache.json').write_text(json.dumps(metadata,indent=2))


class Residual(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.ModuleList([nn.Linear(45,128),nn.Linear(128,128),nn.Linear(128,7)])
        nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)
    def forward(self,x):
        for layer in self.layers:x=torch.tanh(layer(x))
        return x


def train(args):
    args.output.mkdir(parents=True,exist_ok=False);seed_all(42);torch.set_num_threads(4)
    x=np.load(args.cache/'cache.npz');mean=x['train_features'].mean(0);std=np.maximum(x['train_features'].std(0),1e-4)
    if not all(np.isfinite(x[k]).all() for k in x.files):raise ValueError('Nonfinite feedback cache')
    features={s:torch.from_numpy(np.clip((x[s+'_features']-mean)/std,-8,8)) for s in ('train','validation')}
    targets={s:torch.from_numpy(x[s+'_targets']) for s in features}
    proposals={s:torch.from_numpy(x[s+'_features'][:,-13:-6]) for s in features}
    jacobians={s:torch.from_numpy(x[s+'_jacobian']) for s in features}
    cap=torch.full((7,),.12);model=Residual();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5)
    nominal=np.flatnonzero(x['train_nominal']);regular=np.flatnonzero(~x['train_nominal']);rng=np.random.default_rng(42)
    best=float('inf');started=time.monotonic()
    def loss_for(split,indices):
        delta=cap*model(features[split][indices]);error=proposals[split][indices]+delta-targets[split][indices]
        cartesian=torch.einsum('bij,bj->bi',jacobians[split][indices],error)
        loss=(cartesian/.003).square().mean()+.05*(error/.03).square().mean()+.001*(delta/cap).square().mean()
        return loss,error,cartesian
    with (args.output/'metrics.jsonl').open('w') as log:
        for step in range(args.steps+1):
            if step:
                ids=[int(rng.choice(nominal if rng.random()<.25 else regular)) for _ in range(128)]
                loss,_,_=loss_for('train',ids)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite feedback loss')
                optimizer.zero_grad();loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
            if step%100==0 or step==args.steps:
                with torch.no_grad():vl,error,cartesian=loss_for('validation',slice(None))
                record=dict(step=step,validation_loss=float(vl),arm_rmse_rad=float(error.square().mean().sqrt()),
                    linearized_wrist_rmse_m=float(cartesian.square().mean().sqrt()),elapsed_seconds=time.monotonic()-started)
                log.write(json.dumps(record)+'\n');log.flush()
                if float(vl)<best:
                    best=float(vl);weights=dict(mean=mean,std=std,cap=cap.numpy())
                    for i,layer in enumerate(model.layers):weights[f'w{i}']=layer.weight.detach().numpy().copy();weights[f'b{i}']=layer.bias.detach().numpy().copy()
                    np.savez_compressed(args.output/'feedback.npz',**weights)
                    (args.output/'selection.json').write_text(json.dumps(record,indent=2))
                if step%500==0:print(json.dumps(record),flush=True)
    metadata=json.loads((args.cache/'cache.json').read_text());metadata.update(training_steps=args.steps,
        correction_cap_rad=.12,finger_commands_modified=False,gating='Runtime activation gate/mode/gain are selected separately on training-scene rollouts',
        actor_inputs=['robot joint positions','robot motor joint velocities','SmolVLA proposals','own wrist forward kinematics'],
        object_state_used_for_actor=False,source_timestamps_used=False,training_type='Supervised bounded arm residual; not RL',
        normalization_fit_split='train',nominal_sampling_probability=.25)
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print('Saved validation-selected arm feedback model',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('cache');c.add_argument('--checkpoint',type=Path,default=Path('runs/book5-smolvla-02/best'))
    c.add_argument('--dataset',type=Path,default=Path('runs/book5-g1-dataset-01'))
    c.add_argument('--nominal',type=Path,default=Path('runs/book5-g1-ego-09'));c.add_argument('--output',type=Path,required=True)
    c.add_argument('--batch-size',type=int,default=4)
    t=sub.add_parser('train');t.add_argument('--cache',type=Path,required=True);t.add_argument('--output',type=Path,required=True)
    t.add_argument('--steps',type=int,default=5000)
    a=p.parse_args();cache(a) if a.command=='cache' else train(a)
