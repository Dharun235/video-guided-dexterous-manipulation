"""Local SmolVLA fine-tuning against the audited RGB/state/action contract.

Uses the upstream policy directly with a small, explicit JSONL adapter. No Hub
uploads, test-set training, human timestamps or object ground truth are used.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from .dataset import EpisodeDataset,RobotDataset

IMAGE_KEYS=('observation.images.front','observation.images.inspection')


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.backends.mps.is_available():torch.mps.manual_seed(seed)


class Adapter:
    def __init__(self,policy,statistics,device):
        self.device=device
        self.tokenizer=policy.model.vlm_with_expert.processor.tokenizer
        self.max_length=policy.config.tokenizer_max_length
        self.stats=statistics
        self.means={k:torch.tensor(statistics[k]['mean'],dtype=torch.float32,device=device)
                    for k in ('observation.state','action')}
        self.stds={k:torch.tensor(statistics[k]['std'],dtype=torch.float32,device=device).clamp_min(1e-4)
                   for k in self.means}

    def batch(self,samples,training=True):
        result={}
        for key in ('observation.state',*IMAGE_KEYS):
            result[key]=torch.from_numpy(np.stack([s[key] for s in samples])).to(self.device)
        result['observation.state']=(result['observation.state']-self.means['observation.state'])/self.stds['observation.state']
        text=[s['task'].strip()+'\n' for s in samples]
        tokens=self.tokenizer(text,padding='max_length',max_length=self.max_length,truncation=True,return_tensors='pt')
        result['observation.language.tokens']=tokens['input_ids'].to(self.device)
        result['observation.language.attention_mask']=tokens['attention_mask'].to(self.device,dtype=torch.bool)
        if training:
            result['action']=torch.from_numpy(np.stack([s['action'] for s in samples])).to(self.device)
            result['action']=(result['action']-self.means['action'])/self.stds['action']
            result['action_is_pad']=torch.from_numpy(np.stack([s['action_is_pad'] for s in samples])).to(self.device)
        return result

    def decode(self,actions):
        return actions*self.stds['action']+self.means['action']


def load_policy(path,device='mps',image_size=256,chunk_size=None,state_dimension=None):
    config=SmolVLAConfig.from_pretrained(path)
    if (Path(path)/'backbone').is_dir():config.vlm_model_name=str((Path(path)/'backbone').resolve())
    config.device=device;config.push_to_hub=False;config.load_vlm_weights=False
    if state_dimension is None:
        state_dimension=29 if (Path(path)/'training.json').exists() and json.loads((Path(path)/'training.json').read_text()).get('tactile_inputs',False) else 19
    if state_dimension>config.max_state_dim:raise ValueError('Robot state exceeds pretrained state projection width')
    config.input_features={'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(state_dimension,)),
        **{k:PolicyFeature(type=FeatureType.VISUAL,shape=(3,256,256)) for k in IMAGE_KEYS}}
    config.output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(13,))}
    config.freeze_vision_encoder=True;config.train_expert_only=True;config.train_state_proj=True
    config.resize_imgs_with_padding=(image_size,image_size);config.n_action_steps=5
    if chunk_size is not None:config.chunk_size=chunk_size;config.n_action_steps=min(5,chunk_size)
    policy=SmolVLAPolicy.from_pretrained(path,config=config,strict=True)
    return policy


@torch.no_grad()
def validate(policy,adapter,dataset,indices):
    policy.eval();seed_all(991)
    losses=[]
    for index in indices:
        loss,_=policy(adapter.batch([dataset[int(index)]]));losses.append(float(loss))
    return float(np.mean(losses))


def save_checkpoint(policy,adapter,folder,metadata):
    folder.mkdir(parents=True,exist_ok=True)
    policy.save_pretrained(folder)
    (folder/'statistics.json').write_text(json.dumps(adapter.stats,indent=2))
    (folder/'training.json').write_text(json.dumps(metadata,indent=2))
    # Tokenizer and base VLM configuration are cached locally by Hugging Face.
    policy.model.vlm_with_expert.processor.save_pretrained(folder/'backbone')
    policy.model.vlm_with_expert.config.save_pretrained(folder/'backbone')


def train(args):
    root=args.dataset.resolve();output=args.output.resolve()
    prep=json.loads((root/'preparation.json').read_text())
    audit=json.loads((root/'collection-audit.json').read_text())
    if not prep['data_ready'] or not audit['passed']:raise ValueError('Dataset quality gate failed')
    output.mkdir(parents=True,exist_ok=False)
    tactile=getattr(args,'tactile',False)
    stats_path=root/('statistics-tactile.json' if tactile else 'statistics.json')
    stats=json.loads(stats_path.read_text())
    if stats['fit_split']!='train':raise ValueError('Normalization must use training rows only')
    seed_all(args.seed);torch.set_num_threads(4)
    if args.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('Apple GPU unavailable')
    if tactile and args.extra_episode:raise ValueError('Do not mix a nominal episode without tactile measurements')
    policy=load_policy(str(args.resume or args.base),args.device,args.image_size,args.chunk_size,29 if tactile else 19)
    adapter=Adapter(policy,stats,args.device)
    train_data=RobotDataset(root,'train',policy.config.chunk_size,include_tactile=tactile)
    val_data=RobotDataset(root,'validation',policy.config.chunk_size,include_tactile=tactile)
    extra=EpisodeDataset(args.extra_episode,policy.config.chunk_size) if args.extra_episode else None
    phase_groups={}
    if getattr(args,'phase_balanced',False):
        if extra:raise ValueError('Phase-balanced training requires labelled collection rows')
        for i,row in enumerate(train_data.rows):phase_groups.setdefault(row['supervision.phase'],[]).append(i)
    val_indices=np.linspace(0,len(val_data)-1,args.validation_samples,dtype=int)
    params=[p for p in policy.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=args.lr,betas=(.9,.95),weight_decay=1e-10)
    initial=policy.model.state_proj.weight.detach().cpu().clone()
    manifest_hash=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
    metadata=dict(dataset=str(root),dataset_manifest_sha256=manifest_hash,base_model=str(args.base),tactile_inputs=tactile,
        phase_balanced_sampling=bool(phase_groups),teacher_phase_is_policy_input=False,
        recommended_replanning_steps=1 if tactile else 5,
        seed=args.seed,device=args.device,image_size=args.image_size,batch_size=args.batch_size,
        requested_steps=args.steps,lr=args.lr,trainable_parameters=sum(p.numel() for p in params),
        chunk_size=policy.config.chunk_size,initial_checkpoint=str(args.resume.resolve()) if args.resume else None,
        optimizer_resumed=False,normalization_frozen_from_training_split=True,
        extra_training_episode=str(args.extra_episode.resolve()) if extra else None,
        extra_training_episode_sha256=hashlib.sha256((args.extra_episode/'episode.npz').read_bytes()).hexdigest() if extra else None,
        extra_episode_sampling_probability=args.extra_probability if extra else 0.,
        total_parameters=sum(p.numel() for p in policy.parameters()),
        versions={k:importlib.metadata.version(k) for k in ('torch','lerobot','transformers')},
        source_recordings=1,normalization_fit_split='train',test_used_for_training_or_selection=False,
        teacher_at_inference=False,training_complete=False)
    (output/'run.json').write_text(json.dumps(metadata,indent=2))
    shutil.copy2(stats_path,output/'statistics.json')
    if Path('training/base-model.json').exists():shutil.copy2('training/base-model.json',output/'base-model.json')
    started=time.monotonic();best=float('inf');rng=np.random.default_rng(args.seed)
    initial_val=validate(policy,adapter,val_data,val_indices)
    print(json.dumps(dict(step=0,validation_loss=initial_val,trainable=metadata['trainable_parameters'],total=metadata['total_parameters'])),flush=True)
    with (output/'metrics.jsonl').open('a') as log:
        log.write(json.dumps(dict(step=0,validation_loss=initial_val))+'\n');log.flush()
        for step in range(1,args.steps+1):
            tick=time.monotonic();policy.train()
            factor=min(1.,step/max(1,args.warmup_steps))*(.025+.975*.5*(1+math.cos(math.pi*step/args.steps)))
            for group in optimizer.param_groups:group['lr']=args.lr*factor
            samples=[]
            for _ in range(args.batch_size):
                if phase_groups:
                    phase_name=str(rng.choice(sorted(phase_groups)));samples.append(train_data[int(rng.choice(phase_groups[phase_name]))])
                else:
                    source=extra if extra is not None and rng.random()<args.extra_probability else train_data
                    samples.append(source[int(rng.integers(len(source)))])
            loss,_=policy(adapter.batch(samples))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(params,10.)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradients')
            optimizer.step()
            if args.device=='mps':torch.mps.synchronize()
            record=dict(step=step,train_loss=float(loss.detach()),gradient_norm=float(norm.detach()),
                lr=optimizer.param_groups[0]['lr'],step_seconds=time.monotonic()-tick,
                elapsed_seconds=time.monotonic()-started)
            if step%args.eval_every==0 or step==args.steps:
                record['validation_loss']=validate(policy,adapter,val_data,val_indices)
                if record['validation_loss']<best:
                    best=record['validation_loss']
                    save_checkpoint(policy,adapter,output/'best',dict(metadata,steps_completed=step,validation_loss=best))
            log.write(json.dumps(record)+'\n');log.flush()
            if step<=5 or step%10==0 or 'validation_loss' in record:print(json.dumps(record),flush=True)
    delta=float(torch.linalg.vector_norm(policy.model.state_proj.weight.detach().cpu()-initial))
    metadata.update(training_complete=True,steps_completed=args.steps,best_validation_loss=best,
        initial_validation_loss=initial_val,state_projection_parameter_change_l2=delta,
        elapsed_seconds=time.monotonic()-started,learned_policy_task_success_measured=False)
    save_checkpoint(policy,adapter,output/'last',metadata)
    best_metadata=json.loads((output/'best/training.json').read_text())
    best_metadata['training_complete']=True
    (output/'best/training.json').write_text(json.dumps(best_metadata,indent=2))
    torch.save(optimizer.state_dict(),output/'optimizer.pt')
    (output/'run.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset',type=Path,default=Path('runs/book5-g1-dataset-01'))
    p.add_argument('--tactile',action='store_true',help='Append ten external fingertip force features to nineteen joint positions')
    p.add_argument('--phase-balanced',action='store_true',help='Sample teacher phases uniformly using training labels only; phase is not a model input')
    p.add_argument('--base',type=Path,default=Path('.cache/smolvla-base'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--init-from','--resume',dest='resume',type=Path,help='Warm-start weights; optimizer restarts')
    p.add_argument('--chunk-size',type=int,default=50)
    p.add_argument('--extra-episode',type=Path);p.add_argument('--extra-probability',type=float,default=.5)
    p.add_argument('--device',default='mps');p.add_argument('--steps',type=int,default=300)
    p.add_argument('--batch-size',type=int,default=2);p.add_argument('--image-size',type=int,default=256)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--warmup-steps',type=int,default=20);p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--validation-samples',type=int,default=8)
    a=p.parse_args()
    if a.steps<1 or a.batch_size<1 or a.chunk_size<5 or not 0<=a.extra_probability<=1:
        raise SystemExit('Invalid training size or sampling probability')
    train(a)
