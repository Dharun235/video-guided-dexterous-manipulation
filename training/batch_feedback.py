"""Evaluate the frozen correction on every original locked held-out condition."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate import Bridge
from .smolvla import Adapter,load_policy,seed_all


def evaluate(args):
    args.output.mkdir(parents=True,exist_ok=False)
    selection=json.loads(args.selection.read_text());metadata=json.loads((args.feedback.parent/'metadata.json').read_text())
    if hashlib.sha256(args.feedback.read_bytes()).hexdigest()!=selection['feedback_sha256']:raise ValueError('Feedback changed')
    if hashlib.sha256((args.checkpoint/'model.safetensors').read_bytes()).hexdigest()!=metadata['base_weights_sha256']:raise ValueError('Base changed')
    locked=json.loads((args.dataset/'evaluation-scenarios.json').read_text())['scenarios']
    scenarios=[dict(s,evaluation_suite=split) for split in ('validation','test') for s in locked[split]]
    if len(scenarios)!=14:raise ValueError('Missing locked conditions')
    plan=dict(checkpoint=str(args.checkpoint.resolve()),selection=selection,scenarios=scenarios,
        inference_batch_size=args.batch_size,proposal_noise='zero',replan_every=5,maximum_steps=400,
        feedback_sha256=selection['feedback_sha256'],base_weights_sha256=metadata['base_weights_sha256'],
        teacher_enabled=False,thresholds_unchanged=True)
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2))
    seed_all(42);policy=load_policy(str(args.checkpoint.resolve()),'mps',256).eval()
    adapter=Adapter(policy,json.loads((args.checkpoint/'statistics.json').read_text()),'mps');reports=[]
    with ThreadPoolExecutor(max_workers=args.batch_size) as pool:
        for start in range(0,len(scenarios),args.batch_size):
            current=scenarios[start:start+args.batch_size];bridges=[];policy.reset()
            try:
                for j,s in enumerate(current):
                    bridges.append(Bridge(args.dataset.resolve(),s,args.output/f'scenario_{start+j:02d}',400,10,
                        args.feedback,selection['gain'],selection['lift_gate_m'],selection['mode']))
                for step in range(0,400,5):
                    batch=adapter.batch([b.observation() for b in bridges],training=False)
                    noise=torch.zeros((len(bridges),policy.config.chunk_size,policy.config.max_action_dim),device='mps')
                    with torch.inference_mode():chunks=adapter.decode(policy.predict_action_chunk(batch,noise=noise))[:,:5].float().cpu().numpy()
                    futures=[pool.submit(b.step,u) for b,u in zip(bridges,chunks)]
                    for f in futures:f.result()
                    if step%100==0:print(f'Locked batch {start//args.batch_size+1}: {step+5}/400 steps',flush=True)
                futures=[pool.submit(b.finish) for b in bridges];results=[f.result() for f in futures]
            finally:
                for b in bridges:b.close()
            for j,(r,s) in enumerate(zip(results,current)):
                if not r['arm_feedback']['finger_targets_unchanged']:raise ValueError('Finger targets changed')
                r.update(scenario=s,evaluation_suite=s['evaluation_suite'],inference_batch_size=len(current))
                (args.output/f'scenario_{start+j:02d}'/'report.json').write_text(json.dumps(r,indent=2));reports.append(r)
            groups={}
            for split in ('validation','test'):
                rows=[r for r in reports if r['evaluation_suite']==split]
                if rows:groups[split]=dict(scenarios=len(rows),successes=sum(r['task_success'] for r in rows),
                    mean_planar_error_m=float(np.mean([r['final_position_error_m'] for r in rows])),
                    check_pass_counts={k:sum(r['checks'][k] for r in rows) for k in rows[0]['checks']})
            (args.output/'summary.json').write_text(json.dumps(dict(scenarios_completed=len(reports),
                scenarios_expected=14,plan=plan,reports=reports,per_suite=groups),indent=2))
            print(json.dumps(groups),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset',type=Path,default=Path('runs/book5-g1-dataset-01'))
    p.add_argument('--checkpoint',type=Path,default=Path('runs/book5-smolvla-02/best'))
    p.add_argument('--feedback',type=Path,default=Path('runs/book5-arm-feedback-01/model/feedback.npz'))
    p.add_argument('--selection',type=Path,default=Path('runs/book5-arm-feedback-01/selected-controller.json'))
    p.add_argument('--batch-size',type=int,default=2);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not 1<=a.batch_size<=4:raise SystemExit('Batch size must be 1-4')
    evaluate(a)
