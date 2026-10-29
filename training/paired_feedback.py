"""Paired fresh evaluation, batching two independent policies on Apple MPS."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .evaluate import Bridge
from .smolvla import Adapter,load_policy,seed_all


def evaluate(args):
    dataset=args.dataset.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    selection=json.loads(args.selection.read_text())
    if not selection['improved_on_calibration'] or selection['gain']<=0:raise ValueError('No improved calibrated controller selected')
    gain=selection['gain'];metadata=json.loads((args.feedback.parent/'metadata.json').read_text())
    if hashlib.sha256(args.feedback.read_bytes()).hexdigest()!=selection['feedback_sha256']:
        raise ValueError('Selected feedback weights changed')
    if hashlib.sha256((args.checkpoint/'model.safetensors').read_bytes()).hexdigest()!=metadata['base_weights_sha256']:
        raise ValueError('Feedback/base policy mismatch')
    lock=json.loads((args.benchmark/'lock.json').read_text())
    for name,digest in lock['files_sha256'].items():
        if hashlib.sha256((args.benchmark/name).read_bytes()).hexdigest()!=digest:raise ValueError('Benchmark changed: '+name)
    scenarios=json.loads((args.benchmark/'fresh-paired.json').read_text())['scenarios']
    if len(scenarios)!=14:raise ValueError('Expected seven complete pairs')
    plan=dict(base_checkpoint=str(args.checkpoint.resolve()),base_weights_sha256=metadata['base_weights_sha256'],
        feedback=str(args.feedback.resolve()),feedback_sha256=hashlib.sha256(args.feedback.read_bytes()).hexdigest(),
        feedback_gain=gain,feedback_mode=selection.get('mode','joint'),feedback_lift_gate_m=selection.get('lift_gate_m',.025),
        selection=str(args.selection.resolve()),batch_size=2,proposal_noise='zero',
        replan_every=5,maximum_steps=400,physics_step_seconds=.001,
        benchmark_lock_sha256=hashlib.sha256((args.benchmark/'lock.json').read_bytes()).hexdigest(),
        conditions=7,teacher_enabled=False,thresholds_unchanged=True)
    (output/'plan.json').write_text(json.dumps(plan,indent=2))
    seed_all(42);policy=load_policy(str(args.checkpoint.resolve()),'mps',256).eval()
    adapter=Adapter(policy,json.loads((args.checkpoint/'statistics.json').read_text()),'mps');reports=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for start in range(0,14,2):
            pair=scenarios[start:start+2]
            for key in ('scene_xml','initial_state_npz','schema','goal_position_m'):
                if pair[0][key]!=pair[1][key]:raise ValueError('Unpaired reset or goal')
            bridges=[];tick=time.monotonic();policy.reset()
            try:
                for j,s in enumerate(pair):
                    bridges.append(Bridge(dataset,s,output/f'scenario_{start+j:02d}',400,10,
                        args.feedback if s['feedback_enabled'] else None,gain,
                        selection.get('lift_gate_m',.025),selection.get('mode','joint')))
                for step in range(0,400,5):
                    batch=adapter.batch([b.observation() for b in bridges],training=False)
                    noise=torch.zeros((2,policy.config.chunk_size,policy.config.max_action_dim),device='mps')
                    with torch.inference_mode():chunks=adapter.decode(policy.predict_action_chunk(batch,noise=noise))[:,:5].float().cpu().numpy()
                    if step==0 and not np.allclose(chunks[0],chunks[1],atol=1e-5,rtol=0):
                        raise ValueError('Paired reset produces different initial proposals')
                    futures=[pool.submit(b.step,u) for b,u in zip(bridges,chunks)]
                    for f in futures:f.result()
                    if step%100==0:print(f'Fresh pair {start//2+1}/7: {step+5}/400 control steps',flush=True)
                futures=[pool.submit(b.finish) for b in bridges]
                results=[f.result() for f in futures]
            finally:
                for b in bridges:b.close()
            for j,(r,s) in enumerate(zip(results,pair)):
                r.update(scenario=s,evaluation_suite=s['evaluation_suite'],checkpoint=str(args.checkpoint.resolve()),
                    seed=42,proposal_noise='zero',replanning_every_steps=5,inference_batch_size=2,
                    elapsed_pair_seconds=time.monotonic()-tick)
                if not r['arm_feedback']['finger_targets_unchanged']:raise ValueError('Finger commands changed')
                (output/f'scenario_{start+j:02d}'/'report.json').write_text(json.dumps(r,indent=2));reports.append(r)
            groups={}
            for suite in ('fresh_baseline','fresh_feedback'):
                rows=[r for r in reports if r['evaluation_suite']==suite]
                groups[suite]=dict(scenarios=len(rows),successes=sum(r['task_success'] for r in rows),
                    mean_planar_error_m=float(np.mean([r['final_position_error_m'] for r in rows])),
                    check_pass_counts={k:sum(r['checks'][k] for r in rows) for k in rows[0]['checks']})
            summary=dict(scenarios_completed=len(reports),scenarios_expected=14,paired_conditions=len(reports)//2,
                per_suite=groups,plan=plan,reports=reports,
                scope='Paired fresh seeds. Same case reset, appearance and goal for baseline/feedback. One human source recording; no teacher filtering.')
            (output/'summary.json').write_text(json.dumps(summary,indent=2))
            print(json.dumps(groups),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset',type=Path,default=Path('runs/book5-g1-dataset-01'))
    p.add_argument('--checkpoint',type=Path,default=Path('runs/book5-smolvla-02/best'))
    p.add_argument('--benchmark',type=Path,default=Path('runs/book5-feedback-benchmark-01'))
    p.add_argument('--feedback',type=Path,default=Path('runs/book5-arm-feedback-01/model/feedback.npz'))
    p.add_argument('--selection',type=Path,default=Path('runs/book5-arm-feedback-01/selected-controller.json'))
    p.add_argument('--output',type=Path,required=True);evaluate(p.parse_args())
