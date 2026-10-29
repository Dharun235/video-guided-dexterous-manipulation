"""Run a trained SmolVLA checkpoint closed loop through the isolated simulator."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

from .smolvla import Adapter,load_policy,seed_all

REPO=Path(__file__).resolve().parents[1]


class Bridge:
    def __init__(self,dataset,scenario,output,maximum_steps=400,video_stride=1,feedback=None,feedback_gain=1.,feedback_lift_gate=.025,feedback_mode='joint',include_tactile=False):
        self.output=output;self.include_tactile=include_tactile
        self.log=(output.parent/(output.name+'-simulator.log')).open('w')
        schema=dataset/scenario.get('schema',scenario['episode']+'/dataset.json')
        # Rejected preflights may not have image schemas; their robot mapping is
        # the same as the fixed original reference, not inferred from object state.
        if not schema.exists():schema=dataset/'original-reference/dataset.json'
        self.schema=json.loads(schema.read_text())
        cmd=[str(REPO/'.venv-g1/bin/python'),'-u','-m','humanoid.policy_bridge',
            '--scene',str(dataset/scenario['scene_xml']),'--initial',str(dataset/scenario['initial_state_npz']),
            '--schema',str(schema),'--goal',json.dumps(scenario['goal_position_m']),
            '--output',str(output),'--maximum-steps',str(maximum_steps),'--video-stride',str(video_stride)]
        if feedback:cmd+=['--feedback',str(feedback.resolve()),'--feedback-gain',str(feedback_gain),
            '--feedback-lift-gate',str(feedback_lift_gate),'--feedback-mode',feedback_mode]
        if include_tactile:cmd+=['--tactile-inputs']
        self.process=subprocess.Popen(cmd,cwd=REPO,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
        self.response=self.read()

    def read(self):
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('Simulator stopped; inspect '+str(self.log.name))
        return json.loads(line)

    def observation(self):
        with np.load(self.response['observation_npz']) as x:
            sample={'observation.state':x['state'].astype(np.float32),'task':self.schema['task']}
            if self.include_tactile:sample['observation.state']=np.r_[sample['observation.state'],x['tactile'].astype(np.float32)]
            for view in ('front','inspection'):
                sample['observation.images.'+view]=np.transpose(x[view],(2,0,1)).astype(np.float32)/255.
        return sample

    def step(self,actions):
        self.process.stdin.write(json.dumps(dict(command='step',actions=actions.tolist()))+'\n');self.process.stdin.flush()
        self.response=self.read()

    def finish(self):
        self.process.stdin.write(json.dumps(dict(command='finish'))+'\n');self.process.stdin.flush()
        self.response=self.read();self.process.wait(timeout=60);self.log.close()
        if self.process.returncode:raise RuntimeError('Simulator failed')
        return json.loads((self.output/'report.json').read_text())

    def close(self):
        if self.process.poll() is None:self.process.terminate();self.process.wait(timeout=10)
        if self.process.stdin:self.process.stdin.close()
        if self.process.stdout:self.process.stdout.close()
        self.log.close()


def compare_video(dataset,episode,output):
    spec=json.loads((dataset/'reproduction-test.json').read_text())
    source=cv2.VideoCapture(str(dataset/spec['source_video']));rollout=cv2.VideoCapture(str(episode/'execution.mp4'))
    fps=source.get(cv2.CAP_PROP_FPS);start=spec['source_task_interval_seconds'][0]
    end=spec['source_task_interval_seconds'][1]
    raw=output/'original-comparison-raw.mp4'
    rollout_fps=rollout.get(cv2.CAP_PROP_FPS)
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),rollout_fps,(1024,256))
    frame=0
    while True:
        ok,robot=rollout.read()
        if not ok:break
        source.set(cv2.CAP_PROP_POS_FRAMES,round(min(end,start+frame/rollout_fps)*fps));ok,phone=source.read()
        if not ok:raise RuntimeError('Cannot read source comparison frame')
        tile=np.zeros((256,256,3),np.uint8);phone=cv2.resize(phone,(256,143));tile[56:199]=phone
        cv2.putText(tile,'Source video (end held)',(5,20),0,.40,(255,255,255),1)
        writer.write(np.hstack([tile,robot]));frame+=1
    writer.release();source.release();rollout.release()
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(raw),'-c:v','libx264','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(output/'original-comparison.mp4')],check=True)


def evaluate(args):
    dataset=args.dataset.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    if args.feedback:
        metadata=json.loads((args.feedback.parent/'metadata.json').read_text())
        digest=hashlib.sha256((args.checkpoint/'model.safetensors').read_bytes()).hexdigest()
        if digest!=metadata['base_weights_sha256']:raise ValueError('Feedback was trained for a different base policy')
        if args.noise!='zero' or args.replan_every!=metadata['replan_every'] or args.image_size!=256:
            raise ValueError('Feedback requires its trained zero-latent, five-step, 256-pixel proposal contract')
    policy=load_policy(str(args.checkpoint.resolve()),args.device,args.image_size).eval()
    if args.replan_every>policy.config.chunk_size:raise ValueError('Replanning interval exceeds trained chunk size')
    stats=json.loads((args.checkpoint/'statistics.json').read_text());adapter=Adapter(policy,stats,args.device)
    if args.scenarios_file:
        scenarios=json.loads(args.scenarios_file.read_text())['scenarios']
        scenarios=[dict(s,evaluation_suite=s.get('evaluation_suite','custom')) for s in scenarios]
    elif args.suite in ('original','all'):
        s=json.loads((dataset/'reproduction-test.json').read_text());s['episode']='original-reference';scenarios=[s]
        s['evaluation_suite']='original'
        if args.suite=='all':
            locked=json.loads((dataset/'evaluation-scenarios.json').read_text())['scenarios']
            for split in ('validation','test'):
                scenarios.extend([dict(s,evaluation_suite=split) for s in locked[split]])
    else:
        scenarios=json.loads((dataset/'evaluation-scenarios.json').read_text())['scenarios'][args.suite]
        scenarios=[dict(s,evaluation_suite=args.suite) for s in scenarios]
    if args.limit is not None:scenarios=scenarios[:args.limit]
    reports=[]
    for i,scenario in enumerate(scenarios):
        current_suite=scenario['evaluation_suite']
        seed_all(args.seed);policy.reset();started=time.monotonic();episode=output/f'scenario_{i:02d}'
        video_stride=args.video_stride if args.video_stride is not None else (1 if current_suite=='original' else 5)
        feedback=args.feedback if scenario.get('feedback_enabled',True) else None
        feedback_gain=scenario.get('feedback_gain',args.feedback_gain)
        lift_gate=scenario.get('feedback_lift_gate_m',args.feedback_lift_gate)
        feedback_mode=scenario.get('feedback_mode',args.feedback_mode)
        bridge=Bridge(dataset,scenario,episode,args.steps,video_stride,feedback,feedback_gain,lift_gate,feedback_mode,
            include_tactile=policy.config.input_features['observation.state'].shape[0]==29)
        try:
            elapsed=0;inferences=0
            while elapsed<args.steps:
                batch=adapter.batch([bridge.observation()],training=False)
                noise=None
                if args.noise=='zero':noise=torch.zeros((1,policy.config.chunk_size,policy.config.max_action_dim),device=args.device)
                with torch.inference_mode():chunk=adapter.decode(policy.predict_action_chunk(batch,noise=noise))[0]
                action=chunk[:min(args.replan_every,args.steps-elapsed)].float().cpu().numpy()
                bridge.step(action);elapsed+=len(action);inferences+=1
                if inferences==1 or inferences%20==0:print(f'{current_suite} scenario {i}: {elapsed}/{args.steps} simulation steps',flush=True)
            report=bridge.finish()
        finally:bridge.close()
        report.update(scenario=scenario,evaluation_suite=current_suite,checkpoint=str(args.checkpoint.resolve()),seed=args.seed,
            replanning_every_steps=args.replan_every,latent_noise=args.noise,inference_calls=inferences,elapsed_seconds=time.monotonic()-started)
        (episode/'report.json').write_text(json.dumps(report,indent=2));reports.append(report)
        if current_suite=='original':compare_video(dataset,episode,output)
        per_suite={}
        for split in dict.fromkeys(r['evaluation_suite'] for r in reports):
            rows=[r for r in reports if r['evaluation_suite']==split]
            if rows:per_suite[split]=dict(scenarios=len(rows),successes=sum(r['task_success'] for r in rows),
                mean_planar_error_m=float(np.mean([r['final_position_error_m'] for r in rows])))
        summary=dict(suite=args.suite,checkpoint=str(args.checkpoint.resolve()),seed=args.seed,
            scenarios_completed=len(reports),scenarios_expected=len(scenarios),
            task_successes=sum(r['task_success'] for r in reports),
            mean_planar_error_m=float(np.mean([r['final_position_error_m'] for r in reports])),
            teacher_enabled=False,replanning_every_steps=args.replan_every,latent_noise=args.noise,
            arm_feedback=str(args.feedback.resolve()) if args.feedback else None,feedback_gain=args.feedback_gain,
            scope='Actual learned-policy rollouts. Original suite is training-derived reproduction, not independent human-video generalization.',
            per_suite=per_suite,reports=reports)
        (output/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps({k:v for k,v in summary.items() if k!='reports'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset',type=Path,default=Path('runs/book5-g1-dataset-01'))
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--suite',choices=('original','validation','test','all'),default='original')
    p.add_argument('--device',default='mps');p.add_argument('--image-size',type=int,default=256)
    p.add_argument('--steps',type=int,default=400);p.add_argument('--replan-every',type=int,default=None)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--limit',type=int)
    p.add_argument('--noise',choices=('random','zero'),default='random')
    p.add_argument('--video-stride',type=int)
    p.add_argument('--scenarios-file',type=Path)
    p.add_argument('--feedback',type=Path);p.add_argument('--feedback-gain',type=float,default=1.)
    p.add_argument('--feedback-lift-gate',type=float,default=.025)
    p.add_argument('--feedback-mode',choices=('joint','translation'),default='joint')
    a=p.parse_args()
    if a.replan_every is None:
        metadata=json.loads((a.checkpoint/'training.json').read_text()) if (a.checkpoint/'training.json').exists() else {}
        a.replan_every=1 if metadata.get('tactile_inputs',False) else 5
    if not 1<=a.replan_every<=50:raise SystemExit('Replanning interval must be 1-50')
    evaluate(a)
