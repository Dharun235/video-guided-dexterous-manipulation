"""Reproduce the published robot demonstration using saved motor commands."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def replay(args):
    example=ROOT/'examples/contact-demo';output=args.output.resolve()
    if output.exists():raise ValueError('Choose a new output folder: '+str(output))
    checks=json.loads((example/'checksums.json').read_text())['files_sha256']
    for name,expected in checks.items():
        path=example/name
        if not path.is_file():raise ValueError('Missing asset; run scripts.setup_g1 and scripts.setup_book_g1: '+name)
        with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=expected:raise ValueError('Example/robot asset differs from the pinned version: '+name)
    robot=json.loads((example/'robot.json').read_text());reference=np.load(example/'trajectory.npz')
    output.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u','-m','humanoid.policy_bridge','--scene',str(example/'scene.xml'),
        '--initial',str(example/'initial_state.npz'),'--schema',str(example/'robot.json'),
        '--goal',json.dumps(robot['goal_position_m']),'--output',str(output),
        '--maximum-steps',str(len(reference['action'])),'--controller-label','Recorded teacher replay']
    with (output.parent/(output.name+'-simulator.log')).open('w') as log:
        worker=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True)
        def read():
            line=worker.stdout.readline()
            if not line:raise RuntimeError('Simulator stopped; inspect '+str(log.name))
            return json.loads(line)
        def send(message):
            worker.stdin.write(json.dumps(message)+'\n');worker.stdin.flush();return read()
        force_errors=[]
        try:
            observation=read()
            with np.load(observation['observation_npz']) as sample:
                force_errors.append(float(np.max(np.abs(sample['tactile']-reference['observation_tactile'][0]))))
            for start in range(0,len(reference['action']),20):
                chunk=reference['action'][start:start+20];observation=send(dict(command='step',actions=chunk.tolist()))
                following=start+len(chunk)
                if following<len(reference['action']):
                    with np.load(observation['observation_npz']) as sample:
                        force_errors.append(float(np.max(np.abs(sample['tactile']-reference['observation_tactile'][following]))))
            send(dict(command='finish'));worker.wait(timeout=60)
            if worker.returncode:raise RuntimeError('Simulator failed; inspect '+str(log.name))
        finally:
            if worker.poll() is None:worker.terminate();worker.wait(timeout=10)
            worker.stdin.close();worker.stdout.close()
    with np.load(output/'episode.npz') as actual:
        errors={name:float(np.max(np.abs(actual[name]-reference[name]))) for name in
            ('action','observation_state','next_observation_state','object_qpos')}
    if max(errors.values())>1e-7 or max(force_errors)>1e-6:raise ValueError('Replay differs from the published reference')
    task=json.loads((output/'report.json').read_text())
    result=dict(replay_verified=True,task_success=task['task_success'],frames=len(reference['action']),
        maximum_errors=errors,maximum_sampled_force_replay_error_n=max(force_errors),
        video=str(output/'execution.mp4'),
        scope='Saved teacher motor targets replayed through physics; not policy training or automatic video reconstruction')
    (output/'replay-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,default=Path('runs/example-replay'))
    replay(parser.parse_args())
