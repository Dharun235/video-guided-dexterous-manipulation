"""Regression check: the policy bridge must reproduce known actions exactly.

This checks the simulator interface, not a learned policy.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def check(args):
    spec=json.loads((args.dataset/'reproduction-test.json').read_text())
    reference=np.load(args.reference/'episode.npz')
    cmd=[sys.executable,'-u','-m','humanoid.policy_bridge',
        '--scene',str(args.dataset/spec['scene_xml']),'--initial',str(args.dataset/spec['initial_state_npz']),
        '--schema',str(args.dataset/spec['schema']),'--goal',json.dumps(spec['goal_position_m']),
        '--output',str(args.output),'--controller-label','Bridge replay check']
    if args.feedback:cmd+=['--feedback',str(args.feedback),'--feedback-gain','0']
    process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    try:
        initial=json.loads(process.stdout.readline())
        with np.load(initial['observation_npz']) as x:
            assert np.allclose(x['state'],reference['observation_state'][0],atol=1e-10)
        for start in range(0,len(reference['action']),20):
            chunk=reference['action'][start:start+20]
            process.stdin.write(json.dumps(dict(command='step',actions=chunk.tolist()))+'\n');process.stdin.flush()
            result=json.loads(process.stdout.readline())
            with np.load(result['observation_npz']) as x:
                assert np.allclose(x['state'],reference['next_observation_state'][start+len(chunk)-1],atol=1e-7)
        process.stdin.write('{"command":"finish"}\n');process.stdin.flush()
        result=json.loads(process.stdout.readline());process.wait(timeout=60)
        replay=np.load(args.output/'episode.npz')
        errors={name:float(np.max(np.abs(replay[name]-reference[name])))
                for name in ('observation_state','action','object_qpos','next_observation_state')}
        assert max(errors.values())<1e-7 and process.returncode==0
        report=dict(passed=True,maximum_errors=errors,frames=len(reference['action']),
            scope='Simulator interface regression using recorded teacher actions; not learned-policy success')
        (args.output/'bridge-check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    finally:
        if process.poll() is None:process.terminate();process.wait(timeout=10)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset',type=Path,default=Path('runs/book5-g1-dataset-01'))
    p.add_argument('--reference',type=Path,default=Path('runs/book5-g1-ego-09'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--feedback',type=Path)
    check(p.parse_args())
