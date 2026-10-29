"""Lock calibration and fresh paired-test conditions before controller selection."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from humanoid.book_variations import randomize_xml


def build(args):
    root=args.dataset.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    original=json.loads((root/'reproduction-test.json').read_text())
    index=json.loads((root/'index.json').read_text())['episodes']
    entry=next(e for e in index if e['split']=='train' and e['physical_group']==0 and e['domain_seed']%100==0)
    p=root/entry['episode'];report=json.loads((p/'report.json').read_text())
    original_scenario=dict(episode='original-reference',scene_xml=str(root/original['scene_xml']),
        initial_state_npz=str(root/original['initial_state_npz']),schema=str(root/original['schema']),goal_position_m=original['goal_position_m'])
    train_scenario=dict(episode=entry['episode'],scene_xml=str(p/'scene.xml'),initial_state_npz=str(p/'initial_state.npz'),
        schema=str(p/'dataset.json'),goal_position_m=report['goal_position_m'])
    calibration=[]
    for gain in (0.,.25,.5,1.):
        for scenario in (original_scenario,train_scenario):
            calibration.append(dict(scenario,evaluation_suite=f'calibration_gain_{gain}',feedback_gain=gain,
                feedback_mode='joint',feedback_lift_gate_m=.025))
    (output/'calibration.json').write_text(json.dumps(dict(scenarios=calibration,
        selection='Training scenes only; prefer unchanged full task checks, then smaller placement error.'),indent=2))
    translation=[dict(s,evaluation_suite=f'calibration_translation_{gain}',feedback_gain=gain,
        feedback_mode='translation',feedback_lift_gate_m=.05)
        for gain in (.5,1.,1.5) for s in (original_scenario,train_scenario)]
    (output/'calibration-translation.json').write_text(json.dumps(dict(scenarios=translation,
        selection='Training-scene follow-up: delay correction until wrist rises 5 cm and preserve first-order orientation.'),indent=2))
    base=ET.parse(root/'original-reference/scene.xml')
    for a in base.findall('asset/*'):
        if a.get('file'):a.set('file',str((root/'original-reference'/a.get('file')).resolve()))
    base_xml=ET.tostring(base.getroot(),encoding='unicode');fresh=[]
    for i in range(7):
        seed=153100+i;rng=np.random.default_rng(seed);folder=output/f'fresh_{i:02d}';folder.mkdir()
        delta=np.r_[rng.uniform(-.004,.004,2),0.];mass=float(rng.uniform(.052,.058));friction=float(rng.uniform(.9,1.1))
        model=randomize_xml(base_xml,folder,seed,'test',mass,friction,delta)
        data=mujoco.MjData(model)
        mujoco.mj_setState(model,data,np.load(root/original['initial_state_npz'])['integration'],mujoco.mjtState.mjSTATE_INTEGRATION)
        q=model.jnt_qposadr[model.joint('case_free').id];data.qpos[q:q+3]+=delta
        mujoco.mj_forward(model,data);integration=np.zeros(mujoco.mj_stateSize(model,mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(model,data,integration,mujoco.mjtState.mjSTATE_INTEGRATION)
        np.savez_compressed(folder/'initial_state.npz',integration=integration)
        fresh.append(dict(episode=f'fresh_{i:02d}',scene_xml=str(folder/'scene.xml'),
            initial_state_npz=str(folder/'initial_state.npz'),schema=str(root/original['schema']),
            goal_position_m=(np.array(original['goal_position_m'])+delta).tolist(),seed=seed,
            physical_group=i,teacher_task_success=None))
    paired=[]
    for scenario in fresh:
        paired.append(dict(scenario,evaluation_suite='fresh_baseline',feedback_enabled=False))
        paired.append(dict(scenario,evaluation_suite='fresh_feedback',feedback_enabled=True))
    (output/'fresh-paired.json').write_text(json.dumps(dict(scenarios=paired,
        scope='New seeds, no teacher rejection sampling; same saved reset for each baseline/feedback pair. Robot initial pose fixed while case start shifts +/-4 mm.'),indent=2))
    hashes={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()}
    (output/'lock.json').write_text(json.dumps(dict(files_sha256=hashes,prepared_before_test=True,
        seed_range=[153100,153106],position_shift_m=.004,mass_range_kg=[.052,.058],
        friction_multiplier_range=[.9,1.1],teacher_filtering=False,
        protocol_change='Fresh cases keep robot reset fixed rather than retargeting its initial pose to each spawn. Compare paired results within this set, not directly with the older seven tests.'),indent=2))
    print('Locked 8 raw and 6 translation-filtered calibration trials; 7 fresh paired test conditions.')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--dataset',type=Path,default=Path('runs/book5-g1-dataset-01'))
    p.add_argument('--output',type=Path,required=True);build(p.parse_args())
