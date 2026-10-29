"""Check split isolation, episode eligibility, diversity and create a review sheet."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def inspect(folder):
    index=json.loads((folder/'index.json').read_text());entries=index['episodes']
    groups={};domains=set();images={};trajectories={};counts={};panels=[]
    for entry in entries:
        p=folder/entry['episode'];a=json.loads((p/'audit.json').read_text());d=json.loads((p/'domain.json').read_text())
        schema=json.loads((p/'dataset.json').read_text())
        schema['state_units']=['rad']*len(schema['state_joint_names'])
        schema['action_units']=['rad']*len(schema['action_actuator_names'])
        schema['observation_state']='Selected G1 right-arm and Inspire joint positions, radians; includes passive mimic joints'
        schema['action']='Absolute G1 arm and Inspire motor joint-angle targets, radians; finite effort limits'
        (p/'dataset.json').write_text(json.dumps(schema,indent=2))
        groups.setdefault(entry['physical_group'],set()).add(entry['split'])
        domains.add(d['seed']);images.setdefault(a['image_sha256'],set()).add(entry['split'])
        trajectories.setdefault(a['episode_sha256'],set()).add(entry['physical_group'])
        assert entry['verified']==a['passed'],f'Eligibility mismatch {p}'
        row=counts.setdefault(entry['split'],dict(attempted=0,verified=0,samples=0,verified_groups=set(),failed_reasons={}))
        row['attempted']+=1
        if a['passed']:
            row['verified']+=1;row['samples']+=a['frames'];row['verified_groups'].add(entry['physical_group'])
        else:
            r=json.loads((p/'report.json').read_text())
            for name,passed in r['checks'].items():
                if not passed:row['failed_reasons'][name]=row['failed_reasons'].get(name,0)+1
        if entry['episode'].endswith('episode_00'):
            frame=min(110,a['frames']-1)
            image=cv2.imread(str(p/'images/observation.images.front'/f'{frame:06d}.jpg'))
            cv2.rectangle(image,(0,0),(256,38),(20,20,20),-1)
            cv2.putText(image,f'{entry["split"]} group {entry["physical_group"]}',(5,15),0,.42,(255,255,255),1)
            cv2.putText(image,'verified' if a['passed'] else 'excluded: failed task checks',(5,32),0,.36,(50,230,50) if a['passed'] else (80,80,255),1)
            panels.append(image)
    for value in counts.values():value['verified_groups']=sorted(value['verified_groups'])
    checks=dict(group_split_isolation=all(len(s)==1 for s in groups.values()),
                unique_domain_seeds=len(domains)==len(entries),
                no_identical_image_sequences_across_splits=all(len(s)==1 for s in images.values()),
                no_identical_physics_trajectories_across_groups=all(len(s)==1 for s in trajectories.values()),
                expected_attempts_complete=len(entries)==index['groups']*index['variants_per_group'],
                verified_examples_in_every_split=all(counts.get(s,{}).get('verified',0)>0 for s in ('train','validation','test')))
    if index.get('build_protocol')=='preflight-filtered-v1':
        checks['all_recorded_episodes_verified']=all(e['verified'] for e in entries)
    while len(panels)%4:panels.append(np.zeros((256,256,3),np.uint8))
    if panels:cv2.imwrite(str(folder/'variation-review.jpg'),np.vstack([np.hstack(panels[i:i+4]) for i in range(0,len(panels),4)]))
    result=dict(checks=checks,passed=all(checks.values()),counts=counts,
                physical_groups=len(groups),recorded_episodes=len(entries),training_started=False,
                scope='Quality and split checks for synthetic teacher demonstrations. Does not measure learned-policy generalization; all source motion derives from one human video.')
    (folder/'collection-audit.json').write_text(json.dumps(result,indent=2))
    selected={s:[e['episode'] for e in entries if e['split']==s and e['verified']] for s in ('train','validation','test')}
    (folder/'verified-splits.json').write_text(json.dumps(selected,indent=2))
    scenarios={}
    for split in ('validation','test'):
        scenarios[split]=[]
        for e in entries:
            if e['split']!=split:continue
            p=folder/e['episode'];r=json.loads((p/'report.json').read_text())
            scenarios[split].append(dict(episode=e['episode'],physical_group=e['physical_group'],
                scene_xml=e['episode']+'/scene.xml',initial_state_npz=e['episode']+'/initial_state.npz',
                goal_position_m=r['goal_position_m'],teacher_task_success=r['task_success'],
                action_hz=20,physics_step_seconds=.001,maximum_rollout_seconds=20,
                policy_inputs=['RGB front','RGB inspection','selected robot joint positions','task instruction'],
                exclude_from_policy=['object ground truth','teacher actions','source timestamps','audit labels']))
    preflight=json.loads((folder/'preflight-index.json').read_text()) if (folder/'preflight-index.json').exists() else []
    selected_candidates={e.get('preflight') for e in entries}
    for candidate in preflight:
        split=candidate['split']
        if split not in scenarios or candidate['folder'] in selected_candidates:continue
        prefix=candidate['folder']
        scenarios[split].append(dict(episode=prefix,physical_group=candidate['physical_group'],
            scene_xml=prefix+'/scene.xml',initial_state_npz=prefix+'/initial_state.npz',
            goal_position_m=candidate['goal_position_m'],teacher_task_success=candidate['passed'],
            from_failed_preflight=True,action_hz=20,physics_step_seconds=.001,maximum_rollout_seconds=20,
            policy_inputs=['RGB front','RGB inspection','selected robot joint positions','task instruction'],
            exclude_from_policy=['object ground truth','teacher actions','source timestamps','audit labels']))
    (folder/'evaluation-scenarios.json').write_text(json.dumps(dict(
        selection='All locked held-out conditions, including teacher failures; do not cherry-pick successful reference episodes',
        scenarios=scenarios),indent=2))
    (folder/'episode-checksums.json').write_text(json.dumps({e['episode']:dict(
        arrays=hashlib.sha256((folder/e['episode']/'episode.npz').read_bytes()).hexdigest(),
        domain=hashlib.sha256((folder/e['episode']/'domain.json').read_bytes()).hexdigest()) for e in entries},indent=2))
    lines=['# Unitree G1 + five-finger Inspire demonstration dataset','',
           'Synthetic robot demonstrations derived from Book5. Training has not started.', '',
           '| Split | Recorded | Verified | Verified samples | Verified physical groups |',
           '|---|---:|---:|---:|---|']
    for split,row in counts.items():
        lines.append(f'| {split} | {row["attempted"]} | {row["verified"]} | {row["samples"]} | {row["verified_groups"]} |')
    lines += ['', 'Use verified-splits.json to select imitation-learning episodes. Failed and partial attempts remain available for diagnosis.',
              '', 'For learned-policy rollout evaluation, use ALL conditions in evaluation-scenarios.json, including conditions where the teacher failed. Verified validation/test demonstrations are for reference-action metrics, not for selecting easier rollout tests.',
              '', 'Each episode has two synchronized 256x256 RGB cameras, 19 measured simulation joint positions and 13 absolute motor targets at 20 Hz. Images/state precede the action; next state follows 50 x 1 ms physics steps.',
              '', 'Read frames.jsonl for rows, episode.npz for precise arrays, dataset.json for named features and units, domain.json for variation parameters, and audit.json/report.json/contacts.json for evidence.',
              '', 'All variants of one movement group stay in a single split. Colors, table/wall textures, illumination and cameras vary. Case start position varies within +/-4 mm, mass 52-58 g, friction 0.9-1.1, source playback rate 0.65-0.80. Parameters remain fixed within an episode.',
              '', 'The teacher uses a recorded grasp and destination with adapted clearance, closure and release/retreat. The G1 pelvis is fixed on a support. The case is free under physics; no attachment constraints or direct case-pose updates drive it.',
              '', 'Teacher quality checks do not measure learned-policy generalization. All source motion comes from one human recording; geometric diversity is narrow. Larger shifts, new recordings and real hardware remain untested.',
              '', 'This is a local JSONL/NPZ/JPEG collection, not yet native LeRobot parquet. Finalization packages the pinned MuJoCo model assets inside the collection. Robot license and conversion details are in ROBOT_UPSTREAM.json and the included license files.',
              '', 'See variation-review.jpg for one example from every movement group; red labels identify excluded failures.']
    if index.get('build_protocol')=='preflight-filtered-v1':
        lines=[line.replace('Case start position varies within +/-4 mm, mass 52-58 g, friction 0.9-1.1, source playback rate 0.65-0.80.',
            'Case start position varies within +/-4 mm, mass 52-58 g, friction multiplier 0.9-1.1 around each original material, source playback rate 0.65-0.80.') for line in lines]
        lines += ['',f'Preflight selection: {len(preflight)} physical candidates were checked before recording. Failed candidates are retained separately, and all failed held-out conditions remain in evaluation-scenarios.json.',
            '', 'Success demonstrations are rejection sampled. Their acceptance count is not an unbiased policy success rate. Nominal hand friction is 1.0; case/book/table friction is 0.7 before applying the multiplier.']
    (folder/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('folder',type=Path);a=p.parse_args();inspect(a.folder)
