"""Seal tactile data, fit train-only statistics, and verify both input contracts."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .book_dataset_finalize import finalize,save
from .tactile import FEATURE_NAMES
from training.dataset import RobotDataset


def phase(t):
    for end,name in ((4.5,'approach'),(5.1,'close'),(6.8,'lift'),(8.7,'carry'),(9.9,'lower'),(10.9,'release')):
        if t<end:return name
    return 'retreat'


def seal(root):
    index=json.loads((root/'index.json').read_text())
    if index['collection_status']!='complete':raise ValueError('Collection has not completed')
    # Reuse the strict image/physics/split/portable-asset preparation.
    finalize(root)
    index=json.loads((root/'index.json').read_text());train_states=[];train_tactile=[];phase_counts={};corrections=[];peaks=[];initial_heights=[];opposed_holds=[];placement_errors=[];quality_rows=[]
    for entry in index['episodes']:
        folder=root/entry['episode'];x=np.load(folder/'episode.npz')
        if x['observation_tactile'].shape!=(entry['frames'],10):raise ValueError('Tactile dimension mismatch')
        if 'action_reference' not in x:raise ValueError('Missing reference/executed action distinction')
        audit=json.loads((folder/'audit.json').read_text())
        if not audit['checks'].get('tactile_matches_physics_replay'):raise ValueError('Tactile replay not verified')
        if not json.loads((folder/'report.json').read_text())['active_tactile_feedback']['enabled']:raise ValueError('Feedback is disabled')
        rows=[];stable_count=0;grasp_labels=[];raised_labels=[]
        contacts=json.loads((folder/'contacts.json').read_text())
        report=json.loads((folder/'report.json').read_text());goal=np.array(report['goal_position_m'])
        placement_error=float(np.linalg.norm(np.array(report['final_case_position_m'])[:2]-goal[:2]))
        if entry['domain_seed']%100==0:placement_errors.append(placement_error)
        initial_heights.append(float(x['object_qpos'][0,2]-goal[2]))
        for i,line in enumerate((folder/'frames.jsonl').read_text().splitlines()):
            row=json.loads(line);row['supervision.phase']=phase(float(x['source_timestamp'][i]))
            fingers={c['body'][2:].rsplit('_',1)[0] for c in contacts[i]['contacts']
                if c['body'].startswith('R_') and c['body'].endswith(('distal','intermediate')) and c['normal_force_n']>.1}
            contact_grasp='thumb' in fingers and len(fingers)>=2
            stable_count=stable_count+1 if contact_grasp else 0
            grasp_labels.append(stable_count>=3)
            raised_labels.append(contact_grasp and x['object_qpos'][i,2]>x['object_qpos'][0,2]+.02)
            row['supervision.grasp_stable']=bool(grasp_labels[-1]);row['supervision.raised_grasp']=bool(raised_labels[-1])
            row['action.reference']=x['action_reference'][i].tolist();rows.append(row)
            counter=phase_counts.setdefault(entry['split'],collections.Counter());counter[row['supervision.phase']]+=1
        (folder/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        np.savez_compressed(folder/'auxiliary-labels.npz',grasp_stable=grasp_labels,raised_grasp=raised_labels,
            placed_on_table=np.abs(x['object_qpos'][:,2]-goal[2])<.005,
            within_goal_3cm=np.linalg.norm(x['object_qpos'][:,:2]-goal[:2],axis=1)<.03)
        longest=current=0
        for held in raised_labels:
            current=current+1 if held else 0;longest=max(longest,current)
        opposed_holds.append(longest/20)
        if longest/20<.5:raise ValueError('Missing raised thumb/opposing-finger grasp: '+entry['episode'])
        save(folder/'tactile-quality.json',dict(passed=True,raised_thumb_opposition_hold_s=longest/20,
            maximum_sampled_fingertip_normal_force_n=float(x['observation_tactile'][:,::2].max()),
            maximum_feedback_target_change_rad=float(np.max(np.abs(x['action']-x['action_reference']))),
            note='Additional tactile data checks; not learned-policy success or a hardware safety rating'))
        quality_rows.append(dict(episode=entry['episode'],split=entry['split'],physical_group=entry['physical_group'],
            policy_reset_prefix_steps=report['policy_derived_reset_prefix_steps'],frames=entry['frames'],
            placement_error_cm=100*placement_error,raised_thumb_opposition_hold_s=longest/20,
            peak_sampled_fingertip_normal_force_n=float(x['observation_tactile'][:,::2].max()),
            maximum_replay_tactile_error_n=audit['maximum_replay_tactile_error_n']))
        peaks.append(float(x['observation_tactile'][:,::2].max()))
        corrections.append(float(np.max(np.abs(x['action']-x['action_reference']))))
        if entry['split']=='train':
            train_states.append(np.c_[x['observation_state'],x['observation_tactile']]);train_tactile.append(x['observation_tactile'])
    # Rewrite rows with the explicit force and phase labels after portable paths.
    for split in ('train','validation','test'):
        with (root/'data'/f'{split}.jsonl').open('w') as stream:
            for ei,entry in enumerate(index['episodes']):
                if entry['split']!=split:continue
                for line in (root/entry['episode']/'frames.jsonl').read_text().splitlines():
                    row=json.loads(line);row.update(episode=entry['episode'],episode_index=ei)
                    for key in ('observation.images.front','observation.images.inspection'):
                        row[key]=str(Path(entry['episode'])/row[key])
                    stream.write(json.dumps(row)+'\n')
    schema=json.loads((root/index['episodes'][0]['episode']/'dataset.json').read_text())
    stats=json.loads((root/'statistics.json').read_text());tactile_stats=json.loads(json.dumps(stats))
    a=np.vstack(train_states);names=schema['state_joint_names']+FEATURE_NAMES
    tactile_stats['observation.state']=dict(names=names,mean=a.mean(0).tolist(),std=a.std(0).tolist(),min=a.min(0).tolist(),max=a.max(0).tolist())
    tactile_stats['input_contract']='19 joint positions in radians + 10 external fingertip normal/shear forces in newtons'
    save(root/'statistics-tactile.json',tactile_stats)
    # The validation/test rows never fit normalization or phase-sampling weights.
    counts=phase_counts['train'];save(root/'phase-sampling.json',dict(fit_split='train',phase_counts=dict(counts),
        inverse_frequency_weights={k:sum(counts.values())/(len(counts)*v) for k,v in counts.items()},
        note='Optional sampling supervision only; teacher phase must not be a policy observation'))
    for split in ('train','validation','test'):
        ds=RobotDataset(root,split,10,include_tactile=True)
        for i in (0,len(ds)//2,len(ds)-1):
            sample=ds[i]
            if sample['observation.state'].shape!=(29,):raise ValueError('Tactile loader shape')
            if sample['action'].shape!=(10,13):raise ValueError('Action loader shape')
            if not np.isfinite(sample['observation.state']).all():raise ValueError('Nonfinite tactile state')
    chosen=[json.loads(p.read_text()) for p in sorted(root.glob('*/group_*/selection.json'))]
    deltas=np.array([r['delta_m'] for r in chosen]);prefixes=collections.Counter(r.get('recovery_prefix_steps',0) for r in chosen)
    failures=json.loads((root/'preflight-index.json').read_text());rejected=[r for r in failures if not r['passed']]
    coverage=dict(accepted_physical_groups=len(chosen),recorded_verified_episodes=len(index['episodes']),
        spawn_xy_actual_min_m=deltas[:,:2].min(0).tolist(),spawn_xy_actual_max_m=deltas[:,:2].max(0).tolist(),
        spawn_radial_distance_quantiles_m=np.quantile(np.linalg.norm(deltas[:,:2],axis=1),[0,.25,.5,.75,1]).tolist(),
        policy_prefix_step_counts=dict(prefixes),phase_counts={s:dict(c) for s,c in phase_counts.items()},
        initial_case_height_above_table_goal_range_m=[min(initial_heights),max(initial_heights)],
        minimum_raised_thumb_opposition_hold_s=min(opposed_holds),
        teacher_placement_error_quantiles_m=np.quantile(placement_errors,[0,.25,.5,.75,1]).tolist(),
        teacher_mean_placement_error_m=float(np.mean(placement_errors)),
        maximum_sampled_finger_normal_force_n=max(peaks),maximum_feedback_action_change_rad=max(corrections),
        preflight_attempts=len(failures),rejected_preflight_attempts=len(rejected),
        training_frames=sum(counts.values()),source_recordings=1,
        caution='Accepted ranges, not just proposed bounds. Rejection-sampled teacher data is not an unbiased success benchmark. Initial-error recovery is not full on-policy DAgger.')
    coverage['per_split_variation']={}
    for split in ('train','validation','test'):
        selected=[r for r in chosen if r['split']==split];xy=np.array([r['delta_m'][:2] for r in selected])
        coverage['per_split_variation'][split]=dict(groups=len(selected),spawn_xy_min_m=xy.min(0).tolist(),
            spawn_xy_max_m=xy.max(0).tolist(),mass_range_kg=[min(r['mass_kg'] for r in selected),max(r['mass_kg'] for r in selected)],
            friction_multiplier_range=[min(r['friction_multiplier'] for r in selected),max(r['friction_multiplier'] for r in selected)])
    save(root/'coverage.json',coverage)
    with (root/'episode-quality.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(quality_rows[0]));writer.writeheader();writer.writerows(quality_rows)
    prep=json.loads((root/'preparation.json').read_text());prep.update(tactile_data_ready=True,
        tactile_input_dimension=29,vision_proprioception_input_dimension=19,tactile_force_dimension=10,
        privileged_fields_not_policy_inputs=['object_qpos','source_timestamp','supervision.phase','supervision.grasp_stable','supervision.raised_grasp','teacher audit labels'],
        executed_target='action',unmodified_teacher_target='action.reference',
        tactile_runtime='Idealized external fingertip force summaries, 20 Hz; not calibrated physical tactile sensors')
    save(root/'preparation.json',prep)
    for name in ('humanoid/contact_collection.py','humanoid/contact_finalize.py','humanoid/contact_render_parallel.py','humanoid/contact_preview.py','humanoid/tactile.py',
                 'training/smolvla.py','training/evaluate.py','humanoid/policy_bridge.py'):
        target=root/'implementation'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    save(root/'feature-contract.json',dict(images=['observation.images.front','observation.images.inspection'],
        joint_names=schema['state_joint_names'],tactile_names=FEATURE_NAMES,action_names=schema['action_actuator_names'],
        actor_state_options=dict(vision_proprioception=19,vision_proprioception_tactile=29),
        target='Actual bounded motor targets including teacher feedback. Do not add the same feedback again when cloning these targets.',
        auxiliary_target='action.reference is retained separately for future hierarchical-controller experiments',
        privileged_supervision='Grasp/raised-grasp labels use simulator object contacts and pose as targets only, not observations',
        tactile_limitations='Normal/shear magnitudes from external distal/intermediate contacts; robot self-contact excluded. No object identity is used. Not a calibrated taxel array or validated slip sensor.'))
    shutil.copy2('training/CONTACT_TRAINING.md',root/'GPU_TRAINING.md')
    shutil.copy2('training/UPSTREAM.json',root/'POLICY_UPSTREAM.json')
    shutil.copy2('scripts/verify_contact_data.py',root/'verify.py')
    text=f'''# Contact-aware G1 / Inspire training data

{len(index['episodes'])} verified robot episodes in {len(chosen)} physical groups,
three visual versions per group. Source motion: one personally recorded Book5 video.
All variants of a physical group stay together in train, validation or test.

The robot starts from a common configuration with small joint errors. Selected
groups additionally replay a short action prefix from the existing policy's
nominal training-scene rollout, then the teacher recovers from that physical state.
These prefixes initialize states; their mistakes are not imitation targets.
The case spawn varies independently of the fixed destination. Wrist approach
clearance, contact-confirmation pauses and bounded force trims adapt the source
motion. This is privileged geometric teaching, not learned-policy success.

Each 20 Hz row contains two RGB images, 19 joint positions, 10 external fingertip
normal/shear force features and 13 actual motor targets, sampled before action.
Next state follows 50 one-millisecond MuJoCo steps. Sensors use external finger
contacts without object identity. Internal robot collisions do not count as touch.
Every episode passes exact state, object and tactile replay checks, image checks,
long-edge layout checks and unchanged grasp/lift/placement/release task checks.
An additional 40 N sampled fingertip-force filter selects this collection; this
is a data-quality criterion, not a hardware safety rating. All failures remain.

Train either the 19-feature vision/proprioception ablation or the 29-feature
tactile model. Statistics are fitted on training rows only. Teacher phase labels
are for balanced sampling, not inference. Object poses and source clocks are
diagnostics, never actor inputs. `action` already includes feedback; do not
silently add the same correction a second time at inference.

Use `data/*.jsonl`, `statistics.json` or `statistics-tactile.json`, and
`training.dataset.RobotDataset(..., include_tactile=True)` for tactile training.
`coverage.json` reports actual accepted variation, force peaks, recovery starts
and phase counts. `preparation.json`, `collection-audit.json` and per-episode
audits establish data readiness, not policy accuracy. Use ALL locked conditions
in `evaluation-scenarios.json`, including teacher failures, after training.

Textures, light, camera, mass, friction, timing, spawn and reset errors vary.
This remains synthetic data from one recording, with approximate scene geometry
and a fixed supported pelvis. It does not establish real-hardware transfer,
general manipulation skill, calibrated tactile sensing or robust slip detection.
GPU training has not started. See the project `training/CONTACT_TRAINING.md`.
'''
    (root/'README.md').write_text(text)
    from .contact_preview import previews
    previews(root,index['episodes'])
    manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'manifest.json'}
    save(root/'manifest.json',dict(files_sha256=manifest,source_video_sha256=schema['source_video_sha256']))
    print(json.dumps(coverage,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('folder',type=Path)
    seal(parser.parse_args().folder.resolve())
