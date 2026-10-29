"""Contact-controlled demonstrations with fixed resets and genuine start errors."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np

from .book_g1 import make_model,ORIGIN,SCENE_ROTATION,BACKGROUND_CONTROL
from .book_transfer import set_configuration,rollout
from .book_variations import randomize_xml,_save_json,_heldout_reset
from .book_dataset_audit import audit,raised_hold_duration
from .kinematics import solve_arm
from .tactile import GripFeedback


def smooth(x):
    x=np.clip(x,0,1);return x*x*(3-2*x)


def warp(model,active,reference,delta,folder):
    """Approach shifted case from fixed start; return to fixed destination."""
    data=mujoco.MjData(model);actions=[];errors=[]
    for t,u in zip(reference['seconds'],reference['actions']):
        full=np.zeros(model.nu);full[active]=u
        for name,value in BACKGROUND_CONTROL.items():full[model.actuator(name).id]=value
        set_configuration(model,data,full);sid=model.site('landmark_0').id
        offset=delta*smooth((t-3.3)/1.)*(1-smooth((t-7.2)/2.))
        # Video-derived thumb sweep grazes the book before grasp formation.
        # Add clearance only during approach; return to the recorded grasp.
        offset[2]+=.012*np.exp(-((t-4.1)/.30)**2)
        if np.linalg.norm(offset)<1e-10:actions.append(u.copy());continue
        target=data.site_xpos[sid].copy()+offset;rotation=data.site_xmat[sid].reshape(3,3).copy()
        q,error=solve_arm(model,data,'right',target,u[:7],rotation,site_name='landmark_0')
        if error>.003:q,error=solve_arm(model,data,'right',target,u[:7],site_name='landmark_0')
        if error>.003:raise RuntimeError(f'Unreachable warped wrist at {t}: {error}')
        moved=u.copy();moved[:7]=q;actions.append(moved);errors.append(error)
    np.savez_compressed(folder/'retarget.npz',seconds=reference['seconds'],actions=actions,observed=reference['observed'])
    _save_json(folder/'trajectory-adaptation.json',dict(fixed_robot_start=True,fixed_destination=True,
        maximum_ik_position_error_m=max(errors,default=0.),spawn_delta_m=delta.tolist(),
        method='Smooth wrist-offset approach/carry warp of personally recorded motion; no object-state playback'))


def quality(folder):
    report=json.loads((folder/'report.json').read_text());x=np.load(folder/'episode.npz')
    hold=raised_hold_duration(x['object_qpos'],json.loads((folder/'contacts.json').read_text()),20)
    peak=float(x['observation_tactile'][:,::2].max())
    failed=[k for k,v in report['checks'].items() if not v]
    if hold<.5:failed.append('sustained_raised_multifinger_hold')
    if peak>40.:failed.append('finger_force_under_40N')
    return dict(passed=not failed,failed_checks=failed,hold_seconds=hold,peak_finger_normal_force_n=peak)


def execute(model,active,state,folder,parameters,goal,record):
    prefix=parameters.get('recovery_prefix_steps',0)
    recovery=np.load('runs/book5-smolvla-evaluation-02/scenario_00/episode.npz')['action'][:prefix] if prefix else None
    rollout(model,Path('runs/book5-motion-06'),folder,record=record,active_aids=active,state_joint_ids=state,
        world_offset=ORIGIN,world_rotation=SCENE_ROTATION,hand_prefix='R_',source_rate=parameters['source_rate'],
        servo_feedforward=True,record_overview=False,background_control=BACKGROUND_CONTROL,
        contact_controller=GripFeedback(parameters['grip_target_n']),
        initial_joint_delta=parameters['initial_joint_delta_rad'],goal_override=goal,initial_recovery_actions=recovery)


def collect(args):
    root=args.output;root.mkdir(parents=True,exist_ok=args.resume)
    reference=Path('runs/book5-g1-ego-09');source=json.loads((reference/'retarget.json').read_text())
    setup=root/'setup'
    if not setup.exists():setup.mkdir();model,active,state=make_model(Path('runs/book5-space-03'),setup)
    else:
        model=mujoco.MjModel.from_xml_path(str((setup/'scene.xml').resolve()))
        schema=json.loads((reference/'dataset.json').read_text());active=np.array(schema['active_actuator_indices'])
        state=np.array([model.joint(n).id for n in schema['state_joint_names']])
    base=(setup/'scene.xml').read_text();trajectory=np.load(reference/'retarget.npz')
    goal=json.loads((reference/'report.json').read_text())['goal_position_m']
    held=max(2,args.groups//5)
    plan=[dict(physical_group=g,split='train' if g<args.groups-2*held else
        ('validation' if g<args.groups-held else 'test'),spawn_bound_m=(.002,.006,.010,.014)[g%4],
        recovery_prefix_steps=(0,10,20,30)[(g//4)%4]) for g in range(args.groups)]
    if (root/'plan.json').exists():
        if json.loads((root/'plan.json').read_text())['groups']!=plan:raise ValueError('Resume must preserve split/condition plan')
    else:_save_json(root/'plan.json',dict(groups=plan,variants=args.variants,prepared_before_collection=True,
        rationale='Physical diversity and corrective starts, not only visual copies. Contact feedback changes actual motor targets.',
        fixed_destination_m=goal,source_sha256=source['source_sha256'],source_recordings=1,
        recovery_actions_source='runs/book5-smolvla-evaluation-02/scenario_00/episode.npz',
        recovery_actions_sha256=hashlib.sha256(Path('runs/book5-smolvla-evaluation-02/scenario_00/episode.npz').read_bytes()).hexdigest(),
        recovery_scope='Only the nominal training-scene policy trajectory is used, never held-out policy trajectories. Prefix actions initialize physical states, not imitation targets.',
        caveat='Synthetic demonstrations from one recording; no claim of independent human data or calibrated hardware touch.'))
    previous=json.loads((root/'index.json').read_text()) if (root/'index.json').exists() else {}
    entries=previous.get('episodes',[]);candidates=json.loads((root/'preflight-index.json').read_text()) if (root/'preflight-index.json').exists() else []
    index=dict(groups=args.groups,variants_per_group=args.variants,build_protocol='contact-feedback-v2',
        source_reference=str(reference.resolve()),source_video_sha256=source['source_sha256'],episodes=entries,
        collection_status='running',learned_policy_evaluated=False,training_started=False)
    _save_json(root/'index.json',index)
    for specification in plan:
        group=specification['physical_group'];split=specification['split'];group_dir=root/split/f'group_{group:02d}'
        group_dir.mkdir(parents=True,exist_ok=True);selection=group_dir/'selection.json'
        chosen=json.loads(selection.read_text()) if selection.exists() else None
        if chosen is None:
            for trial in range(args.max_candidates):
                seed=263000+group*100+trial;rng=np.random.default_rng(seed);bound=specification['spawn_bound_m']
                delta=np.r_[rng.uniform(-bound,bound,2),0.]
                parameters=dict(delta_m=delta.tolist(),mass_kg=float(rng.uniform(.050,.065)),
                    friction_multiplier=float(rng.uniform(.85,1.15)),source_rate=float(rng.uniform(.65,.8)),
                    grip_target_n=float(rng.uniform(5,7)),initial_joint_delta_rad=np.r_[rng.uniform(-.015,.015,7),np.zeros(6)].tolist(),
                    recovery_prefix_steps=specification['recovery_prefix_steps'])
                folder=root/'preflight'/split/f'group_{group:02d}'/f'candidate_{trial:02d}'
                if folder.exists() and (folder/'quality.json').exists():result=json.loads((folder/'quality.json').read_text())
                else:
                    if folder.exists():
                        import uuid
                        folder.rename(folder.with_name(folder.name+'_interrupted_'+uuid.uuid4().hex[:6]))
                    folder.mkdir(parents=True)
                    instance=randomize_xml(base,folder,seed,split,parameters['mass_kg'],parameters['friction_multiplier'],delta)
                    with (folder/'preflight.log').open('w') as stream,contextlib.redirect_stdout(stream):
                        try:
                            warp(model,active,trajectory,delta,folder);execute(instance,active,state,folder,parameters,goal,False)
                            result=quality(folder)
                        except RuntimeError as exc:
                            _heldout_reset(instance,reference,folder);result=dict(passed=False,failed_checks=['teacher_ik_or_dynamics'],error=str(exc))
                    result.update(parameters,physical_group=group,split=split,candidate=trial,seed=seed,
                        folder=str(folder.relative_to(root)),goal_position_m=goal,recorded_images=False)
                    _save_json(folder/'quality.json',result)
                if not any(c['folder']==result['folder'] for c in candidates):candidates.append(result)
                _save_json(root/'preflight-index.json',candidates)
                print(f'Contact preflight {split} group {group} trial {trial}: {result["passed"]} {result["failed_checks"]}',flush=True)
                if result['passed']:chosen=result;_save_json(selection,chosen);break
            if chosen is None:
                index['collection_status']='stopped_no_valid_teacher';_save_json(root/'index.json',index)
                raise RuntimeError(f'No valid contact-controlled demonstration in group {group}; failures retained')
        if args.preflight_only:continue
        for variant in range(args.variants):
            relative=f'{split}/group_{group:02d}/episode_{variant:02d}'
            if any(e['episode']==relative for e in entries):continue
            folder=root/relative
            if folder.exists():
                import uuid
                folder.rename(folder.with_name(folder.name+'_interrupted_'+uuid.uuid4().hex[:6]))
            folder.mkdir();shutil.copy2(root/chosen['folder']/'retarget.npz',folder/'retarget.npz')
            _save_json(folder/'retarget.json',dict(source,physical_group=group,contact_feedback=True,selected_preflight=chosen['folder']))
            instance=randomize_xml(base,folder,400000+group*100+variant,split,chosen['mass_kg'],chosen['friction_multiplier'],np.array(chosen['delta_m']))
            with (folder/'collection.log').open('w') as stream,contextlib.redirect_stdout(stream):
                execute(instance,active,state,folder,chosen,goal,True)
                try:audit(folder,check_render=True)
                except SystemExit:pass
            verified=json.loads((folder/'audit.json').read_text())['passed'];q=quality(folder)
            if not verified or not q['passed']:
                index['collection_status']='stopped_failed_recorded_variant';_save_json(root/'index.json',index)
                raise RuntimeError(f'Recorded variant failed {relative}; retained and excluded')
            schema=json.loads((folder/'dataset.json').read_text());x=np.load(folder/'episode.npz')
            with (folder/'frames.jsonl').open('w') as stream:
                for i in range(len(x['action'])):
                    row=dict(frame_index=i,timestamp=float(x['timestamp'][i]),task=schema['task'],**{
                        'observation.state':x['observation_state'][i].tolist(),'observation.tactile':x['observation_tactile'][i].tolist(),
                        'action':x['action'][i].tolist(),
                        'observation.images.front':f'images/observation.images.front/{i:06d}.jpg',
                        'observation.images.inspection':f'images/observation.images.inspection/{i:06d}.jpg'})
                    stream.write(json.dumps(row)+'\n')
            entries.append(dict(episode=relative,physical_group=group,split=split,domain_seed=400000+group*100+variant,
                frames=len(x['action']),task_success=True,verified=True,training_eligible=split=='train',preflight=chosen['folder']))
            _save_json(root/'index.json',index);print(f'Recorded verified tactile episode {relative}: {len(x["action"])} frames',flush=True)
    index['collection_status']='preflight_complete' if args.preflight_only else 'complete';_save_json(root/'index.json',index)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--groups',type=int,default=32);p.add_argument('--variants',type=int,default=3)
    p.add_argument('--max-candidates',type=int,default=8);p.add_argument('--resume',action='store_true')
    p.add_argument('--preflight-only',action='store_true',help='Verify every physical group before rendering the image collection')
    a=p.parse_args()
    if a.groups<6:raise SystemExit('At least six physical groups required')
    collect(a)
