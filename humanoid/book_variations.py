"""Collect seeded G1/Inspire physics episodes with grouped held-out conditions.

All variants of one physical teacher trajectory stay in the same split. This
collects demonstrations and evaluates the teacher, not a trained policy.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .book_g1 import make_model,ORIGIN,SCENE_ROTATION,LAYOUT_VERSION,BACKGROUND_CONTROL
from .book_transfer import set_configuration,rollout
from .book_dataset_audit import audit
from .kinematics import solve_arm


def texture(rng,kind):
    y,x=np.mgrid[:256,:256];color=rng.uniform(.25,.8,3)
    if kind=='wood':signal=.12*np.sin(y*.22+2*np.sin(x*.018))+.04*rng.normal(size=x.shape)
    elif kind=='checker':signal=.12*((x//32+y//32)%2)*2-.12
    elif kind=='speckle':signal=.07*rng.normal(size=x.shape)
    elif kind=='stripes':signal=.15*np.sin((x+y)*.12)
    else:signal=.16*np.sin(x*.08)*np.cos(y*.04)
    return np.uint8(np.clip(color+signal[:,:,None],.03,.95)*255)


def randomize_xml(base_xml,folder,seed,split,mass,friction,delta):
    rng=np.random.default_rng(seed);root=ET.fromstring(base_xml)
    families=['wood','checker','speckle'] if split=='train' else (['stripes'] if split=='validation' else ['waves'])
    choices={}
    for name in ('table','wall'):
        kind=str(rng.choice(families));choices[name]=kind
        path=(folder/f'{name}-domain.png').resolve();cv2.imwrite(str(path),texture(rng,kind))
        root.find(f"asset/texture[@name='{name}']").set('file',str(path))
    case=root.find(".//body[@name='case']")
    case.set('pos',' '.join(map(str,np.fromstring(case.get('pos'),sep=' ')+delta)))
    # Friction is a multiplier around the approved model, not a replacement for
    # every material. The hand is 1.0, while case/book/table are 0.7 at baseline.
    solid=case.find("geom[@name='case_solid']");solid.set('mass',str(mass));solid.set('friction',f'{.7*friction} .005 .001')
    for name in ('table_collision','book_collision'):
        geom=root.find(f".//geom[@name='{name}']")
        if geom is not None:geom.set('friction',f'{.7*friction} .005 .001')
    for body in root.findall('.//body'):
        if body.get('name','').startswith('R_'):
            for geom in body.findall('geom'):
                if geom.get('contype','1')!='0':geom.set('friction',f'{friction} .005 .001')
    shade=rng.uniform(.025,.10);solid.set('rgba',f'{shade} {shade} {shade} 1')
    for node in root.iter():
        for light in list(node.findall('light')):node.remove(light)
    light_position=np.array([.35,-.4,1.7])+rng.uniform([-.25,-.25,-.15],[.25,.25,.3])
    diffuse=rng.uniform(.30,.50,3)
    ET.SubElement(root.find('worldbody'),'light',pos=' '.join(map(str,light_position)),dir='0 0 -1',
                  diffuse=' '.join(map(str,diffuse)),castshadow='true')
    visual=root.find('visual')
    if visual is None:visual=ET.SubElement(root,'visual')
    quality=visual.find('quality')
    if quality is None:quality=ET.SubElement(visual,'quality')
    quality.set('shadowsize','512');quality.set('offsamples','0')
    headlight=visual.find('headlight')
    if headlight is None:headlight=ET.SubElement(visual,'headlight')
    ambient=rng.uniform(.15,.25);headlight.set('ambient',f'{ambient} {ambient} {ambient}');headlight.set('diffuse','.12 .12 .12');headlight.set('specular','.05 .05 .05')
    cameras={}
    for name in ('front_dataset','inspection_dataset'):
        camera=root.find(f".//camera[@name='{name}']")
        position=np.fromstring(camera.get('pos'),sep=' ')+rng.uniform(-.012,.012,3)
        base_q=np.fromstring(camera.get('quat'),sep=' ')
        orientation=Rotation.from_euler('xyz',rng.uniform(-2,2,3),degrees=True)*Rotation.from_quat(base_q[[1,2,3,0]])
        q=orientation.as_quat()[[3,0,1,2]]
        base_fovy=float(camera.get('fovy'))
        fovy=float(base_fovy*(rng.uniform(.97,1.03) if split=='train' else rng.uniform(1.06,1.12)))
        camera.set('pos',' '.join(map(str,position)));camera.set('quat',' '.join(map(str,q)));camera.set('fovy',str(fovy))
        cameras[name]=dict(position=position.tolist(),quaternion_wxyz=q.tolist(),fovy_degrees=fovy,
                           parent='torso_link' if name=='front_dataset' else 'world')
    metadata=dict(seed=seed,split=split,textures=choices,light_position=light_position.tolist(),
                  light_diffuse=diffuse.tolist(),headlight_ambient=ambient,cameras=cameras,
                  object_mass_kg=mass,friction_multiplier=friction,hand_sliding_friction=friction,
                  object_table_sliding_friction=.7*friction,object_spawn_delta_m=delta.tolist(),
                  note='Parameters fixed within an episode. Background pixels are procedural, not new personally recorded data.')
    (folder/'domain.json').write_text(json.dumps(metadata,indent=2))
    xml=ET.tostring(root,encoding='unicode');(folder/'scene.xml').write_text(xml)
    return mujoco.MjModel.from_xml_string(xml)


def translate_teacher(model,active,reference,delta,folder):
    data=mujoco.MjData(model);actions=[];fallbacks=[]
    for t,u in zip(reference['seconds'],reference['actions']):
        full=np.zeros(model.nu);full[active]=u;set_configuration(model,data,full)
        sid=model.site('landmark_0').id;target=data.site_xpos[sid]+delta
        rotation=data.site_xmat[sid].reshape(3,3).copy()
        q,error=solve_arm(model,data,'right',target,u[:7],rotation,site_name='landmark_0')
        if error>.003:
            # A few millimetres can put a fixed wrist orientation outside the
            # arm's reachable set. Preserve position and allow orientation to
            # adapt; actual grasp/placement/collision checks still gate the data.
            original_error=error
            q,error=solve_arm(model,data,'right',target,u[:7],site_name='landmark_0')
            fallbacks.append(dict(source_seconds=float(t),fixed_orientation_error_m=original_error,
                                  position_only_error_m=error))
        if error>.003:raise RuntimeError(f'Unreachable translated wrist at {t}: {error}')
        moved=u.copy();moved[:7]=q;actions.append(moved)
    np.savez_compressed(folder/'retarget.npz',seconds=reference['seconds'],actions=actions,observed=reference['observed'])
    (folder/'ik-adaptations.json').write_text(json.dumps(dict(position_only_fallbacks=fallbacks),indent=2))


def _save_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def _heldout_reset(model,reference,folder):
    """A valid nominal robot reset when the teacher cannot solve a trajectory."""
    data=mujoco.MjData(model)
    mujoco.mj_setState(model,data,np.load(reference/'initial_state.npz')['integration'],mujoco.mjtState.mjSTATE_INTEGRATION)
    q=model.jnt_qposadr[model.joint('case_free').id];data.qpos[q:q+7]=model.qpos0[q:q+7]
    mujoco.mj_forward(model,data)
    state=np.zeros(mujoco.mj_stateSize(model,mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model,data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
    np.savez_compressed(folder/'initial_state.npz',integration=state)


def collect(reference,output,groups=12,variants=3,resume=False,max_candidates=16):
    from .book_dataset_audit import raised_hold_duration
    if not json.loads((reference/'audit.json').read_text())['passed']:
        raise ValueError('Reference must pass the physical episode audit before collection')
    source=json.loads((reference/'retarget.json').read_text())
    if source.get('layout_version')!=LAYOUT_VERSION:
        raise ValueError('Reference must use the approved ego/front layout')
    output.mkdir(parents=True,exist_ok=resume);setup=output/'setup'
    if resume and (setup/'scene.xml').exists():
        old=json.loads((output/'index.json').read_text())
        if old.get('build_protocol')!='preflight-filtered-v1':raise ValueError('Use a new folder for the corrected collection protocol')
        if old['groups']!=groups or old['variants_per_group']!=variants:raise ValueError('Resume must preserve the original split plan')
        model=mujoco.MjModel.from_xml_path(str((setup/'scene.xml').resolve()))
        schema=json.loads((reference/'dataset.json').read_text())
        active=np.array(schema['active_actuator_indices']);state=np.array([model.joint(n).id for n in schema['state_joint_names']])
    else:
        setup.mkdir();model,active,state=make_model(Path('runs/book5-space-03'),setup)
    base_xml=(setup/'scene.xml').read_text();r=np.load(reference/'retarget.npz')
    entries=json.loads((output/'index.json').read_text())['episodes'] if resume and (output/'index.json').exists() else []
    candidates=json.loads((output/'preflight-index.json').read_text()) if resume and (output/'preflight-index.json').exists() else []
    completed={e['episode'] for e in entries}
    summary=dict(episodes=entries,groups=groups,variants_per_group=variants,build_protocol='preflight-filtered-v1',
        source_reference=str(reference.resolve()),source_video_sha256=source['source_sha256'],layout_version=LAYOUT_VERSION,
        split_rule='Each movement slot is assigned before sampling; all its appearance variants remain together.',
        selection='Success demonstrations are rejection sampled with unchanged physical checks; all preflight failures are retained.',
        variation_ranges=dict(spawn_xy_m=.004,mass_kg=[.052,.058],friction_multiplier=[.9,1.1],source_rate=[.65,.8]),
        friction_baseline=dict(hand=1.,case_book_table=.7),learned_policy_evaluated=False,training_started=False,collection_status='running')
    _save_json(output/'index.json',summary)
    source_poses=json.loads(Path('runs/book5-motion-06/case_poses.json').read_text())
    base_goal=SCENE_ROTATION@np.array(source_poses[-1]['position_m'])+ORIGIN
    for group in range(groups):
        split='train' if group<groups-4 else ('validation' if group<groups-2 else 'test')
        group_folder=output/split/f'group_{group:02d}';group_folder.mkdir(parents=True,exist_ok=resume)
        selection_path=group_folder/'selection.json'
        chosen=json.loads(selection_path.read_text()) if selection_path.exists() else None
        if chosen is None:
            for trial in range(max_candidates):
                seed=71000+group*100+trial
                rng=np.random.default_rng(seed);delta=np.r_[rng.uniform(-.004,.004,2),0.]
                mass=float(rng.uniform(.052,.058));friction=float(rng.uniform(.9,1.1));rate=float(rng.uniform(.65,.8))
                candidate=output/'preflight'/split/f'group_{group:02d}'/f'candidate_{trial:02d}'
                if candidate.exists() and not (candidate/'quality.json').exists():
                    import uuid
                    candidate.rename(candidate.with_name(candidate.name+'_interrupted_'+uuid.uuid4().hex[:6]))
                candidate.mkdir(parents=True,exist_ok=True)
                if (candidate/'quality.json').exists():quality=json.loads((candidate/'quality.json').read_text())
                else:
                    instance=randomize_xml(base_xml,candidate,seed,split,mass,friction,delta)
                    error=None
                    with (candidate/'preflight.log').open('w') as log,contextlib.redirect_stdout(log):
                        try:
                            translate_teacher(model,active,r,delta,candidate)
                            rollout(instance,Path('runs/book5-motion-06'),candidate,record=False,active_aids=active,state_joint_ids=state,
                                world_offset=ORIGIN+delta,world_rotation=SCENE_ROTATION,hand_prefix='R_',
                                embodiment='Unitree G1 with official Unitree Inspire five-finger right hand; fixed pelvis',
                                zero_finger_start=7,source_rate=rate,servo_feedforward=True,background_control=BACKGROUND_CONTROL)
                            report=json.loads((candidate/'report.json').read_text());x=np.load(candidate/'episode.npz')
                            hold=raised_hold_duration(x['object_qpos'],json.loads((candidate/'contacts.json').read_text()),20)
                            passed=report['task_success'] and hold>=.5
                            failed=[k for k,v in report['checks'].items() if not v]
                            if hold<.5:failed.append('sustained_raised_multifinger_hold')
                        except RuntimeError as exc:
                            error=str(exc);hold=0.;passed=False;failed=['teacher_ik'];_heldout_reset(instance,reference,candidate)
                    quality=dict(physical_group=group,split=split,candidate=trial,seed=seed,passed=passed,hold_seconds=hold,
                        failed_checks=failed,error=error,folder=str(candidate.relative_to(output)),
                        delta_m=delta.tolist(),mass_kg=mass,friction_multiplier=friction,source_rate=rate,
                        goal_position_m=(base_goal+delta).tolist(),recorded_images=False)
                    _save_json(candidate/'quality.json',quality)
                if not any(c['folder']==quality['folder'] for c in candidates):candidates.append(quality)
                _save_json(output/'preflight-index.json',candidates)
                print(f'Preflight {split} group {group:02d} candidate {trial:02d}: {quality["passed"]}',flush=True)
                if quality['passed']:
                    chosen=quality;_save_json(selection_path,chosen)
                    shutil.copy2(output/chosen['folder']/'retarget.npz',group_folder/'retarget.npz')
                    break
            if chosen is None:
                summary['collection_status']='blocked_no_valid_teacher_group';_save_json(output/'index.json',summary)
                raise RuntimeError(f'No verified teacher for {split} group {group}; failures retained, no checks relaxed')
        delta=np.array(chosen['delta_m']);mass=chosen['mass_kg'];friction=chosen['friction_multiplier'];rate=chosen['source_rate']
        for variant in range(variants):
            folder=group_folder/f'episode_{variant:02d}'
            if str(folder.relative_to(output)) in completed:continue
            if folder.exists():
                import uuid
                folder.rename(folder.with_name(folder.name+'_interrupted_'+uuid.uuid4().hex[:6]))
            folder.mkdir();shutil.copy2(group_folder/'retarget.npz',folder/'retarget.npz')
            _save_json(folder/'retarget.json',dict(source,physical_group=group,translation_m=delta.tolist(),selected_preflight=chosen['folder']))
            seed=91000+group*100+variant
            instance=randomize_xml(base_xml,folder,seed,split,mass,friction,delta)
            with (folder/'collection.log').open('w') as stream,contextlib.redirect_stdout(stream):
                rollout(instance,Path('runs/book5-motion-06'),folder,active_aids=active,state_joint_ids=state,
                    world_offset=ORIGIN+delta,world_rotation=SCENE_ROTATION,hand_prefix='R_',
                    embodiment='Unitree G1 with official Unitree Inspire five-finger right hand; fixed pelvis',
                    zero_finger_start=7,source_rate=rate,servo_feedforward=True,record_overview=False,background_control=BACKGROUND_CONTROL)
                try:audit(folder,check_render=True)
                except SystemExit:pass
            report=json.loads((folder/'report.json').read_text());audited=json.loads((folder/'audit.json').read_text())
            schema=json.loads((folder/'dataset.json').read_text());x=np.load(folder/'episode.npz')
            with (folder/'frames.jsonl').open('w') as stream:
                for i in range(report['frames']):
                    stream.write(json.dumps(dict(frame_index=i,timestamp=float(x['timestamp'][i]),task=schema['task'],
                        **{'observation.state':x['observation_state'][i].tolist(),'action':x['action'][i].tolist(),
                           'observation.images.front':f'images/observation.images.front/{i:06d}.jpg',
                           'observation.images.inspection':f'images/observation.images.inspection/{i:06d}.jpg'}))+'\n')
            entry=dict(episode=str(folder.relative_to(output)),physical_group=group,split=split,domain_seed=seed,
                frames=report['frames'],task_success=report['task_success'],verified=audited['passed'],
                training_eligible=audited['passed'] and split=='train',preflight=chosen['folder'])
            entries.append(entry);summary['episodes']=entries;_save_json(output/'index.json',summary)
            print(f'Recorded {split} group {group:02d} variant {variant}: task={report["task_success"]}, audit={audited["passed"]}',flush=True)
    counts={s:dict(attempted=sum(e['split']==s for e in entries),verified=sum(e['split']==s and e['verified'] for e in entries),
        samples=sum(e['frames'] for e in entries if e['split']==s and e['verified'])) for s in ('train','validation','test')}
    _save_json(output/'summary.json',dict(counts=counts,preflight_candidates=len(candidates),
        selection='Quality-filtered teacher demonstrations; rejection sampling is not an unbiased policy success benchmark.',
        warning='Teacher quality is not learned-policy generalization.'))
    summary['collection_status']='complete';_save_json(output/'index.json',summary)
    print(json.dumps(counts,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('reference',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--groups',type=int,default=12);p.add_argument('--variants',type=int,default=3)
    p.add_argument('--resume',action='store_true');p.add_argument('--max-candidates',type=int,default=16)
    a=p.parse_args()
    if a.groups<6:raise SystemExit('Use at least six independent physical groups for three splits')
    collect(a.reference,a.output,a.groups,a.variants,a.resume,a.max_candidates)
