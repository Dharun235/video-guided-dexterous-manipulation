"""Audit synchronized book-transfer episodes and replay their actual actions."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .book_transfer import front_sensor_image
from .layout import verify_layout


def raised_hold_duration(object_qpos,contact_rows,fps,prefix='R_'):
    """Raised grasp duration; count each anatomical finger only once."""
    longest=current=0
    for i,row in enumerate(contact_rows):
        fingers={c['body'][len(prefix):].rsplit('_',1)[0] for c in row['contacts'] if c['body'].startswith(prefix)
                 and c['body'].endswith(('distal','intermediate')) and c['normal_force_n']>.01}
        held=len(fingers)>=2 and object_qpos[i,2]>object_qpos[0,2]+.02
        current=current+1 if held else 0;longest=max(longest,current)
    return longest/fps


def audit(folder,check_render=False):
    schema=json.loads((folder/'dataset.json').read_text())
    report=json.loads((folder/'report.json').read_text())
    x=np.load(folder/'episode.npz');n=len(x['timestamp'])
    model=mujoco.MjModel.from_xml_path(str((folder/'scene.xml').resolve()))
    layout=verify_layout(model,folder)
    data=mujoco.MjData(model)
    mujoco.mj_setState(model,data,np.load(folder/'initial_state.npz')['integration'],mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model,data)
    qadr=schema['state_qpos_indices']
    limits=np.array(schema['action_ctrlrange'])
    rendered=[];valid_images=True;views=('front','inspection')
    digest=hashlib.sha256();sheet=[]
    for i in range(n):
        for view in views:
            p=folder/f'images/observation.images.{view}'/f'{i:06d}.jpg'
            if not p.is_file():valid_images=False;continue
            pixels=cv2.imread(str(p));valid_images &= pixels is not None and pixels.shape==(256,256,3)
            digest.update(p.read_bytes())
        if i in (0,34,54,94,124,n-1) and valid_images:
            a=cv2.imread(str(folder/'images/observation.images.front'/f'{i:06d}.jpg'))
            b=cv2.imread(str(folder/'images/observation.images.inspection'/f'{i:06d}.jpg'))
            panel=np.hstack([a,b]);cv2.rectangle(panel,(0,0),(512,25),(20,20,20),-1)
            cv2.putText(panel,f'frame {i} | source {x["source_timestamp"][i]:.2f}s',(7,18),0,.45,(255,255,255),1)
            sheet.append(panel)
    renderer=mujoco.Renderer(model,height=256,width=256) if check_render else None
    state_error=0.;next_error=0.;case_error=0.;image_errors=[];visible=True;minimum_margin=256.;tactile_error=0.
    cq=model.jnt_qposadr[model.joint('case_free').id]
    for i,u in enumerate(x['action']):
        if 'observation_tactile' in x:
            from .tactile import sense
            tactile_error=max(tactile_error,float(np.max(np.abs(sense(model,data)-x['observation_tactile'][i]))))
        for camera_name in ('front_dataset','inspection_dataset'):
            cid=model.camera(camera_name).id
            # Conservative 3D case bounding box; this checks framing, not visibility
            # through the fingers or correctness of the reconstructed object shape.
            corners=np.array([[a,b,c] for a in (-.031,.031) for b in (-.023,.023) for c in (-.014,.014)])
            bid=model.body('case').id
            world=corners@data.xmat[bid].reshape(3,3).T+data.xpos[bid]
            camera=(world-data.cam_xpos[cid])@data.cam_xmat[cid].reshape(3,3)
            focal=128/np.tan(np.deg2rad(model.cam_fovy[cid])/2)
            pixels=np.column_stack([128+focal*camera[:,0]/-camera[:,2],128-focal*camera[:,1]/-camera[:,2]])
            margin=float(min(pixels.min(),255-pixels.max()))
            if camera_name=='front_dataset' and mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,'source_match')>=0:
                top=(256-round(256*464/832))//2
                margin=min(margin,float(pixels[:,1].min()-top),float(top+round(256*464/832)-1-pixels[:,1].max()))
            visible &= bool((camera[:,2]<0).all() and margin>=2)
            minimum_margin=min(minimum_margin,margin)
        state_error=max(state_error,float(np.max(np.abs(data.qpos[qadr]-x['observation_state'][i]))))
        case_error=max(case_error,float(np.max(np.abs(data.qpos[cq:cq+7]-x['object_qpos'][i]))))
        if renderer and i in (0,34,94,n-1):
            renderer.update_scene(data,camera='front_dataset');rgb=front_sensor_image(model,renderer.render().copy())
            # Compare the same JPEG encoding rather than lossy JPEG to raw RGB.
            _,encoded=cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            expected=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
            actual=cv2.imread(str(folder/'images/observation.images.front'/f'{i:06d}.jpg'))
            image_errors.append(float(np.mean(np.abs(expected.astype(float)-actual.astype(float)))))
        data.ctrl[schema.get('active_actuator_indices',list(range(model.nu)))]=u
        for _ in range(50):mujoco.mj_step(model,data)
        next_error=max(next_error,float(np.max(np.abs(data.qpos[qadr]-x['next_observation_state'][i]))))
    if renderer:renderer.close()
    counts={view:len(list((folder/f'images/observation.images.{view}').glob('*.jpg'))) for view in views}
    contact_rows=json.loads((folder/'contacts.json').read_text())
    hold_seconds=raised_hold_duration(x['object_qpos'],contact_rows,schema['fps'],schema.get('hand_contact_prefix','R_'))
    checks=dict(finite_arrays=all(np.isfinite(x[k]).all() for k in x.files),
                robot_faces_table_long_edge=layout['passed'],
                aligned_lengths=all(len(x[k])==n for k in x.files),
                camera_images_complete=valid_images and all(count==n for count in counts.values()),
                state_schema=x['observation_state'].shape==(n,len(schema['state_joint_names'])),
                action_schema=x['action'].shape==(n,len(schema['action_actuator_names'])),
                action_within_limits=bool(((x['action']>=limits[:,0]-1e-7)&(x['action']<=limits[:,1]+1e-7)).all()),
                regular_timestamps=bool(np.allclose(np.diff(x['timestamp']),1/schema['fps'],atol=1e-8)),
                next_state_alignment=bool(np.allclose(x['next_observation_state'][:-1],x['observation_state'][1:],atol=1e-10)),
                physics_replay_reproduces_states=state_error<1e-7 and next_error<1e-7 and case_error<1e-7,
                case_inside_both_camera_frames=visible,
                sustained_raised_multifinger_hold=hold_seconds>=.5,
                task_checks_pass=bool(report['task_success']),
                no_grasp_assistance=not any(report['physics'][k] for k in ('welds','external_object_forces','object_pose_updates_after_initialization')))
    if check_render:checks['sampled_images_match_state']=max(image_errors)<.5
    if 'observation_tactile' in x:
        checks['tactile_shape']=x['observation_tactile'].shape==(n,10)
        checks['tactile_matches_physics_replay']=tactile_error<1e-6
    result=dict(checks=checks,passed=all(checks.values()),frames=n,camera_image_counts=counts,
                grasp_metric_version='distinct-anatomical-fingers-v2',
                state_dimension=x['observation_state'].shape[1],action_dimension=x['action'].shape[1],
                maximum_replay_state_error=state_error,maximum_replay_next_state_error=next_error,
                maximum_replay_object_error=case_error,sampled_image_mean_absolute_errors=image_errors,
                maximum_replay_tactile_error_n=tactile_error,
                minimum_case_frame_margin_pixels=minimum_margin,
                longest_raised_multifinger_hold_s=hold_seconds,
                image_sha256=digest.hexdigest(),episode_sha256=hashlib.sha256((folder/'episode.npz').read_bytes()).hexdigest(),
                scope='One privileged-teacher pilot episode; verifies synchronization, reproducibility and this task instance. Not policy generalization or real-world validity.')
    (folder/'audit.json').write_text(json.dumps(result,indent=2))
    report['training_ready']=result['passed'];report['dataset_status']='Audited pilot episode' if result['passed'] else 'Audit failed; exclude from training'
    schema['training_ready']=result['passed']
    (folder/'report.json').write_text(json.dumps(report,indent=2));(folder/'dataset.json').write_text(json.dumps(schema,indent=2))
    if len(sheet)==6:cv2.imwrite(str(folder/'review-sheet.jpg'),np.vstack([np.hstack(sheet[i:i+2]) for i in range(0,6,2)]))
    if (folder/'manifest.json').exists():
        manifest=json.loads((folder/'manifest.json').read_text())
        manifest['files_sha256']={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='manifest.json'}
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('folder',type=Path);p.add_argument('--check-render',action='store_true')
    a=p.parse_args();audit(a.folder,a.check_render)
