"""G1 actuator execution, camera observations and synchronized dataset capture.

The teacher uses geometric references; the free case moves through physics.
"""
import json
from pathlib import Path
import subprocess
import time

import cv2
import mujoco
import numpy as np



def front_sensor_image(model,rgb):
    """Keep the phone's 16:9 content in a square policy image without mirroring."""
    if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,'source_match')<0:return rgb
    height,width=rgb.shape[:2];content=min(height,round(width*464/832));top=(height-content)//2
    result=np.zeros_like(rgb);result[top:top+content]=rgb[top:top+content]
    return result




def set_configuration(model,data,control):
    """Kinematic IK workspace only. Never called on the running rollout data."""
    for aid,value in enumerate(control):
        trn=model.actuator_trnid[aid,0]
        if model.actuator_trntype[aid]==mujoco.mjtTrn.mjTRN_JOINT:
            data.qpos[model.jnt_qposadr[trn]]=value
        else:
            adr,num=model.tendon_adr[trn],model.tendon_num[trn]
            for wrap in range(adr,adr+num):
                jid=model.wrap_objid[wrap]
                data.qpos[model.jnt_qposadr[jid]]=value/num
    for eid in range(model.neq):
        if model.eq_type[eid]==mujoco.mjtEq.mjEQ_JOINT:
            dependent,driver=model.eq_obj1id[eid],model.eq_obj2id[eid]
            v=data.qpos[model.jnt_qposadr[driver]]
            data.qpos[model.jnt_qposadr[dependent]]=sum(model.eq_data[eid,k]*v**k for k in range(5))
    mujoco.mj_kinematics(model,data)






def rollout(model,motion,output,*,record=True,offset=(0,0,0),start=3.3,end=12.45,zero_fingers=False,
            active_aids=None,state_joint_ids=None,world_offset=(0,0,0),world_rotation=None,hand_prefix='rh_',
            embodiment='Unitree G1 with Inspire right hand; fixed pelvis',zero_finger_start=7,
            source_rate=1.,servo_feedforward=False,record_overview=True,background_control=None,
            contact_controller=None,initial_joint_delta=None,goal_override=None,maximum_seconds=30.,initial_recovery_actions=None):
    reference=np.load(output/'retarget.npz');times=reference['seconds'];targets=reference['actions']
    data=mujoco.MjData(model)
    active_aids=np.arange(model.nu) if active_aids is None else np.array(active_aids)
    def action(t):
        result=np.array([np.interp(t,times,targets[:,i]) for i in range(targets.shape[1])])
        if zero_fingers:result[zero_finger_start:]=0.
        if servo_feedforward:
            ahead=np.array([np.interp(t+.025,times,targets[:,i]) for i in range(targets.shape[1])])
            behind=np.array([np.interp(t-.025,times,targets[:,i]) for i in range(targets.shape[1])])
            velocity=(ahead-behind)/.05*source_rate
            for j,aid in enumerate(active_aids):
                joint=model.actuator_trnid[aid,0];dof=model.jnt_dofadr[joint]
                kp=model.actuator_gainprm[aid,0];kv=-model.actuator_biasprm[aid,2]
                if kp>0:result[j]+=(data.qfrc_bias[dof]+kv*velocity[j])/kp
            result=np.clip(result,*model.actuator_ctrlrange[active_aids].T)
        return result
    initial=action(start)
    initial_full=np.zeros(model.nu)
    for name,value in (background_control or {}).items():initial_full[model.actuator(name).id]=value
    initial_full[active_aids]=initial
    set_configuration(model,data,initial_full);data.ctrl[:]=initial_full
    mujoco.mj_forward(model,data)
    if initial_joint_delta is not None:
        for aid,delta in zip(active_aids,initial_joint_delta):
            jid=model.actuator_trnid[aid,0];adr=model.jnt_qposadr[jid]
            data.qpos[adr]=np.clip(data.qpos[adr]+delta,*model.jnt_range[jid])
        mujoco.mj_forward(model,data)
    if initial_recovery_actions is not None:
        for prior in initial_recovery_actions:
            data.ctrl[active_aids]=np.clip(prior,*model.actuator_ctrlrange[active_aids].T)
            for _ in range(50):mujoco.mj_step(model,data)
    state_type=mujoco.mjtState.mjSTATE_INTEGRATION
    initial_integration=np.zeros(mujoco.mj_stateSize(model,state_type))
    mujoco.mj_getState(model,data,initial_integration,state_type)
    np.savez_compressed(output/'initial_state.npz',integration=initial_integration)
    episode_start_time=float(data.time)
    cb=model.body('case').id;cq=model.jnt_qposadr[model.joint('case_free').id]
    initial_case=data.qpos[cq:cq+3].copy()
    renderer=mujoco.Renderer(model,height=256,width=256) if record else None
    if record:
        for view in ('front','inspection'):(output/f'images/observation.images.{view}').mkdir(parents=True)
        overview=record_overview
        writer=cv2.VideoWriter(str(output/'execution-raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(768 if overview else 512,256))
    states=[];actions=[];next_states=[];velocities=[];stamp=[];source_times=[];object_states=[];contact_rows=[]
    joint_ids=np.array([j for j in range(model.njnt) if model.joint(j).name!='case_free']) if state_joint_ids is None else np.array(state_joint_ids)
    qadr=model.jnt_qposadr[joint_ids];dadr=model.jnt_dofadr[joint_ids]
    source=json.loads((motion/'case_poses.json').read_text())
    frame=0;started=time.monotonic();max_lift=0.;max_contact=0;max_penetration=0.;clock=start;tactile_rows=[];reference_actions=[]
    while (clock if contact_controller else start+frame/20*source_rate) < end and frame/20<maximum_seconds:
        t=clock if contact_controller else start+frame/20*source_rate;u=action(t);advance=True
        if contact_controller:
            reference_actions.append(u.copy())
            u,advance,tactile=contact_controller(t,u,model,data)
            tactile_rows.append(tactile.copy());u=np.clip(u,*model.actuator_ctrlrange[active_aids].T)
        # observation/state at t, action held on [t,t+dt), next_state at t+dt.
        if renderer:
            renderer.update_scene(data,camera='front_dataset');rgb=front_sensor_image(model,renderer.render().copy())
            cv2.imwrite(str(output/'images/observation.images.front'/f'{frame:06d}.jpg'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            renderer.update_scene(data,camera='inspection_dataset');orbit=renderer.render().copy()
            cv2.imwrite(str(output/'images/observation.images.inspection'/f'{frame:06d}.jpg'),cv2.cvtColor(orbit,cv2.COLOR_RGB2BGR))
            views=[rgb,orbit]
            if overview:
                renderer.update_scene(data,camera='robot_overview');views.append(renderer.render().copy())
            display=np.hstack(views);cv2.putText(display,'Actuated hand | free case | no weld',(5,18),0,.38,(255,255,255),1)
            writer.write(cv2.cvtColor(display,cv2.COLOR_RGB2BGR))
        states.append(data.qpos[qadr].copy());velocities.append(data.qvel[dadr].copy());actions.append(u)
        stamp.append(data.time-episode_start_time);source_times.append(t);object_states.append(data.qpos[cq:cq+7].copy())
        # Explicit contact indexing for measured force, sampled before action.
        row=[]
        for ci in range(data.ncon):
            c=data.contact[ci];b=[model.geom_bodyid[c.geom1],model.geom_bodyid[c.geom2]]
            if cb not in b:continue
            other=b[1] if b[0]==cb else b[0];name=model.body(other).name or 'world'
            force=np.zeros(6);mujoco.mj_contactForce(model,data,ci,force)
            row.append(dict(body=name,normal_force_n=float(force[0]),distance_m=float(c.dist)))
        contact_rows.append(dict(frame=frame,seconds=data.time-episode_start_time,contacts=row))
        max_contact=max(max_contact,sum(c['body'].startswith(hand_prefix) for c in row))
        data.ctrl[active_aids]=u
        for _ in range(50):
            mujoco.mj_step(model,data)
            for c in data.contact:
                max_penetration=max(max_penetration,float(-c.dist))
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():raise RuntimeError('Nonfinite dynamics')
        next_states.append(data.qpos[qadr].copy());max_lift=max(max_lift,float(data.qpos[cq+2]-initial_case[2]))
        frame+=1
        if contact_controller and advance:clock+=source_rate/20
    if renderer:
        writer.release();renderer.close()
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(output/'execution-raw.mp4'),'-c:v','libx264','-pix_fmt','yuv420p',str(output/'execution.mp4')],check=True)
    arrays=dict(timestamp=np.array(stamp),source_timestamp=np.array(source_times),
                observation_state=np.array(states),observation_velocity=np.array(velocities),
                action=np.array(actions),next_observation_state=np.array(next_states),object_qpos=np.array(object_states))
    if contact_controller:
        arrays['observation_tactile']=np.array(tactile_rows)
        arrays['action_reference']=np.array(reference_actions)
    np.savez_compressed(output/'episode.npz',**arrays)
    (output/'contacts.json').write_text(json.dumps(contact_rows,indent=2))
    goal=(np.eye(3) if world_rotation is None else world_rotation)@np.array(source[-1]['position_m'])+world_offset;final=data.qpos[cq:cq+3].copy()
    if goal_override is not None:goal=np.asarray(goal_override)
    stable_speed=float(np.linalg.norm(data.qvel[model.jnt_dofadr[model.joint('case_free').id]:][:3]))
    contact_frames=sum(any(c['body'].startswith(hand_prefix) and c['normal_force_n']>.01 for c in r['contacts']) for r in contact_rows)
    checks=dict(finite_arrays=all(np.isfinite(v).all() for v in arrays.values()),
                synchronized_lengths=len({len(v) for v in arrays.values()})==1,
                hand_contact=contact_frames>5,lift_over_2cm=max_lift>.02,
                final_position_within_3cm=float(np.linalg.norm(final[:2]-goal[:2]))<.03,
                final_settled=stable_speed<.02,
                final_on_table=abs(float(final[2])-float(goal[2]))<.005,
                penetration_under_5mm=max_penetration<.005,
                final_released=not any(c['body'].startswith(hand_prefix) for c in contact_rows[-1]['contacts']))
    if contact_controller:checks['teacher_trajectory_completed']=clock>=end
    report=dict(checks=checks,task_success=all(checks.values()),training_ready=False,
                dataset_status='Pending independent dataset audit' if record else 'Diagnostic state-only rollout; no camera data',
                frames=frame,fps=20,max_lift_m=max_lift,contact_frames=contact_frames,
                maximum_hand_contacts=max_contact,final_case_position_m=final.tolist(),goal_position_m=goal.tolist(),
                maximum_contact_penetration_m=max_penetration,
                final_case_speed_m_s=stable_speed,elapsed_seconds=time.monotonic()-started,
                physics=dict(object_pose_updates_after_initialization=False,welds=False,external_object_forces=False,
                             embodiment=embodiment,timestep_s=.001),
                controller='Privileged video-derived joint-target teacher; not learned or camera-only',
                source_motion=str(motion.resolve()),offset_m=list(offset))
    report['zero_fingers_counterfactual']=zero_fingers
    report['source_rate']=source_rate;report['servo_gravity_velocity_feedforward']=servo_feedforward
    report['policy_derived_reset_prefix_steps']=0 if initial_recovery_actions is None else len(initial_recovery_actions)
    report['reset_simulation_time_s']=episode_start_time
    if contact_controller:
        report['active_tactile_feedback']=dict(enabled=contact_controller.enabled,target_force_n=contact_controller.target,
            events=contact_controller.events,signal='All contacts on each finger; normal/shear force magnitude; no object identity',
            limitation='Idealized simulated force sensing; not calibrated hardware tactile sensors or a proven slip detector')
    (output/'report.json').write_text(json.dumps(report,indent=2))
    schema=dict(format='Local synchronized simulation episode v1; not yet LeRobot parquet',fps=20,
                task='Pick up the black case from the book and place it on the table.',
                state_joint_names=[model.joint(int(j)).name for j in joint_ids],
                action_actuator_names=[model.actuator(int(i)).name for i in active_aids],
                observation_state='Selected G1 right-arm and Inspire joint positions, radians; includes passive mimic joints',
                action='Absolute G1 arm and Inspire motor joint-angle targets, radians; finite native/declared effort limits',
                action_ctrlrange=model.actuator_ctrlrange[active_aids].tolist(),active_actuator_indices=active_aids.tolist(),
                hand_contact_prefix=hand_prefix,
                state_qpos_indices=qadr.tolist(),state_qvel_indices=dadr.tolist(),
                alignment='Image/state/timestamp before action; action held 50 physics steps; next_state after action',
                image_pattern='images/observation.images.front/{frame:06d}.jpg',image_shape=[256,256,3],
                recorded_images=record,additional_image_pattern='images/observation.images.inspection/{frame:06d}.jpg',
                privileged_fields=['object_qpos','source_timestamp'],
                training_ready=report['training_ready'],source_video_sha256=json.loads((motion/'report.json').read_text())['source_sha256'])
    (output/'dataset.json').write_text(json.dumps(schema,indent=2))
    if contact_controller:
        from .tactile import FEATURE_NAMES
        schema.update(tactile_feature_names=FEATURE_NAMES,tactile_units='newtons',
            tactile_alignment='Sampled before action, same timestamp as RGB and joint state',
            optional_tactile_state_dimension=len(joint_ids)+10)
        (output/'dataset.json').write_text(json.dumps(schema,indent=2))
    print(json.dumps(report,indent=2),flush=True)
