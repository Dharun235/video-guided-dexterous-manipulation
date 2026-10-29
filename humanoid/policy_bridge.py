"""MuJoCo JSON bridge for actuator-driven reference execution.

Run in .venv-g1. A separate policy process supplies motor targets. Object poses
and contacts are recorded for scoring and never returned as policy observations.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import mujoco
import numpy as np

from .book_transfer import front_sensor_image
from .book_dataset_audit import raised_hold_duration
from .layout import verify_layout


def serve(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    model=mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    layout=verify_layout(model,out)
    if not layout['passed']:raise ValueError('Robot/table layout check failed')
    data=mujoco.MjData(model)
    mujoco.mj_setState(model,data,np.load(args.initial)['integration'],mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model,data)
    schema=json.loads(args.schema.read_text());goal=np.array(json.loads(args.goal))
    aids=np.array(schema['active_actuator_indices']);qadr=np.array(schema['state_qpos_indices'])
    dadr=np.array(schema['state_qvel_indices'])
    velocity_map=[schema['state_joint_names'].index(n) for n in schema['action_actuator_names']]
    cq=model.jnt_qposadr[model.joint('case_free').id]
    cv=model.jnt_dofadr[model.joint('case_free').id];cb=model.body('case').id
    initial_case=data.qpos[cq:cq+3].copy();renderer=mujoco.Renderer(model,height=256,width=256)
    writer=cv2.VideoWriter(str(out/'execution-raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20/args.video_stride,(768,256))
    states=[];actions=[];objects=[];next_states=[];contact_rows=[];base_actions=[];corrections=[];velocities=[]
    max_lift=0.;max_penetration=0.;clipped=0

    def contacts():
        rows=[]
        for i in range(data.ncon):
            c=data.contact[i];b=[model.geom_bodyid[c.geom1],model.geom_bodyid[c.geom2]]
            if cb not in b:continue
            other=b[1] if b[0]==cb else b[0];force=np.zeros(6);mujoco.mj_contactForce(model,data,i,force)
            rows.append(dict(body=model.body(other).name or 'world',normal_force_n=float(force[0]),distance_m=float(c.dist)))
        return rows

    def images():
        renderer.update_scene(data,camera='front_dataset');front=front_sensor_image(model,renderer.render().copy())
        renderer.update_scene(data,camera='inspection_dataset');inspection=renderer.render().copy()
        return front,inspection

    def observation():
        from .tactile import sense
        front,inspection=images()
        np.savez(out/'observation.npz',state=data.qpos[qadr].copy(),tactile=sense(model,data),front=front,inspection=inspection)
        return dict(observation_npz=str(out/'observation.npz'),steps=len(actions))

    print(json.dumps(observation()),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request['command']=='finish':break
        if request['command']!='step':raise ValueError('Unknown command')
        chunk=np.asarray(request['actions'],dtype=float)
        if chunk.ndim!=2 or chunk.shape[1]!=len(aids) or not np.isfinite(chunk).all():raise ValueError('Invalid actions')
        for action in chunk:
            if len(actions)>=args.maximum_steps:raise ValueError('Rollout step limit exceeded')
            if len(actions)%args.video_stride==0:
                front,inspection=images()
                renderer.update_scene(data,camera='robot_overview');overview=renderer.render().copy()
                panel=cv2.cvtColor(np.hstack([front,inspection,overview]),cv2.COLOR_RGB2BGR)
                cv2.putText(panel,f'{args.controller_label} | t={len(actions)/20:.2f}s',(5,18),0,.42,(255,255,255),1)
                writer.write(panel)
            states.append(data.qpos[qadr].copy());objects.append(data.qpos[cq:cq+7].copy())
            contact_rows.append(dict(frame=len(actions),seconds=float(data.time),contacts=contacts()))
            base_actions.append(action.copy());velocity=data.qvel[dadr][velocity_map].copy();velocities.append(velocity)
            bounded=np.clip(action,*model.actuator_ctrlrange[aids].T)
            corrections.append(np.zeros(7))
            clipped+=int(np.count_nonzero(bounded!=action));actions.append(bounded.copy());data.ctrl[aids]=bounded
            for _ in range(50):
                mujoco.mj_step(model,data)
                for c in data.contact:max_penetration=max(max_penetration,float(-c.dist))
                max_lift=max(max_lift,float(data.qpos[cq+2]-initial_case[2]))
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():raise RuntimeError('Nonfinite dynamics')
            next_states.append(data.qpos[qadr].copy())
        print(json.dumps(observation()),flush=True)
    writer.release();renderer.close()
    if not actions:raise ValueError('Empty policy rollout')
    final=data.qpos[cq:cq+3].copy();speed=float(np.linalg.norm(data.qvel[cv:cv+3]))
    hold=raised_hold_duration(np.array(objects),contact_rows,20)
    contact_frames=sum(any(c['body'].startswith('R_') and c['normal_force_n']>.01 for c in r['contacts']) for r in contact_rows)
    checks=dict(finite_arrays=all(np.isfinite(a).all() for a in (states,actions,objects,next_states)),
        hand_contact=contact_frames>5,lift_over_2cm=max_lift>.02,
        final_position_within_3cm=float(np.linalg.norm(final[:2]-goal[:2]))<.03,
        final_settled=speed<.02,final_on_table=abs(float(final[2]-goal[2]))<.005,
        penetration_under_5mm=max_penetration<.005,
        final_released=not any(c['body'].startswith('R_') for c in contacts()),
        sustained_raised_multifinger_hold=hold>=.5)
    np.savez_compressed(out/'episode.npz',observation_state=states,action=actions,object_qpos=objects,
        next_observation_state=next_states,timestamp=np.arange(len(actions))/20,
        base_action=base_actions,arm_correction=corrections,motor_velocity=velocities)
    (out/'contacts.json').write_text(json.dumps(contact_rows))
    report=dict(checks=checks,task_success=all(checks.values()),frames=len(actions),fps=20,
        final_position_error_m=float(np.linalg.norm(final[:2]-goal[:2])),final_case_position_m=final.tolist(),
        goal_position_m=goal.tolist(),max_lift_m=max_lift,final_speed_m_s=speed,
        maximum_penetration_m=max_penetration,raised_multifinger_hold_s=hold,
        clipped_action_components=clipped,teacher_enabled=False,controller_label=args.controller_label,
        learned_policy=False,diagnostic_video_fps=20/args.video_stride,
        policy_observations=['RGB front','RGB inspection','19 robot joint positions','task instruction'],
        physics=dict(welds=False,object_pose_updates_after_initialization=False,external_object_forces=False),
        source_scene=str(args.scene.resolve()))
    if args.tactile_inputs:report['policy_observations']+=['10 external fingertip normal/shear force features']
    (out/'report.json').write_text(json.dumps(report,indent=2))
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(out/'execution-raw.mp4'),'-c:v','libx264',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(out/'execution.mp4')],check=True)
    print(json.dumps(dict(finished=True,report=str(out/'report.json'),task_success=report['task_success'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('scene','initial','schema','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--goal',required=True);p.add_argument('--maximum-steps',type=int,default=400)
    p.add_argument('--controller-label',default='Reference controller')
    p.add_argument('--video-stride',type=int,default=1)
    p.add_argument('--tactile-inputs',action='store_true')
    a=p.parse_args()
    if a.video_stride<1:raise SystemExit('Video stride must be positive')
    serve(a)
