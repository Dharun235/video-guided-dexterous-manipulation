"""Supported Unitree G1 + official Unitree Inspire assets for book transfer."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .book_transfer import set_configuration, rollout

ROOT=Path(__file__).resolve().parents[1]
HAND=ROOT/'third_party/unitree-xr-book/assets/inspire_hand'
G1=ROOT/'third_party/menagerie/unitree_g1'
ORIGIN=np.array([.42,.03,.78])
# Source camera looks along +y over the table; map that to G1 forward +x.
# Source image-right (+x) then maps to the robot's right (-y).
SCENE_ROTATION=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
LAYOUT_VERSION='ego_front_v2'
BACKGROUND_CONTROL={'left_shoulder_pitch_joint':.6,'left_shoulder_roll_joint':.15,'left_elbow_joint':.2}
ARM=['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw']
DRIVEN=['R_pinky_proximal_joint','R_ring_proximal_joint','R_middle_proximal_joint',
        'R_index_proximal_joint','R_thumb_proximal_pitch_joint','R_thumb_proximal_yaw_joint']


def convert_hand(output):
    urdf=ET.parse(HAND/'inspire_hand_right.urdf').getroot()
    compiler=urdf.find('mujoco/compiler');compiler.set('meshdir',str(HAND/'meshes'));compiler.set('fusestatic','false')
    mimics=[]
    for link in urdf.findall('link'):
        for visual in list(link.findall('visual')):
            if visual.find('geometry/mesh') is None:link.remove(visual) # Source coordinate axes/tip markers only.
    for mesh in urdf.findall('.//mesh'):mesh.set('filename',str(HAND/'meshes'/Path(mesh.get('filename')).name))
    for joint in urdf.findall('joint'):
        mimic=joint.find('mimic')
        if mimic is not None:mimics.append((joint.get('name'),mimic.get('joint'),float(mimic.get('multiplier','1')),float(mimic.get('offset','0'))))
    (output/'inspire-source.urdf').write_text(ET.tostring(urdf,encoding='unicode'))
    imported=mujoco.MjModel.from_xml_path(str((output/'inspire-source.urdf').resolve()))
    mujoco.mj_saveLastXML(str(output/'inspire-converted.xml'),imported)
    converted=ET.parse(output/'inspire-converted.xml').getroot()
    for mesh in converted.findall('asset/mesh'):
        mesh.set('file',str(HAND/'meshes'/Path(mesh.get('file')).name))
    body=converted.find(".//body[@name='R_hand_base_link']")
    for free in list(body.findall('freejoint')):body.remove(free)
    for free in list(body.findall("joint[@type='free']")):body.remove(free)
    # Reference wrist is the anatomical heel, distal to the mechanical mounting flange.
    ET.SubElement(body,'site',name='landmark_0',pos='0 -.045 0',size='.002',rgba='0 1 0 0')
    for fi,finger in enumerate(('thumb','index','middle','ring','pinky')):
        if finger=='thumb':names=['proximal_base','intermediate','distal','tip']
        else:names=['proximal','intermediate','intermediate','tip']
        for j,name in enumerate(names):
            node=body.find(f".//body[@name='R_{finger}_{name}']")
            ET.SubElement(node,'site',name=f'landmark_{1+4*fi+j}',pos='0 .018 0' if j==2 and finger!='thumb' else '0 0 0',size='.002',rgba='0 1 0 0')
    for geom in body.findall('.//geom'):
        if geom.get('contype','1')!='0':
            geom.set('friction','1.0 .005 .001');geom.set('condim','4');geom.set('group','3');geom.set('rgba','1 1 1 0')
    for joint in body.findall('.//joint'):
        joint.set('damping','.06');joint.set('armature','.0002');joint.set('frictionloss','.005')
    return converted,body,mimics


def make_model(scene,output):
    root=ET.parse(G1/'g1.xml').getroot()
    root.find('compiler').set('autolimits','true');root.find('compiler').attrib.pop('meshdir',None)
    for asset in root.findall('asset/*'):
        if asset.get('file'):
            if asset.tag=='mesh' and not asset.get('name'):asset.set('name',Path(asset.get('file')).stem)
            asset.set('file',str(G1/'assets'/asset.get('file')))
    root.remove(root.find('keyframe'))
    pelvis=root.find(".//body[@name='pelvis']");pelvis.remove(pelvis.find('freejoint'))
    option=root.find('option');option.set('timestep','.001');option.set('cone','elliptic');option.set('impratio','10')
    ET.SubElement(root.find('default'),'geom',friction='.7 .005 .001',solref='.005 1',solimp='.95 .99 .001')
    # Keep pelvis on a visible stand. This model does not claim balance/walking.
    world=root.find('worldbody')
    ET.SubElement(world,'geom',name='support',type='box',pos='-.08 0 .40',size='.05 .08 .38',rgba='.25 .3 .35 1',contype='0',conaffinity='0')
    ET.SubElement(world,'geom',name='room_floor',type='plane',size='2 2 .1',rgba='.25 .28 .30 1',contype='0',conaffinity='0')
    wrist=root.find(".//body[@name='right_wrist_yaw_link']")
    for geom in list(wrist.findall('geom')):
        if 'rubber_hand' in geom.get('mesh',''):wrist.remove(geom)
    converted,body,mimics=convert_hand(output)
    for a in converted.find('asset'):root.find('asset').append(a)
    # Official G1 DFQ URDF R_base_link_joint: xyz .0415 0 0,
    # rpy pi 0 -pi/2. Keep the genuine right-hand mesh and rotation.
    body.set('pos','.0415 0 0');body.set('quat','0 .70710678 -.70710678 0');wrist.append(body)
    equality=ET.SubElement(root,'equality')
    for dependent,driver,mult,offset in mimics:
        ET.SubElement(equality,'joint',joint1=dependent,joint2=driver,polycoef=f'{offset} {mult} 0 0 0',solref='.004 1')
    for name in DRIVEN:
        urdfjoint=ET.parse(HAND/'inspire_hand_right.urdf').find(f"joint[@name='{name}']/limit")
        ET.SubElement(root.find('actuator'),'position',name=name,joint=name,kp='4',kv='.12',
                      ctrlrange=f"{urdfjoint.get('lower')} {urdfjoint.get('upper')}",forcerange='-1 1')
    contact=root.find('contact')
    if contact is None:contact=ET.SubElement(root,'contact')
    ET.SubElement(contact,'exclude',body1='right_wrist_yaw_link',body2='R_hand_base_link')
    # Compose video-fitted environment at a reachable tabletop height.
    source=ET.parse(scene/'scene.xml').getroot()
    for asset in source.find('asset'):
        if asset.get('file'):asset.set('file',str((scene/'assets'/asset.get('file')).resolve()))
        root.find('asset').append(asset)
    from scipy.spatial.transform import Rotation
    sq=Rotation.from_matrix(SCENE_ROTATION).as_quat()[[3,0,1,2]]
    frame=ET.SubElement(world,'body',name='scene_frame',pos=' '.join(map(str,ORIGIN)),quat=' '.join(map(str,sq)))
    for node in source.find('worldbody'):
        if node.tag=='camera':continue
        if node.tag=='body' and node.get('name')=='case':
            node.find("geom[@name='case_solid']").set('condim','4')
            node.set('pos',' '.join(map(str,SCENE_ROTATION@np.fromstring(node.get('pos'),sep=' ')+ORIGIN)))
            from scipy.spatial.transform import Rotation
            q=np.fromstring(node.get('quat'),sep=' ')
            node.set('quat',' '.join(map(str,(Rotation.from_matrix(SCENE_ROTATION)*Rotation.from_quat(q[[1,2,3,0]])).as_quat()[[3,0,1,2]])))
            world.append(node)
        else:frame.append(node)
    from scipy.spatial.transform import Rotation
    meta=json.loads((scene/'reconstruction.json').read_text());cam=meta['camera']
    ego_position=SCENE_ROTATION@np.array(cam['position'])+ORIGIN
    ego_rotation=SCENE_ROTATION@Rotation.from_rotvec(cam['rotation']).as_matrix().T@np.diag([1,-1,-1])
    ego_q=Rotation.from_matrix(ego_rotation).as_quat()[[3,0,1,2]]
    # The torso is the rigid parent of the G1 head mesh. This camera follows its
    # pose; the square image widens vertical FOV to retain the phone's width.
    torso=root.find(".//body[@name='torso_link']")
    torso_origin=np.array([-.0039635,0,.837])
    focal=np.array(cam['intrinsics'])[0,0]
    for name,fovy in [('front_dataset',np.rad2deg(2*np.arctan(832/(2*focal)))),
                       ('source_match',np.rad2deg(2*np.arctan(464/(2*focal))))]:
        ET.SubElement(torso,'camera',name=name,pos=' '.join(map(str,ego_position-torso_origin)),
                      quat=' '.join(map(str,ego_q)),fovy=str(fovy))
    for name,pos,target,fovy in [('inspection_dataset',[.95,-.65,1.3],[.42,-.03,.85],50),
                                  ('robot_overview',[1.6,-1.4,1.4],[.2,0,.8],50)]:
        z=np.array(pos)-target;z/=np.linalg.norm(z);x=np.cross([0,0,1],z);x/=np.linalg.norm(x);y=np.cross(z,x)
        q=Rotation.from_matrix(np.column_stack([x,y,z])).as_quat()[[3,0,1,2]]
        ET.SubElement(world,'camera',name=name,pos=' '.join(map(str,pos)),quat=' '.join(map(str,q)),fovy=str(fovy))
    for node in root.iter():
        for light in list(node.findall('light')):node.remove(light)
    ET.SubElement(world,'light',pos='.3 -.2 1.8',dir='0 0 -1',diffuse='.5 .5 .5')
    visual=ET.SubElement(root,'visual');ET.SubElement(visual,'quality',shadowsize='512',offsamples='0')
    ET.SubElement(visual,'headlight',ambient='.22 .22 .22',diffuse='.12 .12 .12',specular='.05 .05 .05')
    custom=ET.SubElement(root,'custom');ET.SubElement(custom,'text',name='book_layout',data=LAYOUT_VERSION)
    xml=ET.tostring(root,encoding='unicode');(output/'scene.xml').write_text(xml)
    model=mujoco.MjModel.from_xml_string(xml)
    from .layout import verify_layout
    if not verify_layout(model,output)['passed']:
        raise RuntimeError('The G1 must face the centre of the table long edge')
    active=[model.actuator('right_'+n+'_joint').id for n in ARM]+[model.actuator(n).id for n in DRIVEN]
    state=[j for j in range(model.njnt) if model.joint(j).name.startswith(('right_shoulder','right_elbow','right_wrist','R_'))]
    return model,np.array(active),np.array(state)


def retarget(model,active,motion,output,offset):
    h=np.load(motion/'hand_world.npz');times=h['seconds'];human=h['joints']@SCENE_ROTATION.T+ORIGIN+offset
    data=mujoco.MjData(model);lo,hi=model.actuator_ctrlrange[active].T.copy()
    cases=json.loads((motion/'case_poses.json').read_text())
    from scipy.spatial.transform import Rotation
    local_h=h['joints'];grasp_index=int(np.argmin(abs(times-4.97)))
    initial_pos=np.array(cases[0]['position_m']);final_pos=np.array(cases[-1]['position_m'])
    initial_rotation=Rotation.from_quat(cases[0]['quaternion_xyzw']).as_matrix()
    grasp_relative=(local_h[grasp_index]-initial_pos)@initial_rotation
    def planned_object(t):
        def ease(a):a=np.clip(a,0,1);return a*a*(3-2*a)
        if t<=5.05:return initial_pos.copy()
        if t<5.8:return initial_pos+np.array([0,0,.07*ease((t-5.05)/.75)])
        if t<8.6:
            q=ease((t-5.8)/2.8);p=initial_pos*(1-q)+final_pos*q;p[2]=initial_pos[2]+.07;return p
        q=ease((t-8.6)/1.0);high=final_pos.copy();high[2]=initial_pos[2]+.07
        return high*(1-q)+final_pos*q
    hi[1]=min(hi[1],.10)
    lo[2]=max(lo[2],-1.2);hi[2]=min(hi[2],1.2)
    seed=np.array([-.5,-.2,0,1.,0,0,0, .3,.3,.3,.3,.2,.5])
    values=[];errors=[]
    for t,target in zip(times,human):
        if not np.isfinite(target).all() or t<3.:values.append(None);errors.append(None);continue
        # Embodiment adaptation: coupled Inspire fingers need object-surface
        # targets, since independent human distal bends are not realizable.
        from scipy.spatial.transform import Rotation
        case=min(cases,key=lambda c:abs(c['seconds']-t))
        planned_pos=planned_object(t)
        if 4.9<=t<=9.65:
            target=(grasp_relative@initial_rotation.T+planned_pos)@SCENE_ROTATION.T+ORIGIN+offset
        elif t>9.65:
            # Use the recorded opening/retreat relative to its object, anchored
            # at the robot placement instead of reintroducing hidden depth drift.
            source_rotation=Rotation.from_quat(case['quaternion_xyzw']).as_matrix()
            relative=(local_h[np.argmin(abs(times-t))]-case['position_m'])@source_rotation
            target=(relative@initial_rotation.T+planned_pos)@SCENE_ROTATION.T+ORIGIN+offset
        blend=min(np.clip((t-4.35)/.55,0,1),np.clip((10.05-t)/.4,0,1))
        desired=target.copy()
        # URDF tip markers are distal to the actual collision pads. Offset the
        # targets so opposing pads meet the case sides near its mid-height.
        points=np.array([[-.005,-.026,.002],[-.034,.004,-.006],[-.006,.026,-.006],[.020,.020,.002],[.032,.020,.002]])
        object_points=(points@initial_rotation.T+planned_pos)@SCENE_ROTATION.T+ORIGIN+offset
        for j,tip in enumerate((4,8,12)):desired[tip]=(1-blend)*target[tip]+blend*object_points[j]
        def objective(u):
            full=np.zeros(model.nu);full[active]=u;set_configuration(model,data,full)
            predicted=np.array([data.site_xpos[model.site(f'landmark_{i}').id] for i in range(21)])
            weights=np.ones(21)*.35;weights[[4,8,12]]=2+8*blend;weights[[16,20]]=1.;weights[0]=.5
            forward=target[9]-target[0];across=target[5]-target[17]
            dorsum=np.cross(forward,across);dorsum/=max(np.linalg.norm(dorsum),1e-9)
            actual_dorsum=data.xmat[model.body('R_hand_base_link').id].reshape(3,3)[:,0]
            actual_forward=-data.xmat[model.body('R_hand_base_link').id].reshape(3,3)[:,1]
            forward/=max(np.linalg.norm(forward),1e-9)
            return np.r_[((predicted-desired)*weights[:,None]).ravel(),.02*(actual_dorsum-dorsum),.02*(actual_forward-forward),.002*(u-seed)]
        fit=least_squares(objective,np.clip(seed,lo+1e-6,hi-1e-6),bounds=(lo+1e-7,hi-1e-7),max_nfev=100,ftol=1e-5)
        seed=fit.x;values.append(seed.tolist());errors.append(float(np.mean(np.abs(objective(seed)[:63]))))
    ids=[i for i,x in enumerate(values) if x is not None];rows=np.array([values[i] for i in ids])
    actions=np.array([np.interp(times,times[ids],rows[:,i]) for i in range(len(active))]).T
    closure=np.minimum(np.clip((times-4.5)/.4,0,1),np.clip((9.85-times)/.3,0,1))
    actions[:,7:11]+=closure[:,None]*.35;actions[:,11]+=closure*.13
    actions=np.clip(actions,*model.actuator_ctrlrange[active].T)
    # Release only after lowering: hold the wrist still while opening, then
    # retreat. Copying the occluded human retreat directly flicked the case.
    release=actions[np.argmin(abs(times-9.6))].copy()
    full=np.zeros(model.nu);full[active]=release;set_configuration(model,data,full)
    sid=model.site('landmark_0').id;release_position=data.site_xpos[sid].copy();release_rotation=data.site_xmat[sid].reshape(3,3).copy()
    from .kinematics import solve_arm
    opened=release[7:].copy();opened[:5]=0.
    for i,t in enumerate(times):
        if t<9.6:continue
        q=np.clip((t-9.8)/1.0,0,1);q=q*q*(3-2*q)
        actions[i]=release;actions[i,7:]=(1-q)*release[7:]+q*opened
        if t>10.3:
            fraction=np.clip((t-10.3)/1.1,0,1);fraction=fraction*fraction*(3-2*fraction)
            full=np.zeros(model.nu);full[active]=actions[i];set_configuration(model,data,full)
            arm,error=solve_arm(model,data,'right',release_position+[0,0,.10*fraction],release[:7],site_name='landmark_0')
            actions[i,:7]=arm
    np.savez_compressed(output/'retarget.npz',seconds=times,actions=actions,observed=np.isfinite(human).all(axis=(1,2)))
    (output/'retarget.json').write_text(json.dumps(dict(source=str(motion.resolve()),offset_m=np.asarray(offset).tolist(),layout_version=LAYOUT_VERSION,
          source_sha256=json.loads((motion/'report.json').read_text())['source_sha256'],
          mean_weighted_coordinate_objective_m=float(np.mean([x for x in errors if x is not None])),per_sample_weighted_coordinate_objective_m=errors,
          fit_metric_note='Weighted pre-closure IK objective; not measured true hand pose accuracy or executed grasp error',
          method='G1 right-arm + six Inspire motor landmark IK with native mimic couplings',
          grasp_closure_bias_rad={'four_fingers':.35,'thumb_pitch':.13},
          placement_wrist_compensation_m=[0,0,0],
          movement_adaptation='Recorded grasp and destination; lift 7 cm before carrying, retain grasp orientation, lower, hold wrist while opening, then retreat. Not exact video pose playback.'),indent=2))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('output',type=Path);p.add_argument('--no-record',action='store_true')
    p.add_argument('--offset',type=float,nargs=3,default=(0,0,0));p.add_argument('--retarget-from',type=Path)
    p.add_argument('--zero-fingers',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    motion=Path('runs/book5-motion-06');model,active,state=make_model(Path('runs/book5-space-03'),a.output)
    if a.retarget_from:
        import shutil
        if json.loads((a.retarget_from/'retarget.json').read_text()).get('layout_version')!=LAYOUT_VERSION:
            raise SystemExit('Reference uses the legacy side-view layout; recompute retargeting for the ego/front layout')
        for name in ('retarget.npz','retarget.json'):shutil.copy2(a.retarget_from/name,a.output/name)
    else:retarget(model,active,motion,a.output,np.array(a.offset))
    rollout(model,motion,a.output,record=not a.no_record,active_aids=active,state_joint_ids=state,
            world_offset=ORIGIN,world_rotation=SCENE_ROTATION,hand_prefix='R_',embodiment='Unitree G1 with official Unitree Inspire five-finger right hand; fixed pelvis',zero_finger_start=7,
            source_rate=.75,servo_feedforward=True,zero_fingers=a.zero_fingers,background_control=BACKGROUND_CONTROL)


if __name__=='__main__':main()
