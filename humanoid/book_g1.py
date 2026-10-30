"""Supported Unitree G1 + official Unitree Inspire assets for book transfer."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


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


def make_model(scene,output,calibration=None):
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
            q=np.fromstring(node.get('quat','1 0 0 0'),sep=' ')
            node.set('quat',' '.join(map(str,(Rotation.from_matrix(SCENE_ROTATION)*Rotation.from_quat(q[[1,2,3,0]])).as_quat()[[3,0,1,2]])))
            world.append(node)
        else:frame.append(node)
    from scipy.spatial.transform import Rotation
    meta=json.loads(((calibration or scene)/'reconstruction.json').read_text());cam=meta['camera']
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
