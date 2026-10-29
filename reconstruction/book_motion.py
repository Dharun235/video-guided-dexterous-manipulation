"""Recover a constrained hand/case replay from book5 on CPU.

Poses are estimates, conditioned on the existing scene scale and reviewed task
intervals. Replay prescribes poses; it is not a force-driven robot grasp.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial.transform import Rotation, Slerp
from scipy.signal import savgol_filter

from .book_space import BOOK_QUAD, BOOK_SIZE, SOURCE_SHA256, case_vertices, fit_camera, project


def palm_frame(points):
    x = points[5] - points[17]
    x /= np.linalg.norm(x)
    y = points[9] - points[0]
    y -= x * np.dot(x, y)
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def support_height(points):
    inside = (np.abs(points[:, 0]) < BOOK_SIZE[0]/2) & (np.abs(points[:, 1]) < BOOK_SIZE[1]/2)
    return np.where(inside, BOOK_SIZE[2], 0.)


def proxy_distance(points,position,rotation,size):
    """Signed distance to the oriented box proxy; not actual contact sensing."""
    q=np.abs((points-position)@rotation)-size/2
    return np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(np.max(q,axis=1),0)


def register_camera(image, reference, k, features):
    sift, kp0, des0 = features
    kp, des = sift.detectAndCompute(image, None)
    if des is None:
        return None
    matches = cv2.BFMatcher().knnMatch(des0, des, k=2)
    good = [m for pair in matches if len(pair)==2 for m,n in [pair] if m.distance < .7*n.distance]
    if len(good) < 10:
        return None
    a = np.float32([kp0[m.queryIdx].pt for m in good])
    b = np.float32([kp[m.trainIdx].pt for m in good])
    hom, inliers = cv2.findHomography(a, b, cv2.RANSAC, 2.5)
    if hom is None or inliers.sum() < 10:
        return None
    quad = cv2.perspectiveTransform(BOOK_QUAD[None], hom)[0]
    _, r, t, error = fit_camera(quad, k[0, 0])
    mapped=cv2.perspectiveTransform(a[None],hom)[0]
    residual=np.linalg.norm(mapped-b,axis=1)[inliers.ravel()>0]
    return dict(rotation=r.ravel().tolist(), translation=t.ravel().tolist(),
                inliers=int(inliers.sum()), median_feature_fit_px=float(np.median(residual)),
                corner_fit_px=float(error.mean()), book_quad=quad.tolist())


def align_hand(hand, k, camera):
    local = np.asarray(hand['keypoints_3d'])
    vertices = np.asarray(hand['vertices'])
    uv = np.asarray(hand['keypoints_2d'])
    rc = cv2.Rodrigues(np.array(camera['rotation']))[0]
    tc = np.array(camera['translation'])
    _, r, t = cv2.solvePnP(local, uv, k, None)
    initial = np.r_[r.ravel(),t.ravel()]
    def residual(p):
        rot = cv2.Rodrigues(p[:3])[0]
        world = (vertices[::8]@rot.T + p[3:] - tc)@rc
        reproj=(project(local,k,p[:3],p[3:])-uv).ravel()/3.
        # Soft scene constraint, not a measured contact label.
        penetration=np.maximum(support_height(world)-world[:,2]-.002,0)*500
        return np.r_[reproj,penetration,(p-initial)*.3]
    fit=least_squares(residual,initial,loss='soft_l1',max_nfev=45)
    rot=cv2.Rodrigues(fit.x[:3])[0]
    world_kp=(local@rot.T+fit.x[3:]-tc)@rc
    world_vertices=(vertices@rot.T+fit.x[3:]-tc)@rc
    error=np.linalg.norm(project(local,k,fit.x[:3],fit.x[3:])-uv,axis=1)
    return world_kp,world_vertices,dict(mean_reprojection_px=float(error.mean()),
        max_scene_penetration_m=float(np.maximum(support_height(world_vertices)-world_vertices[:,2],0).max()),
        note='Fits predicted 2D joints; errors are not against manually labelled ground truth')


def segment_case(image, center):
    x,y=np.round(center).astype(int)
    crop=np.zeros(image.shape[:2],np.uint8)
    crop[max(150,y-85):min(464,y+85),max(0,x-85):min(832,x+85)]=255
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask=((hsv[:,:,2]<105)&(hsv[:,:,1]<110)&(crop>0)).astype(np.uint8)*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for contour in contours:
        area=cv2.contourArea(contour)
        if not 350<area<15000:
            continue
        m=cv2.moments(contour)
        c=np.array([m['m10']/max(m['m00'],1),m['m01']/max(m['m00'],1)])
        solidity=area/max(cv2.contourArea(cv2.convexHull(contour)),1)
        candidates.append((np.linalg.norm(c-center)+40*(1-solidity),contour,solidity))
    if not candidates:
        return None,None,0.
    _,contour,solidity=min(candidates,key=lambda x:x[0])
    mask[:]=0;cv2.drawContours(mask,[contour],-1,255,-1)
    # Red chroma distinguishes visible skin from the neutral dark case.
    b,g,r=cv2.split(image.astype(float))
    skin=((r>b*1.14)&(r>g*1.03)&(hsv[:,:,1]>35)).astype(np.uint8)
    skin=cv2.dilate(skin,np.ones((7,7),np.uint8))
    return contour[:,0,:].astype(float),skin,float(solidity)


def fit_case(image, prior_pos, prior_rot, k, camera, size, supported=None):
    rc=np.array(camera['rotation']);tc=np.array(camera['translation'])
    center=project([prior_pos],k,rc,tc)[0]
    contour,skin,solidity=segment_case(image,center)
    local=case_vertices(size)
    if contour is None or solidity<.65:
        return prior_pos,prior_rot,dict(mode='hand_relative_prior' if supported is None else 'support_prior',
                                       visible_silhouette=False,solidity=solidity)
    angles=np.linspace(0,2*np.pi,48,endpoint=False)
    directions=np.c_[np.cos(angles),np.sin(angles)]
    target=(contour@directions.T).max(0)
    def unpack(p):
        if supported is not None:
            pos=np.r_[p[:2],supported]
            rot=Rotation.from_euler('z',p[2]).as_matrix()
        else:
            pos=p[:3]
            rot=Rotation.from_rotvec(p[3:]).as_matrix()@prior_rot
        return pos,rot
    def residual(p):
        pos,rot=unpack(p)
        xyz=local@rot.T+pos
        uv=project(xyz,k,rc,tc)
        projections=uv@directions.T
        indices=projections.argmax(0)
        values=projections.max(0)
        residual=values-target
        extreme=np.clip(np.round(uv[indices]).astype(int),[0,0],[831,463])
        obscured=skin[extreme[:,1],extreme[:,0]]>0
        # A visible partial mask cannot constrain edges hidden behind the hand.
        residual[(residual>0)&obscured]=0
        if supported is not None:
            prior=np.r_[(pos[:2]-prior_pos[:2])/.04, p[2]/3.]
        else:
            prior=np.r_[(pos-prior_pos)/.018,p[3:]/.45]
        floor=np.maximum(support_height(xyz)-xyz[:,2],0)*700
        return np.r_[residual/3.,prior,floor]
    if supported is not None:
        yaw=Rotation.from_matrix(prior_rot).as_euler('xyz')[2]
        x0=np.r_[prior_pos[:2],yaw]
        bounds=([prior_pos[0]-.05,prior_pos[1]-.05,yaw-1.0],
                [prior_pos[0]+.05,prior_pos[1]+.05,yaw+1.0])
    else:
        x0=np.r_[prior_pos,np.zeros(3)]
        bounds=(np.r_[prior_pos-.05,[-.9]*3],np.r_[prior_pos+.05,[.9]*3])
    fit=least_squares(residual,x0,bounds=bounds,loss='soft_l1',max_nfev=55)
    pos,rot=unpack(fit.x)
    return pos,rot,dict(mode='silhouette_and_hand_prior' if supported is None else 'silhouette_and_support',
        visible_silhouette=True,solidity=solidity,
        silhouette_support_rmse_px=float(np.sqrt(np.mean(residual(fit.x)[:48]**2)))*3,
        depth_source='hand-relative prior' if supported is None else 'estimated supporting plane')


def recover(scene, hand_run, output):
    started=time.monotonic()
    meta=json.loads((scene/'reconstruction.json').read_text())
    manifest=json.loads((hand_run/'manifest.json').read_text())
    if meta['source_sha256']!=SOURCE_SHA256 or manifest['source_sha256']!=SOURCE_SHA256:
        raise ValueError('Reviewed task intervals only apply to book5')
    output.mkdir(parents=True,exist_ok=False)
    rows=json.loads((hand_run/'hands.json').read_text())
    k=np.array(meta['camera']['intrinsics']);size=np.array(meta['priors']['case_dimensions_m'])
    initial_pos=np.array(meta['initial_case_center_m'])
    initial_rot=Rotation.from_euler('z',meta['case_fit']['yaw_radians']).as_matrix()
    reference=cv2.imread(str(hand_run/'frames/000000.jpg'))
    mask=np.zeros(reference.shape[:2],np.uint8);cv2.fillConvexPoly(mask,BOOK_QUAD.astype(int),255)
    mask[220:325,290:425]=0
    sift=cv2.SIFT_create(nfeatures=2200);kp0,des0=sift.detectAndCompute(reference,mask)
    features=(sift,kp0,des0)
    cameras=[];hand_kp=[];hand_verts=[];hand_reports=[];images=[]
    for i,row in enumerate(rows):
        image=cv2.imread(str(hand_run/'frames'/f"{row['source_frame']:06d}.jpg"));images.append(image)
        cam=register_camera(image,reference,k,features)
        if cam is None:
            raise RuntimeError(f"Unregistered camera at {row['seconds']:.3f}s; refusing silent interpolation")
        cameras.append(dict(seconds=row['seconds'],source_frame=row['source_frame'],**cam))
        if row['hands']:
            kp,v,report=align_hand(row['hands'][0],k,cam)
            hand_kp.append(kp);hand_verts.append(v);hand_reports.append(report)
        else:
            hand_kp.append(None);hand_verts.append(None);hand_reports.append(None)
        if i%20==0:print(f'Aligned camera/hand {i}/{len(rows)}',flush=True)
    # Human review: case stays on book until ~5.05s, carried to ~9.8s, released ~10.2s.
    # The reference interval is not an instrumented contact measurement.
    ref_index=min((i for i,h in enumerate(hand_kp) if h is not None),
                  key=lambda i:abs(rows[i]['seconds']-4.77))
    ref_palm=palm_frame(hand_kp[ref_index])
    ref_anchor=hand_kp[ref_index][[4,8,12]].mean(0)
    relative_rotation=ref_palm.T@initial_rot
    relative_position=ref_palm.T@(initial_pos-ref_anchor)
    positions=[];rotations=[];case_reports=[]
    for i,(row,cam,image) in enumerate(zip(rows,cameras,images)):
        sec=row['seconds'];hand=hand_kp[i]
        if sec<=5.05:
            pos,rot=initial_pos.copy(),initial_rot.copy()
            report=dict(mode='reviewed_stationary_on_book',visible_silhouette=sec<3.6,
                        depth_source='estimated book support')
        elif sec>=10.0:
            if positions:
                prior=positions[-1].copy();prior[2]=size[2]/2
                rot=rotations[-1]
            else:prior=initial_pos.copy();rot=initial_rot.copy()
            pos,rot,report=fit_case(image,prior,rot,k,cam,size,supported=size[2]/2)
        elif hand is not None:
            palm=palm_frame(hand)
            prior=hand[[4,8,12]].mean(0)+palm@relative_position
            rot=palm@relative_rotation
            if sec<5.45:
                blend=np.clip((sec-5.05)/.4,0,1)
                prior=(1-blend)*initial_pos+blend*prior
            pos,rot,report=fit_case(image,prior,rot,k,cam,size)
        else:
            raise RuntimeError(f'Missing hand in carrying interval {sec}; manual review required')
        positions.append(pos);rotations.append(rot);case_reports.append(report)
        if i%20==0:print(f'Case pose {i}/{len(rows)}',flush=True)
    positions=np.array(positions)
    # Smooth only moving positions. Stationary book interval remains fixed.
    filtered=savgol_filter(positions,7,2,axis=0)
    for i,row in enumerate(rows):
        if 5.15<row['seconds']<9.9:positions[i]=filtered[i]
    # The final exposed case establishes the stationary placement pose.
    # Propagate it back across the reviewed release/withdraw interval.
    final_pos=np.median(positions[-5:],axis=0)
    final_rot=Rotation.from_matrix(np.array(rotations[-5:])).mean().as_matrix()
    for i,row in enumerate(rows):
        if row['seconds']>=10.:
            positions[i]=final_pos
            rotations[i]=final_rot
            case_reports[i]['mode']='reviewed_stationary_on_table_from_final_visible_fit'
    raw_rotations=np.array(rotations)
    for i,row in enumerate(rows):
        if 5.05<row['seconds']<10.:
            ids=np.arange(max(0,i-2),min(len(rows),i+3))
            weights=3-np.abs(ids-i)
            rotations[i]=Rotation.from_matrix(raw_rotations[ids]).mean(weights=weights).as_matrix()
            if row['seconds']>9.55:
                blend=min(1.,(row['seconds']-9.55)/.45)
                rotations[i]=Slerp([0.,1.],Rotation.from_matrix([rotations[i],final_rot]))([blend]).as_matrix()[0]
                positions[i]=(1-blend)*positions[i]+blend*final_pos
    support_corrections=[]
    for i,(pos,rot) in enumerate(zip(positions,rotations)):
        xyz=case_vertices(size)@rot.T+pos
        shift=max(0.,float((support_height(xyz)-xyz[:,2]).max()))
        positions[i,2]+=shift
        case_reports[i]['support_projection_m']=shift
        support_corrections.append(shift)
    # Monocular hand depth/size is ambiguous. A small, documented scale about the
    # camera center preserves its image projection while fitting contact priors.
    scales=np.ones(len(rows));contact_before=[]
    for i,row in enumerate(rows):
        if not 5.05<row['seconds']<10.0 or hand_kp[i] is None:continue
        cam=cameras[i];rc=cv2.Rodrigues(np.array(cam['rotation']))[0]
        center=-rc.T@np.array(cam['translation'])
        tips=hand_kp[i][[4,8,12]]
        contact_before.append(float(np.mean(np.abs(proxy_distance(tips,positions[i],rotations[i],size)))))
        def contact_cost(scale):
            p=center+scale*(tips-center)
            distances=proxy_distance(p,positions[i],rotations[i],size)
            v=center+scale*(hand_verts[i][::8]-center)
            floor=np.maximum(support_height(v)-v[:,2]-.002,0)
            return float(np.mean((distances/.009)**2)+np.mean((floor/.003)**2)+((scale-1)/.15)**2)
        scales[i]=minimize_scalar(contact_cost,bounds=(.80,1.12),method='bounded').x
    scales=savgol_filter(scales,7,2)
    contact_after=[]
    for i,row in enumerate(rows):
        if hand_kp[i] is None:continue
        rc=cv2.Rodrigues(np.array(cameras[i]['rotation']))[0]
        center=-rc.T@np.array(cameras[i]['translation'])
        # Project the remaining hand-depth ambiguity out of the static supports.
        # Scaling all camera-relative coordinates keeps its 2D projection fixed.
        original_scale=float(scales[i])
        original_vertices=hand_verts[i]
        for _ in range(3):
            v=center+scales[i]*(original_vertices-center)
            denominator=center[2]-original_vertices[:,2]
            valid=denominator>1e-6
            bounds=(center[2]-support_height(v)+.001)/np.maximum(denominator,1e-6)
            scales[i]=min(scales[i],max(.8,float(bounds[valid].min())))
        hand_kp[i]=center+scales[i]*(hand_kp[i]-center)
        hand_verts[i]=center+scales[i]*(hand_verts[i]-center)
        hand_reports[i]['camera_ray_scale']=float(scales[i])
        hand_reports[i]['additional_support_scale_projection']=original_scale-float(scales[i])
        hand_reports[i]['depth_scale_source']='regularized case-contact prior; not a measured hand size'
        if 5.05<row['seconds']<10.:
            distances=proxy_distance(hand_kp[i][[4,8,12]],positions[i],rotations[i],size)
            hand_reports[i]['fingertip_proxy_signed_distances_m']=distances.tolist()
            contact_after.append(float(np.mean(np.abs(distances))))
    case_rows=[]
    for row,pos,rot,report in zip(rows,positions,rotations,case_reports):
        case_rows.append(dict(seconds=row['seconds'],source_frame=row['source_frame'],position_m=pos.tolist(),
            quaternion_xyzw=Rotation.from_matrix(rot).as_quat().tolist(),**report))
    n=len(rows);vertices=np.full((n,778,3),np.nan,np.float32);joints=np.full((n,21,3),np.nan,np.float32)
    for i,(v,j) in enumerate(zip(hand_verts,hand_kp)):
        if v is not None:vertices[i]=v;joints[i]=j
    np.savez_compressed(output/'hand_world.npz',vertices=vertices,joints=joints,
                        seconds=np.array([r['seconds'] for r in rows]),
                        faces=np.load(hand_run/'hand_faces.npy'))
    (output/'camera_poses.json').write_text(json.dumps(cameras,indent=2))
    (output/'case_poses.json').write_text(json.dumps(case_rows,indent=2))
    (output/'hand_alignment.json').write_text(json.dumps(hand_reports,indent=2))
    errors=[r['mean_reprojection_px'] for r in hand_reports if r]
    report=dict(source_sha256=SOURCE_SHA256,scene=str(scene.resolve()),hand_run=str(hand_run.resolve()),
        samples=n,cameras_registered=len(cameras),aligned_hand_frames=len(errors),
        median_predicted_joint_fit_px=float(np.median(errors)),max_predicted_joint_fit_px=float(max(errors)),
        elapsed_seconds=time.monotonic()-started,
        reviewed_intervals=dict(stationary_on_book_until_s=5.05,carry_end_s=9.8,on_table_from_s=10.),
        object_pose_modes={name:sum(r['mode']==name for r in case_reports) for name in set(r['mode'] for r in case_reports)},
        support_constraints=dict(maximum_case_height_correction_m=float(max(support_corrections)),
                                 note='Explicit geometric nonpenetration prior; not recovered depth'),
        rotation_regularization='Five-sample weighted SO(3) mean and reviewed placement transition',
        hand_depth_alignment=dict(min_camera_ray_scale=float(scales.min()),max_camera_ray_scale=float(scales.max()),
            median_fingertip_proxy_error_before_m=float(np.median(contact_before)),
            median_fingertip_proxy_error_after_m=float(np.median(contact_after)),
            note='Fit to contact prior, not independent grasp evidence'),
        limitations=['Single-view reconstruction is ambiguous; poses are estimates, not ground truth',
                     'Scale depends on estimated paperback/case dimensions',
                     'Hand reprojection compares to model-predicted joints, not independent image labels',
                     'Case depth and hidden rotation use a hand-relative grasp prior in the carrying interval',
                     'Hand camera-ray scale varies within a regularized range to resolve monocular depth/size ambiguity',
                     'Silhouette is partial under occlusion; symmetric-case orientation remains ambiguous',
                     'Reviewed support intervals are manually supplied and are not sensor contact labels',
                     'Replay prescribes hand/object poses; it is not a physical grasp or learned robot policy'])
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


def make_replay_scene(scene,output,first_vertices,faces):
    import trimesh
    assets=output/'assets';shutil.copytree(scene/'assets',assets,dirs_exist_ok=True)
    # Mesh stays visual only: inferred geometry cannot establish contact physics.
    wrist=first_vertices.mean(0)
    trimesh.Trimesh(vertices=first_vertices-wrist,faces=faces,process=False).export(assets/'human_hand.obj')
    root=ET.parse(scene/'scene.xml').getroot()
    ET.SubElement(root.find('asset'),'mesh',name='human_hand',file='human_hand.obj',inertia='shell',smoothnormal='true')
    body=ET.SubElement(root.find('worldbody'),'body',name='human_hand',mocap='true',pos=' '.join(map(str,wrist)))
    ET.SubElement(body,'geom',name='human_hand',type='mesh',mesh='human_hand',
                  rgba='.25 .60 .78 1',contype='0',conaffinity='0',group='1')
    ET.indent(root);ET.ElementTree(root).write(output/'replay.xml',encoding='unicode')


def render(output):
    import mujoco
    report=json.loads((output/'report.json').read_text());scene=Path(report['scene']);hand_run=Path(report['hand_run'])
    data_hand=np.load(output/'hand_world.npz');verts=data_hand['vertices'];faces=data_hand['faces']
    cameras=json.loads((output/'camera_poses.json').read_text());cases=json.loads((output/'case_poses.json').read_text())
    first=next(v for v in verts if np.isfinite(v).all())
    make_replay_scene(scene,output,first,faces)
    model=mujoco.MjModel.from_xml_path(str((output/'replay.xml').resolve()))
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'human_hand')
    mid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_MESH,'human_hand')
    body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'human_hand')
    mocap_id=model.body_mocapid[body]
    mesh_offset=model.mesh_vertadr[mid];mesh_count=model.mesh_vertnum[mid]
    # Undo the compiler's mesh normalization before updating the visual vertices.
    geom_pos=model.geom_pos[gid].copy()
    geom_rot=Rotation.from_quat(model.geom_quat[gid][[1,2,3,0]]).as_matrix()
    renderer=mujoco.Renderer(model,height=464,width=832)
    cam=mujoco.MjvCamera();cam.lookat[:]=[.08,-.02,.04];cam.distance=.65;cam.azimuth=130;cam.elevation=-40
    writer=cv2.VideoWriter(str(output/'replay-raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),
                           10,(2496,464))
    (output/'comparisons').mkdir(exist_ok=True)
    metrics=[]
    for i,(camera,case,v) in enumerate(zip(cameras,cases,verts)):
        rc=Rotation.from_rotvec(camera['rotation']).as_matrix();tc=np.array(camera['translation'])
        ctw=rc.T@np.diag([1,-1,-1]);model.cam_pos[0]=-rc.T@tc
        model.cam_quat[0]=Rotation.from_matrix(ctw).as_quat()[[3,0,1,2]]
        data.qpos[:3]=case['position_m'];data.qpos[3:7]=np.array(case['quaternion_xyzw'])[[3,0,1,2]]
        data.qvel[:]=0
        if np.isfinite(v).all():
            center=v.mean(0);data.mocap_pos[mocap_id]=center
            model.mesh_vert[mesh_offset:mesh_offset+mesh_count]=(v-center-geom_pos)@geom_rot
        else:data.mocap_pos[mocap_id]=[0,0,-10]
        mujoco.mj_forward(model,data)
        mujoco.mjr_uploadMesh(model,renderer._mjr_context,mid)
        renderer.update_scene(data,camera='source')
        fitted=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
        renderer.update_scene(data,camera=cam)
        orbit=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
        source=cv2.imread(str(hand_run/'frames'/f"{camera['source_frame']:06d}.jpg"))
        for im,label in [(source,'Original video'),(fitted,'Aligned reconstruction'),(orbit,'3D inspection view')]:
            cv2.rectangle(im,(0,0),(832,38),(30,30,30),-1)
            cv2.putText(im,label+f" | {camera['seconds']:.2f}s",(10,26),0,.65,(255,255,255),1)
        evidence='M3T visual correction + reference prior' if case['mode']=='m3t_with_reference_prior' else ('LOW: hand-relative case prior' if case['mode']=='hand_relative_prior' else (
            'Partial silhouette + priors' if case['mode']=='silhouette_and_hand_prior' else 'Reviewed support interval'))
        cv2.putText(fitted,evidence,(10,426),0,.5,(0,220,255),1)
        cv2.putText(fitted,'Prescribed poses; not physical grasp',(10,450),0,.5,(0,220,255),1)
        frame=np.hstack([source,fitted,orbit]);writer.write(frame)
        if i%10==0:cv2.imwrite(str(output/'comparisons'/f'{i:03d}.jpg'),frame)
        if i%20==0:print(f'Rendered {i}/{len(cameras)}',flush=True)
    writer.release();renderer.close()
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(output/'replay-raw.mp4'),'-c:v','libx264',
                    '-pix_fmt','yuv420p','-movflags','+faststart',str(output/'replay.mp4')],check=True)


def verify(output):
    cases=json.loads((output/'case_poses.json').read_text())
    report=json.loads((output/'report.json').read_text())
    meta=json.loads((Path(report['scene'])/'reconstruction.json').read_text())
    cameras=json.loads((output/'camera_poses.json').read_text())
    size=np.array(meta['priors']['case_dimensions_m']);local=case_vertices(size)
    for row in cases:
        row['confidence']='low' if row['mode'] in ('hand_relative_prior','support_prior') else (
            'medium_with_priors' if row['mode'] in ('silhouette_and_hand_prior','m3t_with_reference_prior') else 'support_constrained')
    (output/'case_poses.json').write_text(json.dumps(cases,indent=2))
    report['pose_confidence_counts']={name:sum(row['confidence']==name for row in cases)
                                     for name in sorted({row['confidence'] for row in cases})}
    (output/'report.json').write_text(json.dumps(report,indent=2))
    pos=np.array([r['position_m'] for r in cases]);rot=Rotation.from_quat([r['quaternion_xyzw'] for r in cases])
    penetrations=[]
    for p,r in zip(pos,rot.as_matrix()):
        xyz=local@r.T+p
        penetrations.append(float(np.maximum(support_height(xyz)-xyz[:,2],0).max()))
    max_step=float(np.linalg.norm(np.diff(pos,axis=0),axis=1).max())
    max_rotation=float((rot[1:]*rot[:-1].inv()).magnitude().max())
    hand=np.load(output/'hand_world.npz');k=np.array(meta['camera']['intrinsics'])
    observed=json.loads((Path(report['hand_run'])/'hands.json').read_text())
    errors=[];hand_pen=[]
    for i,(row,cam) in enumerate(zip(observed,cameras)):
        if not row['hands']:continue
        uv=project(hand['joints'][i],k,np.array(cam['rotation']),np.array(cam['translation']))
        errors.append(float(np.mean(np.linalg.norm(uv-np.array(row['hands'][0]['keypoints_2d']),axis=1))))
        v=hand['vertices'][i]
        hand_pen.append(float(np.maximum(support_height(v)-v[:,2],0).max()))
    checks=dict(finite_case_poses=bool(np.isfinite(pos).all()),all_cameras_registered=len(cases)==len(cameras),
                case_clear_of_supports=max(penetrations)<1e-5,
                bounded_translation_steps=max_step<.04,
                bounded_rotation_steps=max_rotation<1.2,
                hand_clear_of_supports_under_5mm=max(hand_pen)<.005,
                median_predicted_joint_fit_under_8px=float(np.median(errors))<8.)
    result=dict(checks=checks,passed=all(checks.values()),
                maximum_case_support_penetration_m=max(penetrations),maximum_case_step_m=max_step,
                maximum_rotation_step_radians=max_rotation,
                maximum_hand_support_penetration_m=max(hand_pen),
                median_predicted_joint_fit_px=float(np.median(errors)),
                scope='Internal geometric consistency only; does not validate true 6D pose or physical grasp')
    (output/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['recover','render','verify'])
    p.add_argument('output',type=Path);p.add_argument('--scene',type=Path,default=Path('runs/book5-space-03'))
    p.add_argument('--hands',type=Path,default=Path('runs/book5-local-03'))
    args=p.parse_args()
    if args.stage=='recover':recover(args.scene,args.hands,args.output)
    elif args.stage=='render':render(args.output)
    else:verify(args.output)


if __name__=='__main__':main()
