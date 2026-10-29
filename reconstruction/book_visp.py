"""Independent CPU RGB model-based tracking experiment with upstream ViSP.

Only the initial pose is provided to ViSP; no hand poses or reference trajectory
are fed during tracking. Existing reconstruction poses are used separately for
evaluation ROIs, which are not independent ground truth.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .book_space import case_vertices, project
from .book_motion import segment_case


def write_model(path,size,shape):
    if shape=='rounded':
        vertices=case_vertices(size);n=len(vertices)//4
        faces=[list(reversed(range(n))),list(range(3*n,4*n))]
        for ring in range(3):
            for i in range(n):
                j=(i+1)%n;faces.append([ring*n+i,ring*n+j,(ring+1)*n+j,(ring+1)*n+i])
    else:
        w,d,h=np.array(size)/2
        vertices=np.array([[-w,-d,-h],[w,-d,-h],[w,d,-h],[-w,d,-h],
                           [-w,-d,h],[w,-d,h],[w,d,h],[-w,d,h]])
        faces=[[3,2,1,0],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
    lines=['V1',str(len(vertices))]+[' '.join(map(str,v)) for v in vertices]+['0','0',str(len(faces))]
    lines+=[' '.join(map(str,[len(f),*f])) for f in faces]+['0','0']
    path.write_text('\n'.join(lines)+'\n')


def camera_at(rows,seconds):
    times=np.array([r['seconds'] for r in rows])
    tc=np.array([r['translation'] for r in rows])
    rot=Slerp(times,Rotation.from_rotvec([r['rotation'] for r in rows]))(np.clip(seconds,times[0],times[-1]))
    return rot.as_matrix(),np.array([np.interp(seconds,times,tc[:,j]) for j in range(3)])


def run(scene,motion,output,binary,direction,shape,backend='visp'):
    started=time.monotonic();output.mkdir(parents=True,exist_ok=False)
    meta=json.loads((scene/'reconstruction.json').read_text())
    k=np.array(meta['camera']['intrinsics']);size=meta['priors']['case_dimensions_m']
    cams=json.loads((motion/'camera_poses.json').read_text());poses=json.loads((motion/'case_poses.json').read_text())
    write_model(output/'case.cao',size,shape)
    first=poses[0] if direction=='forward' else poses[-1]
    rc,tc=camera_at(cams,first['seconds'])
    co=rc@Rotation.from_quat(first['quaternion_xyzw']).as_matrix()
    ct=rc@np.array(first['position_m'])+tc
    transform=np.eye(4);transform[:3,:3]=co;transform[:3,3]=ct
    (output/'seed.txt').write_text('832 464 '+ ' '.join(map(str,[k[0,0],k[1,1],k[0,2],k[1,2]]))+'\n'+
                                 '\n'.join(' '.join(map(str,row)) for row in transform)+'\n')
    cap=cv2.VideoCapture(meta['source']);fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Keep one small RGB clip in memory to avoid reverse-seeking every frame.
    frames=[]
    while True:
        ok,image=cap.read()
        if not ok:break
        frames.append(image)
    cap.release()
    log=open(output/'native.log','w')
    model_path=(output/'case.cao').resolve()
    if backend=='m3t':
        if shape=='rounded':model_path=(scene/'assets/case.obj').resolve()
        else:
            import trimesh
            trimesh.creation.box(extents=size).export(output/'case.obj')
            model_path=(output/'case.obj').resolve()
    job=subprocess.Popen([str(binary.resolve()),str(model_path),
                          str((output/'seed.txt').resolve()),str((output/'raw.csv').resolve())],
                          stdin=subprocess.PIPE,stdout=log,stderr=log)
    order=range(len(frames)) if direction=='forward' else range(len(frames)-1,-1,-1)
    try:
        for i in order:
            array=frames[i] if backend=='m3t' else cv2.cvtColor(frames[i],cv2.COLOR_BGR2GRAY)
            job.stdin.write(array.tobytes())
        job.stdin.close();job.wait()
    except BrokenPipeError:
        job.wait()
    log.close()
    if job.returncode!=0:raise RuntimeError((output/'native.log').read_text()[-2000:])
    rows=list(csv.DictReader((output/'raw.csv').open()))
    native_time=[];accepted=[];data=[]
    vertices=case_vertices(size)
    writer=cv2.VideoWriter(str(output/'tracking-raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(832,464))
    lookup={i:r for i,r in enumerate(rows)} if direction=='forward' else {count-1-i:r for i,r in enumerate(rows)}
    (output/'frames').mkdir()
    for frame_index,image in enumerate(frames):
        sec=frame_index/fps;row=lookup.get(frame_index)
        if row is None:continue
        matrix=np.array([float(row[f'm{i}']) for i in range(16)]).reshape(4,4)
        uv=project(vertices,k,cv2.Rodrigues(matrix[:3,:3])[0],matrix[:3,3])
        finite=bool(np.isfinite(uv).all())
        inside=finite and matrix[2,3]>.1 and matrix[2,3]<1.2 and np.abs(uv).max()<5000
        # Evaluation mask ROI uses the old replay, never passed to the tracker.
        ref=min(poses,key=lambda r:abs(r['seconds']-sec));rc,tc=camera_at(cams,sec)
        center=project([ref['position_m']],k,cv2.Rodrigues(rc)[0],tc)[0]
        contour,_,solidity=segment_case(image,center)
        overlap=None
        if contour is not None and solidity>.72 and inside:
            observed=np.zeros((464,832),np.uint8);cv2.fillPoly(observed,[contour.astype(int)],1)
            predicted=np.zeros_like(observed);cv2.fillConvexPoly(predicted,cv2.convexHull(uv.astype(np.int32)),1)
            overlap=float(np.logical_and(observed,predicted).sum()/max(np.logical_or(observed,predicted).sum(),1))
        edge_ok=backend=='m3t' or (float(row['projection_error_deg'])<25 and int(row['edge_features'])>=8)
        ok=bool(int(row['tracker_ok']) and inside and edge_ok and overlap is not None and overlap>=.35)
        accepted.append(ok);native_time.append(float(row['milliseconds']))
        ctw=np.eye(4);ctw[:3,:3]=rc;ctw[:3,3]=tc
        world=np.linalg.inv(ctw)@matrix
        data.append(dict(source_frame=frame_index,seconds=sec,tracker_reported_ok=bool(int(row['tracker_ok'])),
                         accepted_by_heuristic=ok,visible_mask_iou=overlap,
                         edge_angle_error_deg=float(row['projection_error_deg']) if backend=='visp' else None,
                         edge_features=int(row['edge_features']) if backend=='visp' else None,
                         object_to_camera=matrix.tolist(),object_to_world=world.tolist()))
        if inside:
            color=(0,220,0) if ok else (0,0,255)
            cv2.polylines(image,[cv2.convexHull(uv.astype(np.int32))],True,color,2)
            axes=project([[0,0,0],[.04,0,0],[0,.04,0],[0,0,.04]],k,
                         cv2.Rodrigues(matrix[:3,:3])[0],matrix[:3,3])
            for j,color_axis in enumerate([(0,0,255),(0,255,0),(255,0,0)]):
                cv2.line(image,tuple(axes[0].astype(int)),tuple(axes[j+1].astype(int)),color_axis,2)
        cv2.rectangle(image,(0,0),(832,54),(25,25,25),-1)
        cv2.putText(image,f'{backend.upper()} CPU {direction} {shape} | {sec:.2f}s | '+('accepted candidate' if ok else 'unverified / rejected'),
                    (8,24),0,.52,(255,255,255),1)
        cv2.putText(image,'Automatic after initial seed; green/red are heuristic checks, not pose ground truth',
                    (8,45),0,.40,(255,255,255),1)
        writer.write(image)
        if frame_index%60==0:cv2.imwrite(str(output/'frames'/f'{frame_index:06d}.jpg'),image)
    writer.release()
    (output/'poses.json').write_text(json.dumps(data,indent=2))
    bins={}
    for start,end in [(0,3.5),(3.5,5.05),(5.05,10),(10,12.5)]:
        subset=[r for r in data if start<=r['seconds']<end]
        bins[f'{start}-{end}s']=dict(frames=len(subset),heuristic_accepted=sum(r['accepted_by_heuristic'] for r in subset))
    report=dict(library='ViSP' if backend=='visp' else 'M3T',
                source_revision='53a7da87095813c8b4376acc52a6ea55fcb14f62' if backend=='visp' else 'f02106186a1e756450226b3b793e309506817c58',
                tracker='CPU model-based '+('moving edges' if backend=='visp' else 'RGB region')+'; no hand trajectory input',direction=direction,model_shape=shape,
                initial_seed='existing fitted first/final case pose',processed_frames=len(data),
                median_tracker_ms=float(np.median(native_time)),heuristic_accepted=sum(accepted),intervals=bins,
                elapsed_seconds=time.monotonic()-started,
                limitations=['Not automatic unknown-object detection; known CAD and one initial pose required',
                             'Mask IoU is a heuristic using evaluation ROIs from the earlier reconstruction',
                             'No ground-truth 6D poses; automatic tracking can drift under hand occlusion',
                             'Rejected poses must not replace the previous reconstruction'])
    (output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(output/'tracking-raw.mp4'),'-c:v','libx264',
                    '-pix_fmt','yuv420p','-movflags','+faststart',str(output/'tracking.mp4')],check=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('output',type=Path)
    p.add_argument('--direction',choices=['forward','backward'],default='forward')
    p.add_argument('--shape',choices=['box','rounded'],default='rounded')
    p.add_argument('--backend',choices=['visp','m3t'],default='visp')
    p.add_argument('--scene',type=Path,default=Path('runs/book5-space-03'))
    p.add_argument('--motion',type=Path,default=Path('runs/book5-motion-04'))
    p.add_argument('--binary',type=Path)
    a=p.parse_args()
    binary=a.binary or Path('native/build/book-m3t-tracker/book_m3t_tracker' if a.backend=='m3t' else 'native/build/book-pose-tracker/book_pose_tracker')
    run(a.scene,a.motion,a.output,binary,a.direction,a.shape,a.backend)


if __name__=='__main__':main()
