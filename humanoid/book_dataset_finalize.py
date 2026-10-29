"""Finalize a complete audited collection into a portable policy-data package."""
import argparse
import contextlib
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

from .book_collection_audit import inspect
from .book_dataset_audit import audit
from training.dataset import RobotDataset


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def preview(folder,entries):
    chosen=[next(e for e in entries if e['split']==split and e['domain_seed']%100==variant)
        for split,variant in (('train',0),('train',1),('validation',0),('test',0))]
    captures=[cv2.VideoCapture(str(folder/e['episode']/'execution.mp4')) for e in chosen]
    raw=folder/'variation-preview-raw.mp4'
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),20,(1024,512))
    last=[None]*4
    for _ in range(max(e['frames'] for e in chosen)):
        panels=[]
        for j,(cap,e) in enumerate(zip(captures,chosen)):
            ok,frame=cap.read()
            if ok:last[j]=cv2.resize(frame,(512,256))
            if last[j] is None:raise ValueError('Cannot read preview video')
            tile=last[j].copy()
            cv2.rectangle(tile,(0,0),(512,28),(20,20,20),-1)
            cv2.putText(tile,f'{e["split"]} group {e["physical_group"]} | teacher',(8,20),0,.48,(255,255,255),1)
            panels.append(tile)
        writer.write(np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])]))
    writer.release()
    for cap in captures:cap.release()
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(raw),'-c:v','libx264','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(folder/'variation-preview.mp4')],check=True)
    raw.unlink()


def finalize(folder):
    folder=folder.resolve();repo=Path(__file__).resolve().parents[1]
    inspect(folder)
    index=json.loads((folder/'index.json').read_text())
    entries=index['episodes'];(folder/'data').mkdir(exist_ok=True)
    states=[];actions=[];panels=[];counts={}
    for split in ('train','validation','test'):
        count=0
        with (folder/'data'/f'{split}.jsonl').open('w') as stream:
            for ei,e in enumerate(entries):
                if e['split']!=split or not e['verified']:continue
                p=folder/e['episode']
                for line in (p/'frames.jsonl').read_text().splitlines():
                    row=json.loads(line);row['episode']=e['episode'];row['episode_index']=ei
                    for key in ('observation.images.front','observation.images.inspection'):
                        row[key]=str(Path(e['episode'])/row[key])
                        if not (folder/row[key]).is_file():raise ValueError('Missing image '+row[key])
                    if not np.isfinite(row['observation.state']+row['action']).all():raise ValueError('Nonfinite data')
                    stream.write(json.dumps(row)+'\n');count+=1
                    if split=='train':states.append(row['observation.state']);actions.append(row['action'])
                frame=min(110,e['frames']-1)
                image=cv2.imread(str(p/'images/observation.images.front'/f'{frame:06d}.jpg'))
                cv2.rectangle(image,(0,0),(256,35),(20,20,20),-1)
                cv2.putText(image,f'{split} g{e["physical_group"]:02d} v{e["domain_seed"]%100}',(5,22),0,.45,(255,255,255),1)
                panels.append(image)
        counts[split]=count
    schema=json.loads((folder/entries[0]['episode']/'dataset.json').read_text())
    stats={'fit_split':'train','samples':len(states)}
    for name,values,names in (('observation.state',states,schema['state_joint_names']),('action',actions,schema['action_actuator_names'])):
        a=np.array(values);stats[name]={'names':names,'mean':a.mean(0).tolist(),'std':a.std(0).tolist(),
            'min':a.min(0).tolist(),'max':a.max(0).tolist()}
    save(folder/'statistics.json',stats)
    while len(panels)%6:panels.append(np.zeros((256,256,3),np.uint8))
    cv2.imwrite(str(folder/'appearance-review.jpg'),np.vstack([np.hstack(panels[i:i+6]) for i in range(0,len(panels),6)]))
    preview(folder,entries)

    # Keep the original reset separate from randomized held-out conditions.
    original=folder/'original-reference';original.mkdir(exist_ok=True)
    reference=repo/'runs/book5-g1-ego-09'
    for name in ('scene.xml','initial_state.npz','dataset.json','report.json','layout-verification.json'):
        shutil.copy2(reference/name,original/name)
    (folder/'source').mkdir(exist_ok=True)
    shutil.copy2(repo/'demonstrations/book5.mp4',folder/'source/book5.mp4')
    source_hash=hashlib.sha256((folder/'source/book5.mp4').read_bytes()).hexdigest()
    if source_hash!=schema['source_video_sha256']:raise ValueError('Source video checksum mismatch')
    for src,name in (('book5-local-03/manifest.json','hand-inference.json'),
                     ('book5-space-03/reconstruction.json','scene-reconstruction.json'),
                     ('book5-motion-06/report.json','motion-estimation.json'),
                     ('book5-motion-06/fusion_evidence.json','image-refinement.json')):
        shutil.copy2(repo/'runs'/src,folder/'source'/name)
    implementation=folder/'implementation';implementation.mkdir(exist_ok=True)
    for name in ('humanoid/book_g1.py','humanoid/book_transfer.py','humanoid/book_variations.py',
                 'humanoid/book_dataset_audit.py','humanoid/book_collection_audit.py',
                 'humanoid/book_dataset_finalize.py','humanoid/kinematics.py','humanoid/layout.py',
                 'training/dataset.py','docs/approach.md','docs/data-and-evaluation.md','LICENSE'):
        target=implementation/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(repo/name,target)
    report=json.loads((original/'report.json').read_text())
    reference_arrays=np.load(reference/'episode.npz')
    save(folder/'reproduction-test.json',dict(source_video='source/book5.mp4',source_video_sha256=source_hash,
        source_task_interval_seconds=[float(reference_arrays['source_timestamp'][0]),float(reference_arrays['source_timestamp'][-1])],
        scene_xml='original-reference/scene.xml',initial_state_npz='original-reference/initial_state.npz',
        schema='original-reference/dataset.json',goal_position_m=report['goal_position_m'],action_hz=20,
        maximum_rollout_seconds=20,teacher_disabled=True,
        interpretation='Run the trained policy closed loop in the original reconstructed scene and compare with source footage. This is reproduction: the source motion contributed to training. It is not an independent generalization test.',
        comparison='Show the source task interval beside the actual policy rollout. Policy timing is generated by the model, not forced to follow source timestamps.',
        final_deliverables=['trained SmolVLA checkpoint and configuration','closed-loop original-scene rollout video',
            'side-by-side source video comparison','success/failure on all locked held-out scenarios']))
    for src,dst in (('third_party/menagerie/unitree_g1/LICENSE','G1_LICENSE.txt'),
                    ('third_party/unitree-xr-book/LICENSE','INSPIRE_LICENSE.txt'),
                    ('humanoid/BOOK_G1_UPSTREAM.json','ROBOT_UPSTREAM.json'),
                    ('humanoid/UPSTREAM.json','G1_UPSTREAM.json')):
        shutil.copy2(repo/src,folder/dst)
    # Deduplicate shared external assets; retain implicit mesh names before renaming.
    shared=folder/'shared_assets';shared.mkdir(exist_ok=True)
    scene_count=0
    for scene in sorted(folder.rglob('scene.xml')):
        tree=ET.parse(scene);root=tree.getroot()
        for asset in root.findall('asset/*'):
            if not asset.get('file'):continue
            source=Path(asset.get('file'))
            if not source.is_absolute():source=(scene.parent/source).resolve()
            if asset.tag=='mesh' and not asset.get('name'):asset.set('name',source.stem)
            if not source.is_relative_to(folder):
                digest=hashlib.sha256(source.read_bytes()).hexdigest()
                target=shared/(digest[:16]+'_'+source.name)
                if not target.exists():shutil.copy2(source,target)
                source=target
            asset.set('file',os.path.relpath(source,scene.parent))
        tree.write(scene,encoding='unicode')
        mujoco.MjModel.from_xml_path(str(scene));scene_count+=1
    portable_audits=[]
    with (folder/'portable-replay-checks.log').open('w') as log,contextlib.redirect_stdout(log):
        for split in ('train','validation','test'):
            entry=next(e for e in entries if e['split']==split)
            audit(folder/entry['episode'],check_render=True)
            portable_audits.append(entry['episode'])
    smoke={}
    for split in counts:
        ds=RobotDataset(folder,split,50)
        for i in sorted({0,len(ds)//2,len(ds)-1}):
            sample=ds[i]
            assert sample['observation.state'].shape==(19,)
            assert sample['action'].shape==(50,13)
            for key in ('observation.images.front','observation.images.inspection'):
                assert sample[key].shape==(3,256,256) and np.isfinite(sample[key]).all()
        smoke[split]={'samples':len(ds),'representative_samples_decoded':3}
    save(folder/'preparation.json',dict(data_ready=True,training_started=False,frames=counts,
        runtime=dict(python=platform.python_version(),mujoco=mujoco.__version__,numpy=np.__version__,opencv=cv2.__version__),
        portable_scenes_compiled=scene_count,portable_render_replay_audits=portable_audits,
        loader_checks=smoke,action_horizon=50,
        source_recordings=1,physical_groups=index['groups'],visual_variants_per_group=index['variants_per_group'],
        scope='Small-task imitation pilot; teacher demonstrations, not learned-policy success. Native LeRobot conversion and fine-tuning remain next.'))
    with (folder/'README.md').open('a') as stream:
        stream.write('\nPortable package: every scene references assets inside this folder. '
            'data/{train,validation,test}.jsonl contains policy-only rows; statistics.json is fitted on training rows only. '
            'training.dataset.RobotDataset reads RGB/state and masked action chunks without crossing episode boundaries. '
            'The original source is in source/book5.mp4; reproduction-test.json fixes the final original-scene comparison. '
            'No SmolVLA checkpoint has been trained yet.\n')
    hashes={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    save(folder/'manifest.json',{'files_sha256':hashes,'source_video_sha256':source_hash})
    print(json.dumps({'frames':counts,'portable_scenes':scene_count,'data_ready':True},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('folder',type=Path)
    finalize(p.parse_args().folder)
