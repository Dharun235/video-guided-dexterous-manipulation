"""Show actual dataset cameras and force samples together, not a learned policy."""
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


def preview_episode(root,entry,name):
    folder=root/entry['episode'];x=np.load(folder/'episode.npz')
    rows=[json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
    capture=cv2.VideoCapture(str(folder/'execution.mp4'))
    raw=root/(name+'-raw.mp4');writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),20,(768,256))
    for i in range(len(x['action'])):
        ok,images=capture.read()
        if not ok:raise ValueError('Missing teacher video frame')
        panel=np.zeros((256,256,3),np.uint8)
        cv2.putText(panel,'Simulated fingertip forces',(6,20),0,.45,(255,255,255),1)
        cv2.putText(panel,'Blue: normal | Orange: shear',(6,38),0,.34,(210,210,210),1)
        for j,finger in enumerate(('Thumb','Index','Middle','Ring','Pinky')):
            y=60+j*33;normal,shear=x['observation_tactile'][i,2*j:2*j+2]
            cv2.putText(panel,f'{finger}: {normal:.1f} / {shear:.1f} N',(6,y),0,.37,(255,255,255),1)
            cv2.rectangle(panel,(6,y+5),(6+round(min(normal,40)/40*235),y+10),(240,160,60),-1)
            cv2.rectangle(panel,(6,y+12),(6+round(min(shear,40)/40*235),y+16),(40,140,250),-1)
        cv2.putText(panel,'Teacher phase: '+rows[i]['supervision.phase'],(6,239),0,.37,(255,255,255),1)
        cv2.rectangle(images,(0,0),(512,25),(0,0,0),-1)
        cv2.putText(images,f'Contact-controlled teacher | t={i/20:.2f}s',(5,18),0,.43,(255,255,255),1)
        writer.write(np.hstack([images,panel]))
    capture.release();writer.release()
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(raw),'-c:v','libx264','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(root/(name+'.mp4'))],check=True);raw.unlink()


def previews(root,entries):
    standard=next(e for e in entries if e['split']=='train' and e['domain_seed']%100==0)
    recoveries=[e for e in entries if e['split']=='train' and e['domain_seed']%100==0 and
        json.loads((root/e['episode']/'report.json').read_text())['policy_derived_reset_prefix_steps']==30]
    preview_episode(root,standard,'training-sample')
    if recoveries:preview_episode(root,recoveries[0],'recovery-sample')
