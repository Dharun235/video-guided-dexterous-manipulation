"""Compare source phone footage with the actual motor-driven ego rollout."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import mujoco
import numpy as np


def render(folder):
    model=mujoco.MjModel.from_xml_path(str((folder/'scene.xml').resolve()))
    data=mujoco.MjData(model);mujoco.mj_setState(model,data,np.load(folder/'initial_state.npz')['integration'],mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model,data)
    model.vis.global_.offwidth=832;model.vis.global_.offheight=464
    renderer=mujoco.Renderer(model,height=464,width=832)
    x=np.load(folder/'episode.npz');schema=json.loads((folder/'dataset.json').read_text())
    motion=Path(json.loads((folder/'report.json').read_text())['source_motion'])
    source=json.loads((motion/'report.json').read_text());hand_run=Path(source['hand_run'])
    hands=json.loads((hand_run/'hands.json').read_text())
    writer=cv2.VideoWriter(str(folder/'ego-comparison-raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(2496,464))
    for i,u in enumerate(x['action']):
        row=min(hands,key=lambda h:abs(h['seconds']-x['source_timestamp'][i]))
        original=cv2.imread(str(hand_run/'frames'/f'{row["source_frame"]:06d}.jpg'))
        renderer.update_scene(data,camera='source_match');ego=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
        renderer.update_scene(data,camera='robot_overview');overview=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
        for im,label in [(original,'Original right-hand video'),(ego,'G1 right hand | phone-aligned ego view'),(overview,'G1 facing the table | physics teacher')]:
            cv2.rectangle(im,(0,0),(832,34),(25,25,25),-1);cv2.putText(im,label,(8,24),0,.60,(255,255,255),1)
        combined=np.hstack([original,ego,overview]);writer.write(combined)
        if i in (0,44,100,160,200,len(x['action'])-1):cv2.imwrite(str(folder/f'ego-comparison-{i:03d}.jpg'),combined)
        data.ctrl[schema['active_actuator_indices']]=u
        for _ in range(50):mujoco.mj_step(model,data)
    writer.release();renderer.close()
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(folder/'ego-comparison-raw.mp4'),'-c:v','libx264','-pix_fmt','yuv420p',str(folder/'ego-comparison.mp4')],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('folder',type=Path);a=p.parse_args();render(a.folder)
