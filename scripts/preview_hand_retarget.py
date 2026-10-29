"""Render source frames beside fitted robot poses. This is explicitly kinematic playback."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import mujoco
import numpy as np

from humanoid.kinematics import set_configuration
from humanoid.book_g1 import BACKGROUND_CONTROL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hands', type=Path, required=True)
    parser.add_argument('--retarget', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_xml_path(str((args.retarget / 'scene.xml').resolve()))
    data = mujoco.MjData(model)
    names = json.loads((args.retarget / 'joint-map.json').read_text())['action_names']
    active = [model.actuator(name).id for name in names]
    track = np.load(args.retarget / 'retarget.npz')
    renderer = mujoco.Renderer(model, height=232, width=416)
    frames = track['source_frame']
    source_fps = json.loads((args.hands / 'manifest.json').read_text())['fps']
    fps = source_fps / 3
    command = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
               '-s', '832x272', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.output / 'kinematic-fit.mp4')]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for i in range(0, len(frames), 3):
            frame = int(frames[i])
            source = cv2.imread(str(args.hands / 'frames' / f'{frame:06d}.jpg'))
            source = cv2.resize(source, (416, 232))
            valid = bool(track['valid'][i])
            if valid:
                control = np.zeros(model.nu)
                control[active] = track['actions'][i]
                for name, value in BACKGROUND_CONTROL.items():
                    control[model.actuator(name).id] = value
                set_configuration(model, data, control)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera='source_match')
            robot = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            panel = np.zeros((272, 832, 3), dtype=np.uint8)
            panel[40:] = np.hstack([source, robot])
            cv2.putText(panel, f'Source | {track["seconds"][i]:.2f} s', (8, 17), 0, .45, (255, 255, 255), 1)
            label = 'Kinematic fit only | object static | no physics execution'
            cv2.putText(panel, label, (420, 17), 0, .36, (255, 255, 255), 1)
            status = (f'Wrist residual: {1000 * track["landmark_error_m"][i, 0]:.1f} mm'
                      if valid else 'Missing hand estimate | previous pose displayed')
            cv2.putText(panel, status, (420, 34), 0, .38, (0, 220, 255), 1)
            encoder.stdin.write(panel.tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError('Video encoding failed')
    renderer.close()
    print(args.output / 'kinematic-fit.mp4')


if __name__ == '__main__':
    main()
