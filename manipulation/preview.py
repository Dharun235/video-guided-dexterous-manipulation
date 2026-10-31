"""Render recorded physics states; rendering never changes the saved rollout."""

import argparse
import subprocess
from pathlib import Path

import cv2
import mujoco
import numpy as np


def main():
    """Render recorded physics states for inspection without re-executing controls."""
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--rollout", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stills", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    m = mujoco.MjModel.from_xml_path((a.rollout / "scene-path.txt").read_text().strip())
    d = mujoco.MjData(m)
    x = np.load(a.rollout / "rollout.npz")
    r = mujoco.Renderer(m, height=360, width=480)

    def panel(i):
        d.qpos[:] = x["qpos"][i]
        mujoco.mj_forward(m, d)
        imgs = []
        for camera in ["robot_overview", "inspection_dataset"]:
            r.update_scene(d, camera=camera)
            imgs.append(cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
        image = np.hstack(imgs)
        cv2.putText(
            image,
            f"Actuator-driven diagnostic | t={x['time_s'][i]:.2f}s",
            (12, 24),
            0,
            0.55,
            (0, 220, 255),
            1,
        )
        return image

    if a.stills:
        for t in [0, 1, 3, 5, 8]:
            i = int(np.argmin(abs(x["time_s"] - t)))
            cv2.imwrite(str(a.output / f"{t:02d}.jpg"), panel(i))
    else:
        fps = 1 / np.median(np.diff(x["time_s"]))
        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            "960x360",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(a.output / "physics.mp4"),
        ]
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as enc:
            for i in range(len(x["time_s"])):
                enc.stdin.write(panel(i).tobytes())
            enc.stdin.close()
            if enc.wait():
                raise RuntimeError("Encoding failed")
    r.close()


if __name__ == "__main__":
    main()
