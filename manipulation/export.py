"""Export synchronized simulated cameras, states and actions from a physics rollout."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--rollout", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--size", type=int, default=256)
    a = p.parse_args()
    if a.stride < 1 or a.size < 32:
        p.error("Positive stride and image size >=32 required")
    a.output.mkdir(parents=True, exist_ok=False)
    m = mujoco.MjModel.from_xml_path((a.rollout / "scene-path.txt").read_text().strip())
    d = mujoco.MjData(m)
    x = np.load(a.rollout / "rollout.npz")
    ids = np.arange(0, len(x["time_s"]), a.stride)
    r = mujoco.Renderer(m, height=a.size, width=a.size)
    cameras = ["front_dataset", "inspection_dataset"]
    paths = {name: [] for name in cameras}
    try:
        for name in cameras:
            (a.output / name).mkdir()
        for k, i in enumerate(ids):
            d.qpos[:] = x["qpos"][i]
            mujoco.mj_forward(m, d)
            for name in cameras:
                r.update_scene(d, camera=name)
                image = r.render()
                relative = f"{name}/{k:06d}.jpg"
                if not cv2.imwrite(
                    str(a.output / relative),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88],
                ):
                    raise RuntimeError("Image write failed")
                paths[name].append(relative)
    finally:
        r.close()
    data = {key: x[key][ids] for key in x.files if key != "qpos"}
    for name in cameras:
        data[f"image_{name}"] = paths[name]
    np.savez_compressed(a.output / "episode.npz", **data)
    report = json.loads((a.rollout / "report.json").read_text())
    schema = json.loads((a.rollout / "schema.json").read_text())
    manifest = dict(
        frames=len(ids),
        fps=1 / float(np.median(np.diff(data["time_s"]))),
        image_shape=[a.size, a.size, 3],
        source_rollout_sha256=hashlib.sha256((a.rollout / "rollout.npz").read_bytes()).hexdigest(),
        schema=schema,
        execution_report=report,
        observations_are_simulated=True,
        control_update_hz=1000,
        physics_log_hz=50,
        export_is_downsampled=True,
        ready_for_training=False,
        action_meaning="Samples of 1 kHz closed-loop position targets; 13 controls, not human actions or a verified 10 Hz zero-order-hold policy",
        touch_meaning="Five fingers x external normal/shear contact-force summaries in N; no physical tactile sensor model",
        camera_meaning="Fixed robot-mounted front camera and inspection camera; not the moving original phone camera",
    )
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    from .data import validate_episode

    validation = validate_episode(a.output)
    (a.output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(f"Exported and checked {len(ids)} synchronized frames from each of two cameras")


if __name__ == "__main__":
    main()
