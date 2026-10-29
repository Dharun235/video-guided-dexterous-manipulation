"""Check free-object settling and render a candidate scene; this is not a grasp test."""
import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    args = parser.parse_args()
    folder = args.scene.parent
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initial = data.qpos.copy()
    penetration = speed = 0.
    steps = round(5 / model.opt.timestep)
    for _ in range(steps):
        mujoco.mj_step(model, data)
        speed = max(speed, float(np.linalg.norm(data.qvel)))
        for contact in data.contact:
            penetration = max(penetration, float(-contact.dist))
    report = dict(duration_s=steps * model.opt.timestep,
                  finite=bool(np.isfinite(data.qpos).all()),
                  initial_position=initial[:3].tolist(), final_position=data.qpos[:3].tolist(),
                  final_velocity_norm=float(np.linalg.norm(data.qvel)),
                  final_linear_speed_m_s=float(np.linalg.norm(data.qvel[:3])),
                  final_angular_speed_rad_s=float(np.linalg.norm(data.qvel[3:])),
                  maximum_velocity_norm=speed, maximum_penetration_m=penetration,
                  final_contact_count=data.ncon,
                  scope="Free case settling on book under gravity; no robot grasp")
    (folder / "physics.json").write_text(json.dumps(report, indent=2) + "\n")
    renderer = mujoco.Renderer(model, height=464, width=832)
    data.qpos[:] = initial
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="source")
    cv2.imwrite(str(folder / "source-view.jpg"),
                cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, .03]
    camera.distance = .38
    camera.azimuth = 135
    camera.elevation = -25
    renderer.update_scene(data, camera=camera)
    cv2.imwrite(str(folder / "perspective.jpg"),
                cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    renderer.close()
    print(json.dumps(report, indent=2))
    if (not report["finite"] or report["final_linear_speed_m_s"] > .005
            or report["final_angular_speed_rad_s"] > .05):
        raise RuntimeError("Candidate did not settle reliably")


if __name__ == "__main__":
    main()
