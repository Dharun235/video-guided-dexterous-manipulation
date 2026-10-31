"""Render source-aligned kinematic and physical views without hiding failed segments."""

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from humanoid.book_g1 import ORIGIN, SCENE_ROTATION
from humanoid.kinematics import set_configuration
from manipulation.fit import control_vector


def run(reference_path, rollout_path, cameras_path, source, output, stride=4):
    """Make a complete-timeline comparison; each panel retains the source frame number."""
    output.mkdir(parents=True, exist_ok=False)
    ref = np.load(reference_path / "reference.npz")
    run = np.load(rollout_path / "rollout.npz")
    cameras = np.load(cameras_path / "cameras.npz")
    m = mujoco.MjModel.from_xml_path(str((reference_path / "scene.xml").resolve()))
    proposed = mujoco.MjData(m)
    physical = mujoco.MjData(m)
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(
        str(output / "comparison.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (1664, 272)
    )
    frames = list(range(0, len(ref["source_frame"]), stride))
    if frames[-1] != len(ref["source_frame"]) - 1:
        frames.append(len(ref["source_frame"]) - 1)
    camera = m.camera("source_match").id
    torso = m.cam_bodyid[camera]
    mujoco.mj_forward(m, proposed)

    def descendant(body, ancestor):
        while body:
            if body == ancestor:
                return True
            body = int(m.body_parentid[body])
        return False

    hidden = {
        g
        for g in range(m.ngeom)
        if descendant(int(m.geom_bodyid[g]), m.body("pelvis").id)
        and not descendant(int(m.geom_bodyid[g]), m.body("right_shoulder_pitch_link").id)
    }
    with mujoco.Renderer(m, height=232, width=416) as renderer:
        for i in frames:
            r = cameras["rotation_world_to_camera"][i]
            t = cameras["translation_world_to_camera_m"][i]
            position = SCENE_ROTATION @ (-r.T @ t) + ORIGIN
            rotation = SCENE_ROTATION @ r.T @ np.diag([1, -1, -1])
            parent = proposed.xmat[torso].reshape(3, 3)
            m.cam_pos[camera] = parent.T @ (position - proposed.xpos[torso])
            m.cam_quat[camera] = Rotation.from_matrix(parent.T @ rotation).as_quat()[[3, 0, 1, 2]]
            set_configuration(m, proposed, control_vector(m, ref["active_actuators"], ref["actions"][i]))
            mujoco.mj_forward(m, proposed)
            physical.qpos[:] = run["qpos"][i]
            mujoco.mj_forward(m, physical)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(ref["source_frame"][i]))
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read source frame {i}")
            images = [cv2.resize(image, (416, 232))]
            for data in [proposed, physical]:
                parent = data.xmat[torso].reshape(3, 3)
                m.cam_pos[camera] = parent.T @ (position - data.xpos[torso])
                m.cam_quat[camera] = Rotation.from_matrix(parent.T @ rotation).as_quat()[[3, 0, 1, 2]]
                mujoco.mj_camlight(m, data)
                renderer.update_scene(data, camera="source_match")
                for geom in renderer.scene.geoms[: renderer.scene.ngeom]:
                    if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid in hidden:
                        geom.rgba[3] = 0
                images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
            renderer.update_scene(physical, camera="robot_overview")
            images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
            canvas = np.zeros((272, 1664, 3), np.uint8)
            for j, (image, title) in enumerate(
                zip(
                    images,
                    [
                        "Recorded video",
                        "Fitted hand; torso hidden",
                        "Physics; torso hidden",
                        "Physics; full robot",
                    ],
                )
            ):
                canvas[40:, 416 * j : 416 * (j + 1)] = image
                cv2.putText(
                    canvas, title, (416 * j + 8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1
                )
                cv2.putText(
                    canvas,
                    f"frame {i} | source {ref['source_time_s'][i]:.2f}s | execution {ref['execution_time_s'][i]:.2f}s",
                    (416 * j + 8, 33),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (220, 220, 220),
                    1,
                )
            writer.write(canvas)
            if i in [frames[min(len(frames) - 1, j)] for j in [15, 37, 60, 100, 125, 150]]:
                cv2.imwrite(str(output / f"frame-{i:04d}.jpg"), canvas)
    writer.release()
    cap.release()
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "all_source_timeline_shown": True,
                "preview_stride": stride,
                "source_frames": len(ref["source_frame"]),
                "preview_frames": len(frames),
                "playback": "Source-time playback; execution is separately retimed",
                "camera": "Estimated moving phone view, not robot onboard camera",
                "diagnostic_visibility": "Torso and other limbs hidden visually in phone views only; collisions unchanged; full body at right",
                "case_in_middle_panel": "Static at reset; only hand kinematics shown",
            },
            indent=2,
        )
        + "\n"
    )


def main():
    """Render the full source timeline next to proposed and physically executed motion."""
    p = argparse.ArgumentParser(__doc__)
    for name in ["reference", "rollout", "cameras", "source", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    run(a.reference, a.rollout, a.cameras, a.source, a.output)


if __name__ == "__main__":
    main()
