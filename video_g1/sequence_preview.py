"""Render the complete recording beside source-aligned physical G1 motion."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from humanoid.book_g1 import ORIGIN, SCENE_ROTATION


def render(reference_path, rollout_path, cameras_path, source, output, stride=3):
    """Show source-time playback with every phase retained and retiming disclosed."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg is required to encode the comparison for browser playback")
    output.mkdir(parents=True, exist_ok=False)
    ref = np.load(reference_path / "reference.npz")
    run = np.load(rollout_path / "rollout.npz")
    cam = np.load(cameras_path / "cameras.npz")
    model = mujoco.MjModel.from_xml_path(str((reference_path / "scene.xml").resolve()))
    data = mujoco.MjData(model)
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(
        str(output / "comparison-raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (1280, 832)
    )
    frames = list(range(0, len(ref["source_frame"]), stride))
    if frames[-1] != len(ref["source_frame"]) - 1:
        frames.append(len(ref["source_frame"]) - 1)
    camera = model.camera("source_match").id
    parent = model.cam_bodyid[camera]
    robot_root = model.body("pelvis").id
    arm_root = model.body("right_shoulder_pitch_link").id

    def descendant(body, root):
        while body:
            if body == root:
                return True
            body = int(model.body_parentid[body])
        return False

    overview = mujoco.MjvCamera()
    overview.lookat[:] = [0.24, -0.05, 0.75]
    overview.distance = 1.65
    overview.azimuth = 140
    overview.elevation = -20
    hidden = {
        g
        for g in range(model.ngeom)
        if descendant(int(model.geom_bodyid[g]), robot_root)
        and not descendant(int(model.geom_bodyid[g]), arm_root)
    }
    contact_camera = mujoco.MjvCamera()
    contact_camera.distance = 0.45
    contact_camera.azimuth = 90
    contact_camera.elevation = -20
    with mujoco.Renderer(model, height=356, width=640) as renderer:
        for i in frames:
            data.qpos[:] = run["qpos"][i]
            mujoco.mj_forward(model, data)
            r = cam["rotation_world_to_camera"][i]
            t = cam["translation_world_to_camera_m"][i]
            pos = SCENE_ROTATION @ (-r.T @ t) + ORIGIN
            rotation = SCENE_ROTATION @ r.T @ np.diag([1, -1, -1])
            basis = data.xmat[parent].reshape(3, 3)
            model.cam_pos[camera] = basis.T @ (pos - data.xpos[parent])
            model.cam_quat[camera] = Rotation.from_matrix(basis.T @ rotation).as_quat()[[3, 0, 1, 2]]
            mujoco.mj_camlight(model, data)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(ref["source_frame"][i]))
            ok, original = cap.read()
            if not ok:
                raise RuntimeError(f"Missing source frame {i}")
            images = [cv2.resize(original, (640, 356))]
            renderer.update_scene(data, camera="source_match")
            for geom in renderer.scene.geoms[: renderer.scene.ngeom]:
                if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid in hidden:
                    geom.rgba[3] = 0
            images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
            renderer.update_scene(data, camera=overview)
            images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
            object_position = data.xpos[model.body("case").id]
            hand_centre = run["landmarks_m"][i, [0, 4, 8, 12, 16, 20]].mean(axis=0)
            separation = np.linalg.norm(hand_centre - object_position)
            if separation > 0.2:
                contact_camera.lookat[:] = 0.5 * (object_position + hand_centre)
                contact_camera.distance = 0.45 + 1.5 * (separation - 0.2)
            else:
                contact_camera.lookat[:] = object_position + [0, 0, 0.02]
                contact_camera.distance = 0.45
            renderer.update_scene(data, camera=contact_camera)
            images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
            canvas = np.zeros((832, 1280, 3), np.uint8)
            angle = np.rad2deg(
                Rotation.from_matrix(run["case_rotation"][i] @ run["case_rotation"][0].T).magnitude()
            )
            for j, (im, title) in enumerate(
                zip(
                    images,
                    [
                        "Recorded demonstration",
                        "Physical hand; torso hidden visually",
                        "Physical G1; supported pelvis",
                        "Hand/contact inspection; native Inspire",
                    ],
                )
            ):
                x, y = (j % 2) * 640, (j // 2) * 416
                canvas[y + 60 : y + 416, x : x + 640] = im
                cv2.putText(canvas, title, (x + 7, y + 17), 0, 0.45, (255, 255, 255), 1)
                cv2.putText(
                    canvas,
                    f"frame {i} | source {ref['source_time_s'][i]:.2f}s | physics {run['execution_time_s'][i]:.2f}s",
                    (x + 7, y + 35),
                    0,
                    0.35,
                    (220, 220, 220),
                    1,
                )
                cv2.putText(
                    canvas,
                    f"case turn {angle:.1f} deg | opposition {run['case_opposition'][i]:.2f}"
                    if j
                    else "Source-time playback; simulation motion is retimed",
                    (x + 7, y + 51),
                    0,
                    0.37,
                    (180, 230, 180),
                    1,
                )
            writer.write(canvas)
            if i in [90, 135, 240, 330, 390, 450, 480, 570, 600, 630, 675, 690, 705, 747]:
                cv2.imwrite(str(output / f"frame-{i:04d}.jpg"), canvas)
    writer.release()
    cap.release()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(output / "comparison-raw.mp4"),
            "-c:v",
            "libx264",
            "-crf",
            "25",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output / "comparison.mp4"),
        ],
        check=True,
    )
    (output / "comparison-raw.mp4").unlink()
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "source_frames": len(ref["source_frame"]),
                "preview_frames": len(frames),
                "stride": stride,
                "playback": "Source time; physical execution is separately retimed",
                "all_phases_included": True,
                "visibility": "Only phone-view torso hidden; all collision geometry remains active",
            },
            indent=2,
        )
        + "\n"
    )


def main():
    """Render source video and numerical physics states; requires graphics access."""
    p = argparse.ArgumentParser(__doc__)
    for name in ["reference", "rollout", "cameras", "source", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    render(args.reference, args.rollout, args.cameras, args.source, args.output)


if __name__ == "__main__":
    main()
