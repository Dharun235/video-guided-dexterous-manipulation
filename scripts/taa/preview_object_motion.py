"""Compare source frames to estimated case poses. Pose replay is kinematic only."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from reconstruction.book_space import BOOK_SIZE, corners


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--taa", type=Path, required=True)
    parser.add_argument("--taa-check", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--track", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    track = np.load(args.track / "object_track.npz")
    manifest = json.loads((args.taa / "input-manifest.json").read_text())
    cameras = {r["source_frame"]: r for r in json.loads(args.cameras.read_text())}
    point_file = args.track / "feature_pixels.npy"
    points = np.load(point_file) if point_file.exists() else None
    k = np.asarray(json.loads((args.taa_check / "sam3d_scaled/case/depth_scale.json").read_text())["intrinsics"])
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=464, width=832)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "source")
    model.cam_fovy[camera_id] = np.degrees(2 * np.arctan(232 / k[1, 1]))
    temporary = args.output / "preview-raw.mp4"
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), manifest["fps"], (832, 296))
    tiles = []
    for i in range(len(track["valid"])):
        source = cv2.imread(str(args.taa / "frames" / f"{i:06d}.jpg"))
        accepted = bool(track["valid"][i])
        uncertain = bool(track["requires_review"][i]) if "requires_review" in track.files else False
        if accepted:
            pixels = points[i][np.isfinite(points[i]).all(axis=1)] if points is not None else []
            for x, y in pixels:
                cv2.circle(source, (round(float(x)), round(float(y))), 1, (30, 220, 30), -1)
            _, rvec, tvec = cv2.solvePnP(corners(BOOK_SIZE[2]), np.asarray(cameras[i]["book_quad"]), k, None)
            rotation = cv2.Rodrigues(rvec)[0]
            object_rvec = cv2.Rodrigues(rotation @ track["rotation_world"][i])[0]
            object_translation = rotation @ track["translation_world_m"][i] + tvec.ravel()
            cv2.drawFrameAxes(source, k, None, object_rvec, object_translation, .02, 1)
            model.cam_pos[camera_id] = (-rotation.T @ tvec).ravel()
            q = Rotation.from_matrix(rotation.T @ np.diag([1., -1., -1.])).as_quat()
            model.cam_quat[camera_id] = q[[3, 0, 1, 2]]
            q = Rotation.from_matrix(track["rotation_world"][i]).as_quat()
            data.qpos[:] = np.r_[track["translation_world_m"][i], q[[3, 0, 1, 2]]]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="source")
            simulation = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        else:
            simulation = np.full_like(source, 25)
            cv2.putText(simulation, "POSE REJECTED / UNOBSERVED", (100, 230), 0, .8, (70, 90, 250), 2)
        tile = np.zeros((296, 832, 3), np.uint8)
        tile[32:264, :416] = cv2.resize(source, (416, 232))
        tile[32:264, 416:] = cv2.resize(simulation, (416, 232))
        cv2.putText(tile, "Recorded video + estimated case axes", (8, 22), 0, .45, (255, 255, 255), 1)
        cv2.putText(tile, "MuJoCo pose estimate (kinematic)", (425, 22), 0, .47, (255, 255, 255), 1)
        status = "calibration anchor" if i == 0 else "pose hypothesis / review required" if accepted and uncertain else "estimated" if accepted else "no estimate"
        label = f"frame {i}/{len(track['valid'])-1} | {i / manifest['fps']:.2f}s | " + status
        color = (60, 190, 245) if uncertain and accepted else (120, 230, 120) if accepted else (70, 90, 250)
        cv2.putText(tile, label, (8, 286), 0, .48, color, 1)
        writer.write(tile)
        if i in [0, 180, 300, 360, 480, 700]:
            tiles.append(tile)
    writer.release()
    renderer.close()
    cv2.imwrite(str(args.output / "review-grid.jpg"), np.vstack(tiles))
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(temporary), "-c:v", "libx264",
                    "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(args.output / "object-motion.mp4")], check=True)
    temporary.unlink()


if __name__ == "__main__":
    main()
