"""Fit a free rigid object's motion to visible TrackCraft3R 3D correspondences.

Does not infer a hinge or attach the case to a hand. Unprocessed/rejected source
frames remain invalid, including frames between sparse tracking observations.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .book_space import BOOK_SIZE, corners


def kabsch(source, target):
    a, b = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - a).T @ (target - b))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    return rotation, b - rotation @ a


def robust_rigid(source, target, threshold, min_points=20, seed=0):
    source, target = np.asarray(source, float), np.asarray(target, float)
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    selected = np.flatnonzero(finite)
    a, b = source[finite], target[finite]
    if len(a) < min_points:
        return None, "insufficient_visible_points"
    singular = np.linalg.svd(a - a.mean(axis=0), compute_uv=False)
    if singular[0] < 1e-8 or singular[1] / singular[0] < .02:
        return None, "nearly_collinear_reference_points"
    generator = np.random.default_rng(seed)
    best = np.zeros(len(a), bool)
    for _ in range(200):
        sample = generator.choice(len(a), 3, replace=False)
        if np.linalg.norm(np.cross(a[sample[1]] - a[sample[0]], a[sample[2]] - a[sample[0]])) < 1e-10:
            continue
        rotation, translation = kabsch(a[sample], b[sample])
        mask = np.linalg.norm(a @ rotation.T + translation - b, axis=1) < threshold
        if mask.sum() > best.sum():
            best = mask
        if best.mean() > .95:
            break
    if best.sum() < min_points or best.mean() < .6:
        return None, "nonrigid_or_inconsistent_3d_tracks"
    for _ in range(3):
        rotation, translation = kabsch(a[best], b[best])
        errors = np.linalg.norm(a @ rotation.T + translation - b, axis=1)
        new = errors < threshold
        if new.sum() < min_points:
            return None, "insufficient_refined_inliers"
        if np.array_equal(new, best):
            break
        best = new
    rotation, translation = kabsch(a[best], b[best])
    errors = np.linalg.norm(a @ rotation.T + translation - b, axis=1)
    return dict(rotation=rotation, translation=translation, inliers=selected[best],
                median_residual=float(np.median(errors[best])),
                p90_residual=float(np.percentile(errors[best], 90)),
                visible_points=len(a), inlier_fraction=float(best.mean())), None


def frame_manifest(folder):
    manifest = json.loads((folder / "input-manifest.json").read_text())
    if "selected_frames" in manifest:
        mapping = {r["frame"]: (r["source_frame"], r["source_seconds"]) for r in manifest["selected_frames"]}
        return manifest["source_sha256"], manifest["source_frames"], manifest["source_fps"], mapping
    mapping = {r["frame_index"]: (r["source_frame"], r["timestamp_s"]) for r in manifest["frames"]}
    return manifest["sha256"], manifest["source_frames_read"], manifest["fps"], mapping


def window_motion(scene, visibility_threshold=.7):
    """Return independently fitted reference-to-frame motion in metric book space."""
    folder = scene / "trackcraft"
    config = json.loads((folder / "config.json").read_text())
    if not config.get("visibility_exported"):
        raise ValueError("Visibility probabilities are required")
    cache_path = folder / "rigid-fit.json"
    signature = hashlib.sha256(Path(__file__).read_bytes() +
                               (folder / "config.json").read_bytes() + str(visibility_threshold).encode()).hexdigest()
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached["signature"] == signature:
            result = {int(k): v for k, v in cached["frames"].items()}
            for row in result.values():
                if row["estimate"] is not None:
                    for name in ["rotation", "translation"]:
                        row["estimate"][name] = np.array(row["estimate"][name])
            return result
    cameras = np.load(scene / "da3/cameras.npz")
    reference = np.load(folder / "case/pts3d_ref.npy")
    ref = config["ref_frame"]
    ref_visibility = np.load(folder / "case/visibility" / f"{ref:06d}.npy")
    extent = np.linalg.norm(np.percentile(reference, 95, axis=0) - np.percentile(reference, 5, axis=0))
    threshold = max(.0025, .04 * extent)
    results = {}
    for frame in config["tracked_frames"]:
        target = reference if frame == ref else reference + np.load(folder / "case/scene_flow" / f"{frame:06d}.npy")
        visible = np.load(folder / "case/visibility" / f"{frame:06d}.npy")
        selected = (visible >= visibility_threshold) & (ref_visibility >= visibility_threshold)
        rotation = Rotation.from_quat(cameras["cam_quats_xyzw"][frame]).as_matrix()
        points = (target - cameras["cam_trans"][frame]) @ rotation
        homogeneous = points @ cameras["intrinsics"][frame].T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = homogeneous[:, :2] / homogeneous[:, 2:]
        mask = cv2.imread(str(scene / "masks/case" / f"{frame:06d}.png"), 0)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        inside = np.isfinite(uv).all(axis=1) & (points[:, 2] > 0)
        inside &= (uv[:, 0] >= 0) & (uv[:, 0] < mask.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < mask.shape[0])
        indices = np.flatnonzero(inside)
        inside[indices] &= mask[uv[indices, 1].astype(int), uv[indices, 0].astype(int)] > 0
        selected &= inside
        if selected.sum() < max(20, .05 * len(reference)):
            estimate, reason = None, "insufficient_visible_surface"
        else:
            a, b = reference[selected], target[selected]
            # Deterministic subsampling bounds RANSAC cost for dense surfaces.
            sample = np.linspace(0, len(a)-1, min(len(a), 2000), dtype=int)
            estimate, reason = robust_rigid(a[sample], b[sample], threshold)
        results[frame] = dict(estimate=estimate, rejection_reason=reason,
                              qualified_points=int(selected.sum()), threshold_m=float(threshold))
    serializable = {}
    for frame, row in results.items():
        row = dict(row)
        if row["estimate"] is not None:
            estimate = row["estimate"]
            row["estimate"] = {k: v.tolist() if isinstance(v, np.ndarray) else v
                               for k, v in estimate.items() if k != "inliers"}
        serializable[frame] = row
    cache_path.write_text(json.dumps(dict(signature=signature, frames=serializable)) + "\n")
    return results


def main():
    import mujoco
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--taa", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--geometry-report", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visibility", type=float, default=.7)
    args = parser.parse_args()
    folder = args.taa / "trackcraft"
    config = json.loads((folder / "config.json").read_text())
    if not config.get("visibility_exported"):
        raise ValueError("Require the visibility-preserving TAA wrapper")
    ref = config["ref_frame"]
    if ref != 0:
        raise ValueError("This adapter requires frame-0 anchoring; arbitrary window stitching is not verified")
    sha, count, fps, mapping = frame_manifest(args.taa)
    args.output.mkdir(parents=True, exist_ok=False)
    reference = np.load(folder / "case/pts3d_ref.npy")
    visibility_ref = np.load(folder / "case/visibility" / f"{ref:06d}.npy")
    extent = float(np.linalg.norm(np.percentile(reference, 95, axis=0) - np.percentile(reference, 5, axis=0)))
    threshold = .04 * extent
    camera_data = np.load(args.taa / "da3/cameras.npz")
    camera_index = {int(f): i for i, f in enumerate(camera_data["frame_indices"])}
    c2w = Rotation.from_quat(camera_data["cam_quats_xyzw"][camera_index[ref]]).as_matrix()
    c2w_t = camera_data["cam_trans"][camera_index[ref]]
    k = camera_data["intrinsics"][camera_index[ref]]
    reviewed = {r["source_frame"]: r for r in json.loads(args.cameras.read_text())}
    _, rvec, tvec = cv2.solvePnP(corners(BOOK_SIZE[2]), np.asarray(reviewed[0]["book_quad"]), k, None)
    r_camera = cv2.Rodrigues(rvec)[0]
    metric_scale = json.loads(args.geometry_report.read_text())["metric_scale"]
    align_r = r_camera.T @ c2w.T
    align_t = -metric_scale * align_r @ c2w_t - r_camera.T @ tvec.ravel()
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "case")
    initial_t = model.body_pos[body]
    initial_r = Rotation.from_quat(model.body_quat[body][[1, 2, 3, 0]]).as_matrix()
    translations = np.full((count, 3), np.nan)
    rotations = np.full((count, 3, 3), np.nan)
    valid = np.zeros(count, bool)
    rows = []
    for frame in config["tracked_frames"]:
        source_frame, seconds = mapping[frame]
        visible = np.load(folder / "case/visibility" / f"{frame:06d}.npy")
        target = reference if frame == ref else reference + np.load(folder / "case/scene_flow" / f"{frame:06d}.npy")
        selected = (visible >= args.visibility) & (visibility_ref >= args.visibility)
        # A predicted visible point must project onto the visible object mask.
        ci = camera_index[frame]
        camera_r = Rotation.from_quat(camera_data["cam_quats_xyzw"][ci]).as_matrix()
        camera_points = (target - camera_data["cam_trans"][ci]) @ camera_r
        homogeneous = camera_points @ camera_data["intrinsics"][ci].T
        uv = homogeneous[:, :2] / homogeneous[:, 2:]
        mask = cv2.imread(str(args.taa / "masks/case" / f"{frame:06d}.png"), 0)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        finite_uv = np.isfinite(uv).all(axis=1)
        inside = finite_uv & (camera_points[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < mask.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < mask.shape[0])
        indices = np.flatnonzero(inside)
        inside[indices] &= mask[uv[indices, 1].astype(int), uv[indices, 0].astype(int)] > 0
        selected &= inside
        estimate, reason = robust_rigid(reference[selected], target[selected], threshold)
        row = dict(frame=frame, source_frame=source_frame, timestamp_s=seconds,
                   valid=estimate is not None, rejection_reason=reason,
                   qualified_points=int(selected.sum()))
        if estimate is not None:
            delta_r = align_r @ estimate["rotation"] @ align_r.T
            delta_t = metric_scale * align_r @ estimate["translation"] + align_t - delta_r @ align_t
            rotations[source_frame] = delta_r @ initial_r
            translations[source_frame] = delta_r @ initial_t + delta_t
            valid[source_frame] = True
            row.update(inlier_fraction=estimate["inlier_fraction"],
                       median_rigid_residual_m=metric_scale * estimate["median_residual"],
                       p90_rigid_residual_m=metric_scale * estimate["p90_residual"],
                       translation_world_m=translations[source_frame].tolist(),
                       rotation_world=rotations[source_frame].tolist())
        rows.append(row)
    np.savez_compressed(args.output / "object_track.npz", source_frame=np.arange(count),
                        timestamp_s=np.arange(count) / fps, valid=valid,
                        rotation_world=rotations, translation_world_m=translations)
    (args.output / "poses.json").write_text(json.dumps(rows, indent=2) + "\n")
    report = dict(source_sha256=sha, total_source_frames=count,
                  tracked_frames=len(rows), accepted_frames=int(valid.sum()),
                  visibility_threshold=args.visibility,
                  rigid_threshold_m=metric_scale * threshold,
                  rows=rows, method="Visibility/mask-filtered TrackCraft3R + RANSAC rigid SE(3)",
                  limitations=["Predicted 3D tracks and visibility are not ground truth",
                               "Rigid residual tests internal consistency, not absolute pose accuracy",
                               "Unprocessed and rejected frames remain NaN; no interpolation",
                               "Frame 0 is the existing case calibration anchor"])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Accepted {valid.sum()}/{len(rows)} tracked frames; total source frames {count}")


if __name__ == "__main__":
    main()
