"""Join overlapping rigid-motion windows, checking overlaps and visible masks.

This produces an estimated trajectory. No interpolation, hand attachment,
gravity rollout or robot control is performed here.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .track_utils import intervals
from .rigid_tracks import window_motion
from .image_pose_refine import refine_pose, relocalize


def anchored_pose(target, anchor_delta, anchor_rotation, anchor_translation):
    rotation = target["rotation"] @ anchor_delta["rotation"].T
    translation = target["translation"] - rotation @ anchor_delta["translation"]
    return rotation @ anchor_rotation, rotation @ anchor_translation + translation


def mask_agreement(surface, rotation, translation, frame, cameras, mask_path):
    camera_r = Rotation.from_quat(cameras["cam_quats_xyzw"][frame]).as_matrix()
    points = (surface @ rotation.T + translation - cameras["cam_trans"][frame]) @ camera_r
    if points[:, 2].min() <= 0:
        return 0., 0.
    homogeneous = points @ cameras["intrinsics"][frame].T
    uv = homogeneous[:, :2] / homogeneous[:, 2:]
    if not np.isfinite(uv).all() or np.max(np.abs(uv)) > 10000:
        return 0., 0.
    mask = cv2.imread(str(mask_path), 0)
    silhouette = np.zeros_like(mask)
    cv2.fillConvexPoly(silhouette, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 255)
    observed = mask > 0
    supported = cv2.dilate(silhouette, np.ones((9, 9), np.uint8)) > 0
    count = max(1, observed.sum())
    return float((observed & supported).sum() / count), float((silhouette > 0).sum() / count)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--scene-data", type=Path, required=True)
    parser.add_argument("--initial-pose", type=Path, required=True)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--occlusion-masks", type=Path)
    parser.add_argument("--hand-prior", type=Path)
    parser.add_argument("--hand-anchor-frame", type=int, default=320)
    parser.add_argument("--backward-pass", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.scene_data / "input-manifest.json").read_text())
    count = manifest["source_frames_read"]
    camera_data = np.load(args.scene_data / "da3/cameras.npz")
    times = np.array([r["timestamp_s"] for r in manifest["frames"]])
    translations = np.full((count, 3), np.nan)
    rotations = np.full((count, 3, 3), np.nan)
    valid = np.zeros(count, bool)
    components = np.full(count, -1, int)
    components[0] = 0
    hand = np.load(args.hand_prior) if args.hand_prior is not None else None
    if args.hand_prior is not None:
        hand_manifest = json.loads((args.hand_prior.parent / "manifest.json").read_text())
        if hand_manifest["source_sha256"] != manifest["sha256"]:
            raise ValueError("Hand prior must come from this same source recording")
    depth_file = args.scene_data / "case_depth_camera_m.npy"
    surface_depth = np.load(depth_file) if depth_file.exists() else None
    reinitializations = []
    initial = json.loads(args.initial_pose.read_text())
    rotations[0], translations[0], valid[0] = initial["rotation"], initial["translation"], True
    surface = np.load(args.surface)
    rows = [dict(source_frame=i, timestamp_s=float(times[i]), valid=False,
                 status="unobserved", rejection_reason="no_accepted_window") for i in range(count)]
    rows[0].update(valid=True, status="calibration_anchor", rejection_reason=None)
    audits = []
    windows = [Path(p) for p in json.loads(args.windows.read_text())]
    worklist = windows + list(reversed(windows)) if args.backward_pass else windows
    args.output.mkdir(parents=True, exist_ok=False)
    for index, scene in enumerate(worklist):
        backward = index >= len(windows)
        if not (scene / "trackcraft/config.json").exists():
            audits.append(dict(window=scene.name, status="missing_tracking_output"))
            continue
        try:
            complete = json.loads((scene / "trackcraft/config.json").read_text()).get("visibility_exported")
        except json.JSONDecodeError:
            complete = False
        if not complete:
            audits.append(dict(window=scene.name, status="incomplete_tracking_output"))
            continue
        wm = json.loads((scene / "input-manifest.json").read_text())
        if wm["sha256"] != manifest["sha256"]:
            raise ValueError("Window source checksum mismatch")
        mapping = {r["frame_index"]: r["source_frame"] for r in wm["frames"]}
        motions = window_motion(scene)
        anchors = [j for j, m in motions.items() if m["estimate"] is not None and valid[mapping[j]]]
        audit = dict(window=scene.name, window_id=str(scene.resolve()), backward=backward,
                     fitted_frames=sum(m["estimate"] is not None for m in motions.values()),
                     tracked_frames=len(motions))
        if not anchors:
            if backward:
                audit["status"] = "no_connected_pose_anchor"
                audits.append(audit)
                continue
            frame = mapping[min(mapping)]
            reference = args.hand_anchor_frame
            hypothesis = None
            if (hand is not None and surface_depth is not None and args.occlusion_masks is not None
                    and valid[reference] and hand["valid"][reference] and hand["valid"][frame]):
                cr = Rotation.from_quat(camera_data["cam_quats_xyzw"][frame]).as_matrix()
                hr = Rotation.from_quat(camera_data["cam_quats_xyzw"][reference]).as_matrix() @ Rotation.from_matrix(hand["global_orient"][reference]).as_matrix()
                ht = cr @ Rotation.from_matrix(hand["global_orient"][frame]).as_matrix()
                prior = ht @ hr.T @ rotations[reference]
                case = cv2.imread(str(args.scene_data / "masks/case" / f"{frame:06d}.png"), 0)
                occluder = cv2.imread(str(args.occlusion_masks / f"{frame:06d}.png"), 0)
                hypothesis = relocalize(surface, prior, cr, camera_data["cam_trans"][frame],
                                        camera_data["intrinsics"][frame], case, occluder, surface_depth[frame])
                if hypothesis is not None:
                    cov, ratio = mask_agreement(surface, hypothesis["rotation"], hypothesis["translation"], frame,
                                                camera_data, args.scene_data / "masks/case" / f"{frame:06d}.png")
                    if cov < .85 or not .7 < ratio < 4 or not -.005 < hypothesis["translation"][2] < .5:
                        hypothesis = None
            if hypothesis is None:
                audit["status"] = "no_connected_pose_anchor"
                audits.append(audit)
                continue
            rotations[frame], translations[frame], valid[frame] = hypothesis["rotation"], hypothesis["translation"], True
            components[frame] = len(reinitializations) + 1
            reinitializations.append(dict(source_frame=frame, component=int(components[frame]),
                                          hand_anchor_frame=reference, orientation_confidence="uncertain",
                                          method="Visible case + predicted depth + observed hand-orientation prior",
                                          median_edge_error_px=hypothesis["median_visible_edge_error_px"]))
            rows[frame].update(valid=True, status="relocalized_hypothesis", rejection_reason=None,
                               requires_review=True, component=int(components[frame]))
            anchors = [j for j, m in motions.items() if m["estimate"] is not None and valid[mapping[j]]]
            if not anchors:
                audit["status"] = "relocalization_without_rigid_support"
                audits.append(audit)
                continue
        # Prefer an early overlap anchor; frame 0 remains the sole absolute anchor.
        anchor = max(anchors) if backward else min(anchors)
        source_anchor = mapping[anchor]
        delta_anchor = motions[anchor]["estimate"]
        proposals = {}
        differences = []
        for j, motion in motions.items():
            if motion["estimate"] is None:
                continue
            r, t = anchored_pose(motion["estimate"], delta_anchor,
                                  rotations[source_anchor], translations[source_anchor])
            proposals[j] = (r, t)
            source_frame = mapping[j]
            if valid[source_frame]:
                difference_r = Rotation.from_matrix(r @ rotations[source_frame].T).magnitude()
                differences.append((np.linalg.norm(t - translations[source_frame]), difference_r))
        error_t, error_r = np.median(differences, axis=0)
        audit.update(anchor_source_frame=source_anchor, overlap_position_m=float(error_t),
                     overlap_rotation_rad=float(error_r))
        if error_t > .015 or error_r > .35:
            audit["status"] = "overlap_disagreement"
            audits.append(audit)
            continue
        accepted = 0
        for j, (r, t) in proposals.items():
            frame = mapping[j]
            if valid[frame]:
                continue
            coverage, ratio = mask_agreement(surface, r, t, frame, camera_data,
                                              args.scene_data / "masks/case" / f"{frame:06d}.png")
            refinement = None
            if coverage < .95 and args.occlusion_masks is not None:
                occluder = cv2.imread(str(args.occlusion_masks / f"{frame:06d}.png"), 0)
                case = cv2.imread(str(args.scene_data / "masks/case" / f"{frame:06d}.png"), 0)
                if occluder is not None:
                    camera_r = Rotation.from_quat(camera_data["cam_quats_xyzw"][frame]).as_matrix()
                    candidate = refine_pose(surface, r, t, camera_r, camera_data["cam_trans"][frame],
                                                   camera_data["intrinsics"][frame], case, occluder)
                    if candidate is not None:
                        new_coverage, new_ratio = mask_agreement(surface, candidate["rotation"], candidate["translation"], frame, camera_data,
                                                               args.scene_data / "masks/case" / f"{frame:06d}.png")
                        if new_coverage > coverage:
                            refinement = candidate
                            r, t, coverage, ratio = candidate["rotation"], candidate["translation"], new_coverage, new_ratio
            if coverage < .8 or not .6 < ratio < 6 or not -.005 < t[2] < .5:
                rows[frame].update(rejection_reason="mesh_mask_or_depth_disagreement",
                                   visible_mask_coverage=coverage, projected_to_visible_area_ratio=ratio)
                continue
            previous = frame - 1
            if previous >= 0 and valid[previous]:
                jump_t = np.linalg.norm(t - translations[previous])
                jump_r = Rotation.from_matrix(r @ rotations[previous].T).magnitude()
                if jump_t > .03 or jump_r > .35:
                    rows[frame]["rejection_reason"] = "implausible_interframe_jump"
                    continue
            rotations[frame], translations[frame], valid[frame] = r, t, True
            components[frame] = components[source_anchor]
            estimate = motions[j]["estimate"]
            rows[frame].update(valid=True, status="estimated", rejection_reason=None,
                               window=scene.name, proposal_median_rigid_residual_m=estimate["median_residual"],
                               inlier_fraction=estimate["inlier_fraction"],
                               visible_mask_coverage=coverage, projected_to_visible_area_ratio=ratio)
            rows[frame].update(component=int(components[frame]), requires_review=bool(components[frame] > 0))
            if refinement is not None:
                rows[frame].update(image_refined=True,
                                   camera_translation_correction_m=refinement["camera_correction_m"].tolist(),
                                   image_rotation_correction_rad=refinement["rotation_correction_rad"].tolist(),
                                   median_visible_edge_error_px=refinement["median_visible_edge_error_px"])
            accepted += 1
        audit.update(status="joined", new_accepted_frames=accepted)
        audits.append(audit)
        print(f"Window {index+1}/{len(worklist)}: {audit['status']}, new poses {accepted}, total {valid.sum()}/{count}", flush=True)
    for i in np.flatnonzero(valid):
        rotations[i] = Rotation.from_matrix(rotations[i]).as_matrix()
        rows[i].update(rotation_world=rotations[i].tolist(), translation_world_m=translations[i].tolist())
    np.savez_compressed(args.output / "object_track.npz", source_frame=np.arange(count), timestamp_s=times,
                        valid=valid, rotation_world=rotations, translation_world_m=translations,
                        component=components, requires_review=components > 0)
    (args.output / "poses.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "windows.json").write_text(json.dumps(audits, indent=2) + "\n")
    report = dict(source_sha256=manifest["sha256"], processed_frames=count,
                  accepted_frames=int(valid.sum()), calibration_anchor_frames=1, coverage=float(valid.mean()),
                  missing_intervals=intervals(np.flatnonzero(~valid), times), windows=len(windows),
                  completed_tracking_windows=len({r.get("window_id", r["window"]) for r in audits if r["status"] not in ["missing_tracking_output", "incomplete_tracking_output"]}),
                  joined_windows=len({r["window_id"] for r in audits if r["status"] == "joined"}),
                  backward_pass=args.backward_pass,
                  image_refined_frames=sum(bool(r.get("image_refined")) for r in rows),
                  continuously_connected_frames=int((valid & (components == 0)).sum()),
                  relocalized_candidate_frames=int((valid & (components > 0)).sum()),
                  reinitializations=reinitializations, ready_for_robot_control=False,
                  method="Overlapping TrackCraft3R windows + visibility/mask filtering + rigid RANSAC + overlap checks",
                  limitations=["Monocular geometry, book dimensions and initial mesh pose are estimated",
                               "Rigid residuals describe the 3D proposal before image-space correction",
                               "Internal consistency and mask agreement do not measure true 6D accuracy",
                               "Window composition can accumulate drift",
                               "Relocalized components use an unverified hand-object orientation prior; they are hypotheses",
                               "Rejected/unsupported poses remain NaN; no hand attachment or interpolation",
                               "Pose replay is kinematic and does not establish physical robot manipulation"])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Accepted {valid.sum()}/{count} poses")


if __name__ == "__main__":
    main()
