"""Estimate case SE(3) from visible features; reject unsupported/ambiguous poses.

Uses the TAA case mesh as geometry, SAM3 masks to exclude the hand/background,
forward/backward optical flow, feature-map relocalisation and RANSAC PnP.
New feature anchors are estimates from accepted poses, not measured 3D points.
No interpolation or attachment to the hand is used.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .book_space import BOOK_SIZE, corners
from .track_utils import intervals


def detect_features(gray, mask, sift):
    interior = cv2.erode(mask, np.ones((5, 5), np.uint8))
    x, y, w, h = cv2.boundingRect(interior)
    if not w or not h:
        return np.empty((0, 2), np.float32), None
    crop = cv2.resize(gray[y:y+h, x:x+w], None, fx=3, fy=3)
    roi = cv2.resize(interior[y:y+h, x:x+w], None, fx=3, fy=3,
                     interpolation=cv2.INTER_NEAREST)
    keypoints, descriptors = sift.detectAndCompute(crop, roi)
    points = np.array([(k.pt[0] / 3 + x, k.pt[1] / 3 + y) for k in keypoints],
                      np.float32).reshape(-1, 2)
    # SIFT can return multiple orientations at the same image position. They
    # are one observation, and must not inflate inlier counts or confidence.
    selected = []
    for index in sorted(range(len(keypoints)), key=lambda j: -keypoints[j].response):
        if not selected or np.linalg.norm(points[selected] - points[index], axis=1).min() > 2.:
            selected.append(index)
    return points[selected], descriptors[selected] if descriptors is not None else None


def flow_points(previous, current, points, mask):
    if len(points) == 0:
        return points, np.zeros(0, bool)
    p = np.float32(points).reshape(-1, 1, 2)
    options = dict(winSize=(19, 19), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    q, status, error = cv2.calcOpticalFlowPyrLK(previous, current, p, None, **options)
    back, status_back, _ = cv2.calcOpticalFlowPyrLK(current, previous, q, None, **options)
    q = q.reshape(-1, 2)
    valid = (status.ravel() > 0) & (status_back.ravel() > 0)
    valid &= np.linalg.norm(back.reshape(-1, 2) - points, axis=1) < .7
    valid &= error.ravel() < 20
    ix = np.rint(q[:, 0]).astype(int).clip(0, mask.shape[1] - 1)
    iy = np.rint(q[:, 1]).astype(int).clip(0, mask.shape[0] - 1)
    valid &= (q[:, 0] >= 0) & (q[:, 0] < mask.shape[1])
    valid &= (q[:, 1] >= 0) & (q[:, 1] < mask.shape[0])
    valid &= cv2.erode(mask, np.ones((3, 3), np.uint8))[iy, ix] > 0
    return q, valid


def estimate_pose(xyz, pixels, intrinsics):
    if len(xyz) < 8:
        return None, "fewer_than_8_correspondences"
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.float64(xyz), np.float64(pixels), intrinsics, None,
            iterationsCount=200, reprojectionError=2., confidence=.999,
            flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        return None, "degenerate_pnp_sample"
    if not ok or inliers is None or len(inliers) < 8:
        return None, "insufficient_ransac_inliers"
    inliers = inliers.ravel()
    points, targets = np.float64(xyz[inliers]), np.float64(pixels[inliers])
    rvec, tvec = cv2.solvePnPRefineLM(points, targets, intrinsics, None, rvec, tvec)
    image, jac = cv2.projectPoints(points, rvec, tvec, intrinsics, None)
    errors = np.linalg.norm(image.reshape(-1, 2) - targets, axis=1)
    hull_area = float(cv2.contourArea(cv2.convexHull(targets.astype(np.float32))))
    covariance = np.linalg.pinv(jac[:, :6].T @ jac[:, :6], rcond=1e-12)
    std = np.sqrt(np.maximum(np.diag(covariance), 0))  # 1 px observation noise
    if tvec[2, 0] <= .05 or tvec[2, 0] > 2:
        return None, "invalid_camera_depth"
    if np.percentile(errors, 90) > 2. or hull_area < 30:
        return None, "poor_fit_or_concentrated_features"
    if max(std[:3]) > .35 or max(std[3:]) > .025:
        return None, "weak_pose_observability"
    return dict(rvec=rvec.ravel(), tvec=tvec.ravel(), inliers=inliers,
                residual_px=float(np.median(errors)), hull_area_px2=hull_area,
                rotation_std_rad=std[:3], translation_std_m=std[3:]), None


class Geometry:
    def __init__(self, scene):
        self.model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
        self.data = mujoco.MjData(self.model)
        self.body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "case")
        self.geomgroup = np.array([1, 0, 0, 0, 0, 0], np.uint8)
        # Use compiled geometry so original and neural mesh candidates share
        # exactly the same body-coordinate convention as MuJoCo.
        mujoco.mj_forward(self.model, self.data)
        geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "taa_case_appearance")
        if geom < 0:
            geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "case_solid")
        mesh = self.model.geom_dataid[geom]
        first, count = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
        vertices = self.model.mesh_vert[first:first+count]
        world = vertices @ self.data.geom_xmat[geom].reshape(3, 3).T + self.data.geom_xpos[geom]
        points = (world - self.data.xpos[self.body]) @ self.data.xmat[self.body].reshape(3, 3)
        self.surface = points[np.linspace(0, len(points) - 1, 1600, dtype=int)].astype(float)

    def mask_consistency(self, rvec, tvec, intrinsics, mask):
        camera_points = self.surface @ cv2.Rodrigues(rvec)[0].T + tvec
        if camera_points[:, 2].min() <= 0:
            return 0., 0.
        uv = cv2.projectPoints(self.surface, rvec, tvec, intrinsics, None)[0].reshape(-1, 2)
        if not np.isfinite(uv).all() or np.max(np.abs(uv)) > 10000:
            return 0., 0.
        silhouette = np.zeros_like(mask)
        cv2.fillConvexPoly(silhouette, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 255)
        supported = cv2.dilate(silhouette, np.ones((7, 7), np.uint8)) > 0
        observed = mask > 0
        observed_pixels = max(1, int(observed.sum()))
        return float((supported & observed).sum() / observed_pixels), float((silhouette > 0).sum() / observed_pixels)

    def anchors(self, pixels, rotation_world, translation_world, camera_r, camera_t, k):
        quaternion = Rotation.from_matrix(rotation_world).as_quat()[[3, 0, 1, 2]]
        self.data.qpos[:] = np.r_[translation_world, quaternion]
        mujoco.mj_forward(self.model, self.data)
        origin = -camera_r.T @ camera_t
        points, selected = [], []
        for index, pixel in enumerate(pixels):
            direction = camera_r.T @ np.linalg.solve(k, np.r_[pixel, 1.])
            direction /= np.linalg.norm(direction)
            geom = np.array([-1], np.int32)
            distance = mujoco.mj_ray(self.model, self.data, origin, direction,
                                     self.geomgroup, True, -1, geom)
            if distance > 0 and self.model.geom_bodyid[geom[0]] == self.body:
                point = origin + direction * distance
                points.append(rotation_world.T @ (point - translation_world))
                selected.append(index)
        return np.asarray(points).reshape(-1, 3), np.asarray(selected, int)


def run(args):
    manifest = json.loads((args.taa / "input-manifest.json").read_text())
    camera_report = json.loads((args.cameras.parent / "report.json").read_text())
    if camera_report["source_sha256"] != manifest["sha256"]:
        raise ValueError("Camera and object inputs must use the same source video")
    geometry_manifest = json.loads((args.taa_check / "input-manifest.json").read_text())
    if geometry_manifest.get("source_sha256", geometry_manifest.get("sha256")) != manifest["sha256"]:
        raise ValueError("Case geometry inputs must use the same source video")
    if manifest["stride"] != 1:
        raise ValueError("Use full-frame inputs, not subsampled frames")
    cameras = json.loads(args.cameras.read_text())
    if {c["source_frame"] for c in cameras} != set(range(len(manifest["frames"]))):
        raise ValueError("Camera coverage must match every input frame")
    cameras = {c["source_frame"]: c for c in cameras}
    args.output.mkdir(parents=True, exist_ok=False)
    k = np.array(json.loads((args.taa_check / "sam3d_scaled/case/depth_scale.json").read_text())["intrinsics"])
    geometry = Geometry(args.scene)
    initial_world_t = geometry.model.body_pos[geometry.body].copy()
    initial_world_r = Rotation.from_quat(geometry.model.body_quat[geometry.body][[1, 2, 3, 0]]).as_matrix()
    sift = cv2.SIFT_create(nfeatures=400, contrastThreshold=.008, edgeThreshold=15)
    matcher = cv2.BFMatcher()
    bank_xyz, bank_descriptors = [], []
    active_ids, active_pixels = np.array([], int), np.empty((0, 2), np.float32)
    previous_gray = None
    last_world = None
    rows, rotations, translations, valid = [], [], [], []
    debug_points = []
    started = time.monotonic()
    for record in manifest["frames"]:
        i = record["source_frame"]
        image = cv2.imread(str(args.taa / "frames" / f"{i:06d}.jpg"))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.imread(str(args.taa / "masks/case" / f"{i:06d}.png"), 0)
        if mask is None:
            raise ValueError(f"Missing object mask {i}")
        quad = np.asarray(cameras[i]["book_quad"], np.float64)
        ok, camera_rvec, camera_tvec = cv2.solvePnP(corners(BOOK_SIZE[2]), quad, k, None)
        if not ok:
            raise RuntimeError(f"Camera calibration failed at {i}")
        camera_r = cv2.Rodrigues(camera_rvec)[0]
        camera_t = camera_tvec.ravel()
        if previous_gray is not None:
            moved, keep = flow_points(previous_gray, gray, active_pixels, mask)
            active_pixels, active_ids = moved[keep], active_ids[keep]
        # Match to the accumulated appearance map to recover dropped features.
        detected, descriptors = detect_features(gray, mask, sift)
        if bank_descriptors and descriptors is not None:
            matches = matcher.knnMatch(np.float32(bank_descriptors), descriptors, k=2)
            chosen = {}
            for index, pair in enumerate(matches):
                if len(pair) == 2 and pair[0].distance < .7 * pair[1].distance and pair[0].distance < 200:
                    match = pair[0]
                    if match.trainIdx not in chosen or match.distance < chosen[match.trainIdx][1]:
                        chosen[match.trainIdx] = (index, match.distance)
            for target, (identifier, _) in chosen.items():
                if identifier not in active_ids:
                    active_ids = np.r_[active_ids, identifier]
                    active_pixels = np.vstack([active_pixels, detected[target]])
        estimate = None
        reason = None
        anchor_frame = i == 0
        if anchor_frame:
            world_r, world_t = initial_world_r, initial_world_t
            reason = "first_frame_calibration_anchor"
        else:
            xyz = np.asarray(bank_xyz)[active_ids] if len(active_ids) else np.empty((0, 3))
            estimate, reason = estimate_pose(xyz, active_pixels, k)
            if estimate is not None:
                coverage, area_ratio = geometry.mask_consistency(estimate["rvec"], estimate["tvec"], k, mask)
                estimate.update(visible_mask_coverage=coverage, projected_to_visible_area_ratio=area_ratio)
                if coverage < .85 or not .7 < area_ratio < 6:
                    estimate, reason = None, "mesh_mask_disagreement"
            if estimate is not None:
                camera_object_r = cv2.Rodrigues(estimate["rvec"])[0]
                world_r = camera_r.T @ camera_object_r
                world_t = camera_r.T @ (estimate["tvec"] - camera_t)
                if last_world is not None and i - last_world[0] <= 10:
                    gap = i - last_world[0]
                    jump_t = np.linalg.norm(world_t - last_world[2])
                    jump_r = Rotation.from_matrix(world_r @ last_world[1].T).magnitude()
                    if jump_t > min(.03, .015 * gap) or jump_r > min(.6, .3 * gap):
                        estimate, reason = None, "implausible_interframe_jump"
                if estimate is not None:
                    keep = estimate["inliers"]
                    active_ids, active_pixels = active_ids[keep], active_pixels[keep]
        accepted = anchor_frame or estimate is not None
        row = dict(source_frame=i, timestamp_s=record["timestamp_s"], valid=accepted,
                   status="calibration_anchor" if anchor_frame else "estimated" if accepted else "unobserved",
                   rejection_reason=reason if not accepted else None,
                   correspondences=int(len(active_ids)))
        if accepted:
            last_world = (i, world_r.copy(), world_t.copy())
            row.update(rotation_world=world_r.tolist(), translation_world_m=world_t.tolist())
            if estimate is not None:
                row.update(inliers=len(estimate["inliers"]), reprojection_median_px=estimate["residual_px"],
                           feature_hull_area_px2=estimate["hull_area_px2"],
                           visible_mask_coverage=estimate["visible_mask_coverage"],
                           projected_to_visible_area_ratio=estimate["projected_to_visible_area_ratio"],
                           conditional_rotation_std_rad=estimate["rotation_std_rad"].tolist(),
                           conditional_translation_std_m=estimate["translation_std_m"].tolist())
            # Add only features on the mesh interior, at low-error accepted poses.
            # These anchors inherit pose/model error: covariance does not cover drift.
            if descriptors is not None and (anchor_frame or (i % 5 == 0 and estimate["residual_px"] < .8)):
                if len(active_pixels):
                    distance = np.linalg.norm(detected[:, None] - active_pixels[None], axis=2).min(axis=1)
                    selected = np.flatnonzero(distance > 3.)
                else:
                    selected = np.arange(len(detected))
                mapped, selection = geometry.anchors(detected[selected], world_r, world_t,
                                                     camera_r, camera_t, k)
                selected = selected[selection]
                for point, pixel, desc in zip(mapped, detected[selected], descriptors[selected]):
                    identifier = len(bank_xyz)
                    bank_xyz.append(point)
                    bank_descriptors.append(desc)
                    active_ids = np.r_[active_ids, identifier]
                    active_pixels = np.vstack([active_pixels, pixel])
            rotations.append(world_r)
            translations.append(world_t)
        else:
            rotations.append(np.full((3, 3), np.nan))
            translations.append(np.full(3, np.nan))
        valid.append(accepted)
        rows.append(row)
        debug_points.append(active_pixels.copy() if accepted else np.empty((0, 2)))
        previous_gray = gray
        if i % 60 == 0:
            print(f"Frame {i}: {row['status']}, features={len(active_ids)}, map={len(bank_xyz)}", flush=True)
    times = np.array([r["timestamp_s"] for r in rows])
    valid = np.array(valid)
    report = dict(source_sha256=manifest["sha256"], processed_frames=len(rows),
                  accepted_frames=int(valid.sum()), calibration_anchor_frames=1,
                  estimated_frames=int(valid.sum()) - 1, coverage=float(valid.mean()),
                  missing_intervals=intervals(np.flatnonzero(~valid), times),
                  elapsed_seconds=time.monotonic() - started,
                  appearance_map_points=len(bank_xyz),
                  method="TAA/SAM3 masks + mesh-ray feature anchors + SIFT/LK + RANSAC PnP",
                  scope="Monocular object pose estimates, not measured states or physical robot execution",
                  limitations=["Metric scale and mesh depend on the estimated book dimensions",
                               "First frame is a calibration anchor, not an independently tracked pose",
                               "New feature anchors inherit previous pose/model error and can drift",
                               "Conditional covariance excludes mesh, camera, association and accumulated map error",
                               "Hidden or weakly constrained poses remain NaN; no interpolation"])
    (args.output / "poses.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output / "object_track.npz", source_frame=np.arange(len(rows)),
                        timestamp_s=times, valid=valid, rotation_world=np.asarray(rotations),
                        translation_world_m=np.asarray(translations))
    # A compact point archive for the visual review, without Python pickle arrays.
    padded = np.full((len(rows), max(map(len, debug_points), default=0), 2), np.nan, np.float32)
    for i, points in enumerate(debug_points):
        padded[i, :len(points)] = points
    np.save(args.output / "feature_pixels.npy", padded)
    provenance = dict(scene_sha256=hashlib.sha256(args.scene.read_bytes()).hexdigest(),
                      cameras_sha256=hashlib.sha256(args.cameras.read_bytes()).hexdigest(),
                      arguments={k: str(v) for k, v in vars(args).items()})
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--taa", type=Path, required=True)
    parser.add_argument("--taa-check", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
