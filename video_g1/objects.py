"""Refit measured rigid-case candidates to actual masks in the new camera frame.

Silhouette fitting cannot identify a symmetric object's face or hidden rotation.
The old trajectory supplies orientation hypotheses only; review flags remain.
"""

import argparse
import json
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from reconstruction.image_pose_refine import refine_pose


def hand_occluder(pixels, shape):
    """Rasterize a conservative 2D bone/palm proxy, not an observed hand segmentation."""
    mask = np.zeros(shape, np.uint8)
    if not np.isfinite(pixels).all():
        return mask
    p = np.rint(pixels).astype(np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(p[[0, 5, 9, 13, 17]]), 255)
    for finger in range(5):
        chain = [0] + list(range(1 + 4 * finger, 5 + 4 * finger))
        for a, b in pairwise(chain):
            cv2.line(mask, tuple(p[a]), tuple(p[b]), 255, 12)
    return mask


def box_surface(size):
    """Sample the six measured box faces for silhouette projection."""
    half = np.asarray(size) / 2
    points = []
    for axis in range(3):
        other = [a for a in range(3) if a != axis]
        for side in [-1, 1]:
            for u in np.linspace(-1, 1, 9):
                for v in np.linspace(-1, 1, 9):
                    p = np.zeros(3)
                    p[axis] = side * half[axis]
                    p[other] = [u * half[other[0]], v * half[other[1]]]
                    points.append(p)
    return np.array(points)


def main():
    """Write mask-supported candidates and explicit gaps without filling observations."""
    p = argparse.ArgumentParser(__doc__)
    for name in ["hands", "cameras", "prior", "case-masks", "scene", "config", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    h = np.load(a.hands / "hand_track.npz")
    c = np.load(a.cameras / "cameras.npz")
    old = np.load(a.prior)
    cfg = json.loads(a.config.read_text())
    scene = json.loads((a.scene / "reconstruction.json").read_text())
    n = len(h["valid"])
    valid = np.zeros(n, bool)
    positions = np.full((n, 3), np.nan)
    rotations = np.full((n, 3, 3), np.nan)
    errors = np.full(n, np.nan)
    centre = np.array(scene["initial_case_position_m"])
    rotation = Rotation.from_euler("z", scene["initial_case_yaw_rad"]).as_matrix()
    surface = box_surface(cfg["case_size_m"])
    previous_prior = None
    rows = []
    for i in range(n):
        mask = cv2.imread(str(a.case_masks / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Missing mask {i}")
        occluder = hand_occluder(h["joints_image_px"][i], mask.shape)
        camera_r = c["rotation_world_to_camera"][i].T
        camera_t = -camera_r @ c["translation_world_to_camera_m"][i]
        proposal = rotation
        if old["valid"][i] and previous_prior is not None:
            proposal = old["rotation_world"][i] @ previous_prior.T @ rotation
        result = refine_pose(surface, proposal, centre, camera_r, camera_t, c["intrinsics"], mask, occluder)
        if result is not None:
            if old["valid"][i]:
                previous_prior = old["rotation_world"][i]
            centre = result["translation"]
            rotation = result["rotation"]
            valid[i] = True
            positions[i] = centre
            rotations[i] = rotation
            errors[i] = result["median_visible_edge_error_px"]
        rows.append(
            {
                "source_frame": i,
                "valid": bool(valid[i]),
                "requires_review": True,
                "visible_edge_error_px": float(errors[i]) if valid[i] else None,
            }
        )
    np.savez_compressed(
        a.output / "object-candidates.npz",
        source_frame=h["source_frame"],
        valid=valid,
        requires_review=np.ones(n, bool),
        original_requires_review=old["requires_review"],
        translation_world_m=positions,
        rotation_world=rotations,
        visible_edge_error_px=errors,
    )
    (a.output / "frames.json").write_text(json.dumps(rows, indent=2) + "\n")
    (a.output / "report.json").write_text(
        json.dumps(
            {
                "method": "Measured-box silhouette registration with conservative hand occlusion proxy",
                "source_frames": n,
                "valid_candidates": int(valid.sum()),
                "median_visible_edge_error_px": float(np.nanmedian(errors)),
                "rotation_prior": "Previous candidate with legacy orientation increments; not independent rotation evidence",
                "occlusion": "Hand-keypoint bone/palm proxy; not observed segmentation",
                "all_candidates_require_review": True,
                "ready_for_training": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
