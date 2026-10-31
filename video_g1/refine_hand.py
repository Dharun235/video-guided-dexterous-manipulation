"""Refine monocular hand registration using explicit table and contact priors.

This adjusts a rigid camera registration of each estimated hand, not the raw
observations or its articulation. Contact candidates come from image proximity;
they are hypotheses, not measured tactile labels.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from humanoid.book_g1 import ORIGIN, SCENE_ROTATION


def box_sdf(points, half_size):
    """Return signed Euclidean distance to an axis-aligned box in metres."""
    q = abs(points) - half_size
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0)


def refine(local, pixels, k, camera_r, camera_t, object_p, object_r, mask, book_size, case_size):
    """Fit a rigid hand registration with image evidence and soft physical priors."""
    ok, r, t = cv2.solvePnP(local, pixels, k, None)
    if not ok or t[2, 0] <= 0:
        return None
    initial = local @ cv2.Rodrigues(r)[0].T + t.ravel()
    centre = initial[0].copy()
    # Sample bone interiors too: human contact need not occur at a fingertip.
    parents = [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
    children = list(range(1, 21))
    samples = np.vstack(
        [initial] + [(initial[parents] * (1 - u) + initial[children] * u) for u in [0.25, 0.5, 0.75]]
    )
    projected = samples[:, :2] / samples[:, 2:] * k[0, 0] + k[:2, 2]
    distance = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
    px = np.rint(projected).astype(int)
    inside = (px[:, 0] >= 0) & (px[:, 0] < mask.shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < mask.shape[0])
    candidates = np.zeros(len(samples), bool)
    candidates[inside] = distance[px[inside, 1], px[inside, 0]] < 8
    has_contact = bool(candidates.sum() >= 2)
    depth = float(np.median(initial[:, 2]))
    bone_half = np.array(case_size) / 2

    def transformed(x):
        return (samples - centre) @ Rotation.from_rotvec(x[:3]).as_matrix().T + centre + x[3:]

    def residual(x):
        cp = transformed(x)
        world = (cp - camera_t) @ camera_r
        uv = cp[:21, :2] / cp[:21, 2:] * k[0, 0] + k[:2, 2]
        on_book = (abs(world[:, 0]) < book_size[0] / 2) & (abs(world[:, 1]) < book_size[1] / 2)
        floor = np.where(on_book, book_size[2] + 0.008, 0.008)
        # The table footprint is assumed; this cannot prove true scene depth.
        table = np.minimum(world[:, 2] - floor, 0) * 8
        obj = (world - object_p) @ object_r
        sdf = box_sdf(obj, bone_half)
        penetration = np.minimum(sdf + 0.002, 0) * 4
        contact = np.array([np.min(abs(sdf[candidates])) * 4 if has_contact else 0])
        return np.r_[
            (uv - pixels).ravel() * depth / k[0, 0], table, penetration, contact, x[:3] * 0.015, x[3:] * 0.03
        ]

    fit = least_squares(
        residual,
        np.zeros(6),
        bounds=([-0.7] * 3 + [-0.3] * 3, [0.7] * 3 + [0.3] * 3),
        max_nfev=50,
        loss="soft_l1",
        f_scale=0.01,
    )
    cp = transformed(fit.x)[:21]
    world = (cp - camera_t) @ camera_r
    uv = cp[:, :2] / cp[:, 2:] * k[0, 0] + k[:2, 2]
    return world, {
        "pixel_error": float(np.mean(np.linalg.norm(uv - pixels, axis=1))),
        "contact_proposed": has_contact,
        "rotation_correction_rad": fit.x[:3].tolist(),
        "translation_correction_camera_m": fit.x[3:].tolist(),
        "converged": bool(fit.success),
    }


def main():
    """Write all source frames and explicit refinement uncertainty for a diagnostic fit."""
    p = argparse.ArgumentParser(__doc__)
    for name in ["hands", "cameras", "objects", "case-masks", "config", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    h = np.load(a.hands / "hand_track.npz")
    c = np.load(a.cameras / "cameras.npz")
    o = np.load(a.objects)
    cfg = json.loads(a.config.read_text())
    local = h["joints_hand_centered_m"]
    straight = np.linalg.norm(local[:, 12] - local[:, 0], axis=1)
    chain = np.linalg.norm(np.diff(local[:, [0, 9, 10, 11, 12]], axis=1), axis=2).sum(axis=1)
    flat = h["valid"] & (straight / chain > 0.97)
    scale = cfg["recording"]["hand_length_m"] / np.median(straight[flat])
    points = np.full_like(local, np.nan)
    valid = h["valid"].copy()
    report = []
    for i in np.flatnonzero(valid):
        # Object gaps are not silently converted to observed poses.
        if not o["valid"][i]:
            valid[i] = False
            continue
        mask = cv2.imread(str(a.case_masks / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Missing case mask {i}")
        result = refine(
            local[i] * scale,
            h["joints_image_px"][i],
            c["intrinsics"],
            c["rotation_world_to_camera"][i],
            c["translation_world_to_camera_m"][i],
            o["translation_world_m"][i],
            o["rotation_world"][i],
            mask,
            cfg["book_size_m"],
            cfg["case_size_m"],
        )
        if result is None:
            valid[i] = False
            continue
        world, row = result
        points[i] = world @ SCENE_ROTATION.T + ORIGIN
        row["source_frame"] = int(i)
        row["object_requires_review"] = bool(o["requires_review"][i])
        report.append(row)
    np.savez_compressed(
        a.output / "observations.npz",
        source_frame=h["source_frame"],
        source_time_s=h["seconds"],
        landmarks_m=points,
        valid=valid,
        raw_hand_observed=h["valid"],
        requires_review=np.ones(len(valid), bool),
    )
    (a.output / "frames.json").write_text(json.dumps(report, indent=2) + "\n")
    (a.output / "report.json").write_text(
        json.dumps(
            {
                "method": "Rigid hand registration with soft image, assumed support and image-inferred contact priors",
                "hand_scale": float(scale),
                "source_frames": len(valid),
                "refined_frames": int(valid.sum()),
                "mean_pixel_error": float(np.mean([r["pixel_error"] for r in report])),
                "raw_observations_changed": False,
                "contact_is_inferred": True,
                "ready_for_training": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
