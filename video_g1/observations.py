"""Register observed hand estimates into the calibrated book world and robot frame."""

import cv2
import numpy as np

from humanoid.book_g1 import ORIGIN, SCENE_ROTATION


def world_hands(hands, cameras, hand_scale=1.0):
    """Return robot-world hand landmarks and validity without filling missing observations."""
    if not np.array_equal(hands["source_frame"], cameras["source_frame"]):
        raise ValueError("Camera and hand timelines differ")
    n = len(hands["valid"])
    points = np.full((n, 21, 3), np.nan)
    valid = np.asarray(hands["valid"], bool).copy()
    errors = np.full(n, np.nan)
    k = cameras["intrinsics"]
    for i in np.flatnonzero(valid):
        local = hands["joints_hand_centered_m"][i] * hand_scale
        uv = hands["joints_image_px"][i]
        if not np.isfinite(local).all() or not np.isfinite(uv).all():
            valid[i] = False
            continue
        ok, r, t = cv2.solvePnP(local, uv, k, None)
        if not ok or not np.isfinite(t).all() or t[2, 0] <= 0:
            valid[i] = False
            continue
        camera = local @ cv2.Rodrigues(r)[0].T + t.ravel()
        world = (camera - cameras["translation_world_to_camera_m"][i]) @ cameras["rotation_world_to_camera"][
            i
        ]
        points[i] = world @ SCENE_ROTATION.T + ORIGIN
        projected = cv2.projectPoints(local, r, t, k, None)[0].reshape(-1, 2)
        errors[i] = np.mean(np.linalg.norm(projected - uv, axis=1))
    return points, valid, errors
