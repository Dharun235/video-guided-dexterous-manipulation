"""Validate the saved simulated observation/action interface."""

import json
from pathlib import Path

import cv2
import numpy as np


def validate_episode(root):
    """Check array shapes, timestamps and image paths; this does not certify task quality."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    data = np.load(root / "episode.npz")
    n = manifest["frames"]
    expected = {
        "joint_positions": (n, 19),
        "joint_velocities": (n, 19),
        "actions": (n, 13),
        "touch": (n, 5, 2),
        "object_pose": (n, 7),
        "feedforward_joint_torque_Nm": (n, 19),
    }
    for name, shape in expected.items():
        if data[name].shape != shape or not np.isfinite(data[name]).all():
            raise ValueError(f"Invalid {name} shape or values")
    if not np.all(np.diff(data["time_s"]) > 0):
        raise ValueError("Times must increase")
    if (data["touch"] < 0).any():
        raise ValueError("Contact summaries must be nonnegative")
    if not np.allclose(np.linalg.norm(data["object_pose"][:, 3:], axis=1), 1, atol=1e-5):
        raise ValueError("Object quaternions must be normalized")
    for camera in ["front_dataset", "inspection_dataset"]:
        paths = data[f"image_{camera}"]
        if len(paths) != n:
            raise ValueError("Image and state counts differ")
        for path in paths:
            relative = Path(str(path))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Image path must stay inside episode")
            image = cv2.imread(str(root / relative))
            if image is None or list(image.shape) != manifest["image_shape"]:
                raise ValueError(f"Invalid image: {relative}")
    return {
        "frames": n,
        "cameras": 2,
        "finite_arrays": True,
        "synchronized_rows": True,
        "images_readable": True,
        "ready_for_training": False,
    }
