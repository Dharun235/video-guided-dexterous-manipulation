"""A metric scale change must move hand depth without changing image evidence."""

import cv2
import numpy as np

from humanoid.book_g1 import ORIGIN, SCENE_ROTATION
from video_g1.observations import world_hands


def test_hand_scale_changes_depth_but_preserves_projection():
    rng = np.random.default_rng(12)
    points = rng.uniform(-0.04, 0.04, (21, 3))
    k = np.array([[900.0, 0, 416], [0, 900, 232], [0, 0, 1]])
    pixels = cv2.projectPoints(points, np.zeros(3), np.array([0.01, 0.02, 0.6]), k, None)[0][:, 0]
    hands = {
        "source_frame": np.array([0]),
        "valid": np.array([True]),
        "joints_hand_centered_m": points[None],
        "joints_image_px": pixels[None],
    }
    cameras = {
        "source_frame": np.array([0]),
        "intrinsics": k,
        "rotation_world_to_camera": np.eye(3)[None],
        "translation_world_to_camera_m": np.zeros((1, 3)),
    }
    baseline, valid, _ = world_hands(hands, cameras)
    scaled, scaled_valid, _ = world_hands(hands, cameras, 1.3)
    assert valid.all() and scaled_valid.all()
    baseline_camera = (baseline - ORIGIN) @ SCENE_ROTATION
    scaled_camera = (scaled - ORIGIN) @ SCENE_ROTATION
    np.testing.assert_allclose(scaled_camera, 1.3 * baseline_camera, atol=1e-7)
    uv = scaled_camera[0, :, :2] / scaled_camera[0, :, 2:] * 900 + k[:2, 2]
    np.testing.assert_allclose(uv, pixels, atol=1e-6)
