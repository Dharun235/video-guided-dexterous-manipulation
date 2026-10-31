"""Check physical-prior primitives without accepting inferred contacts as labels."""

import cv2
import numpy as np

from video_g1.refine_hand import box_sdf, refine


def test_signed_box_distance_has_correct_inside_and_outside_units():
    points = np.array([[0, 0, 0], [0.03, 0, 0], [0.04, 0, 0], [0.04, 0.04, 0]])
    distances = box_sdf(points, np.array([0.03, 0.03, 0.01]))
    np.testing.assert_allclose(distances, [-0.01, 0, 0.01, np.sqrt(0.0002)], atol=1e-12)


def test_compatible_pose_without_contact_stays_unchanged():
    rng = np.random.default_rng(8)
    world = rng.uniform([0.04, 0.02, 0.07], [0.08, 0.06, 0.10], (21, 3))
    r = cv2.Rodrigues(np.array([2.7, 0.03, 0.0]))[0]
    t = np.array([0.0, 0.0, 0.6])
    camera = world @ r.T + t
    local = camera - camera[0]
    k = np.array([[900.0, 0, 416], [0, 900, 232], [0, 0, 1.0]])
    pixels = camera[:, :2] / camera[:, 2:] * 900 + k[:2, 2]
    result, report = refine(
        local,
        pixels,
        k,
        r,
        t,
        np.array([0.5, 0.5, 0.03]),
        np.eye(3),
        np.zeros((464, 832), np.uint8),
        [0.22, 0.22, 0.02],
        [0.06, 0.06, 0.02],
    )
    assert report["contact_proposed"] is False
    np.testing.assert_allclose(result, world, atol=1e-6)
    assert report["pixel_error"] < 1e-5
