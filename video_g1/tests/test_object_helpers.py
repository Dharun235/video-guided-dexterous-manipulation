"""Check geometric extent and conservative hand-occlusion proxy contracts."""

import numpy as np

from video_g1.objects import box_surface, hand_occluder


def test_surface_points_cover_only_measured_box_faces():
    size = np.array([0.06, 0.06, 0.02])
    points = box_surface(size)
    np.testing.assert_allclose(points.min(0), -size / 2)
    np.testing.assert_allclose(points.max(0), size / 2)
    assert np.isclose(abs(points), size / 2).any(axis=1).all()


def test_missing_hand_does_not_invent_an_occluder():
    mask = hand_occluder(np.full((21, 2), np.nan), (40, 60))
    assert not mask.any()
    assert mask.dtype == np.uint8
