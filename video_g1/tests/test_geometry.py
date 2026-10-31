"""Check palm orientation derivatives independently by finite differences."""

import numpy as np

from video_g1.geometry import palm_frame


def test_palm_derivative_matches_finite_difference():
    rng = np.random.default_rng(42)
    points = rng.normal(size=(21, 3))
    directions = rng.normal(size=(21, 3, 4))
    basis, derivative = palm_frame(points, directions)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
    assert np.linalg.det(basis) > 0
    step = 1e-6
    for channel in range(4):
        delta = directions[:, :, channel] * step
        numerical = (palm_frame(points + delta) - palm_frame(points - delta)) / (2 * step)
        np.testing.assert_allclose(derivative[:, :, channel], numerical, atol=1e-8)
