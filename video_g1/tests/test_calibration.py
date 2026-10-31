"""Check camera conventions against a synthetic scene of known metric size."""

import cv2
import numpy as np

from video_g1.calibration import plane_corners, solve_plane


def test_metric_plane_registration_recovers_camera():
    size = [0.22, 0.22, 0.02]
    k = np.array([[900.0, 0, 416], [0, 900, 232], [0, 0, 1]])
    r = np.array([2.4, 0.08, -0.03])
    t = np.array([0.02, 0.01, 0.7])
    uv = cv2.projectPoints(plane_corners(size), r, t, k, None)[0][:, 0]
    recovered_r, recovered_t, residual = solve_plane(uv, size, k)
    np.testing.assert_allclose(recovered_r, cv2.Rodrigues(r)[0], atol=1e-7)
    np.testing.assert_allclose(recovered_t, t, atol=1e-7)
    assert max(residual) < 1e-5
