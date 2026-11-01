"""Known-motion checks for the estimators, independent of the recorded sequence."""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from reconstruction.image_pose_refine import refine_pose
from reconstruction.rigid_tracks import robust_rigid
from reconstruction.stitch_object_windows import anchored_pose


def test_rigid_motion_recovers_rotation_and_translation_with_outliers():
    rng = np.random.default_rng(4)
    source = rng.uniform(-0.03, 0.03, (100, 3))
    rotation = Rotation.from_rotvec([0.3, -0.2, 0.5]).as_matrix()
    translation = np.array([0.15, -0.07, 0.09])
    target = source @ rotation.T + translation + rng.normal(0, 0.0002, source.shape)
    target[:25] = rng.uniform(-0.3, 0.3, (25, 3))
    result, reason = robust_rigid(source, target, 0.001)
    assert reason is None
    assert len(result["inliers"]) >= 74
    assert np.linalg.norm(result["translation"] - translation) < 0.0002
    assert Rotation.from_matrix(result["rotation"] @ rotation.T).magnitude() < 0.01
    assert np.linalg.det(result["rotation"]) > 0.999


def test_rigid_motion_rejects_a_line_and_inconsistent_points():
    line = np.c_[np.linspace(0, 1, 40), np.zeros((40, 2))]
    result, reason = robust_rigid(line, line + 1, 0.01)
    assert result is None and reason == "nearly_collinear_reference_points"
    rng = np.random.default_rng(5)
    result, _ = robust_rigid(rng.normal(size=(100, 3)), rng.normal(size=(100, 3)), 0.01)
    assert result is None


def test_window_composition_uses_the_overlap_anchor_frame():
    object_r = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    object_t = np.array([0.02, -0.03, 0.04])
    a = dict(
        rotation=Rotation.from_rotvec([-0.1, 0.4, 0.2]).as_matrix(), translation=np.array([0.1, 0.2, -0.1])
    )
    b = dict(
        rotation=Rotation.from_rotvec([0.3, 0.2, -0.4]).as_matrix(), translation=np.array([-0.2, 0.1, 0.3])
    )
    anchor_r, anchor_t = a["rotation"] @ object_r, a["rotation"] @ object_t + a["translation"]
    r, t = anchored_pose(b, a, anchor_r, anchor_t)
    np.testing.assert_allclose(r, b["rotation"] @ object_r, atol=1e-12)
    np.testing.assert_allclose(t, b["rotation"] @ object_t + b["translation"], atol=1e-12)


def test_image_refinement_aligns_visible_edges_under_partial_occlusion():
    from itertools import product

    surface = np.array(list(product([-0.03, 0.03], [-0.022, 0.022], [-0.012, 0.012])))
    k = np.array([[700.0, 0, 416], [0, 700, 232], [0, 0, 1]])
    rotation = Rotation.from_rotvec([0.2, -0.15, 0.1]).as_matrix()
    translation = np.array([0.0, 0.0, 0.5])
    pixels = (surface @ rotation.T + translation) @ k.T
    pixels = pixels[:, :2] / pixels[:, 2:]
    case = np.zeros((464, 832), np.uint8)
    cv2.fillConvexPoly(case, cv2.convexHull(pixels.astype(np.float32)).astype(int), 255)
    occluder = np.zeros_like(case)
    occluder[160:300, 425:500] = 255
    case[occluder > 0] = 0
    initial_rotation = Rotation.from_rotvec([0.0, 0.0, 0.05]).as_matrix() @ rotation
    result = refine_pose(
        surface,
        initial_rotation,
        translation + [0.004, -0.003, 0.002],
        np.eye(3),
        np.zeros(3),
        k,
        case,
        occluder,
    )
    assert result is not None
    assert result["median_visible_edge_error_px"] < 1.5
    assert np.linalg.det(result["rotation"]) > 0.999
