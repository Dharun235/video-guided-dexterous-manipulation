"""Known-motion checks for the estimators, independent of the recorded sequence."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from reconstruction.object_motion import estimate_pose
from reconstruction.rigid_tracks import robust_rigid
from reconstruction.stitch_object_windows import anchored_pose
from reconstruction.image_pose_refine import refine_pose


def test_rigid_motion_recovers_rotation_and_translation_with_outliers():
    rng = np.random.default_rng(4)
    source = rng.uniform(-.03, .03, (100, 3))
    rotation = Rotation.from_rotvec([.3, -.2, .5]).as_matrix()
    translation = np.array([.15, -.07, .09])
    target = source @ rotation.T + translation + rng.normal(0, .0002, source.shape)
    target[:25] = rng.uniform(-.3, .3, (25, 3))
    result, reason = robust_rigid(source, target, .001)
    assert reason is None
    assert len(result["inliers"]) >= 74
    assert np.linalg.norm(result["translation"] - translation) < .0002
    assert Rotation.from_matrix(result["rotation"] @ rotation.T).magnitude() < .01
    assert np.linalg.det(result["rotation"]) > .999


def test_rigid_motion_rejects_a_line_and_inconsistent_points():
    line = np.c_[np.linspace(0, 1, 40), np.zeros((40, 2))]
    result, reason = robust_rigid(line, line + 1, .01)
    assert result is None and reason == "nearly_collinear_reference_points"
    rng = np.random.default_rng(5)
    result, _ = robust_rigid(rng.normal(size=(100, 3)), rng.normal(size=(100, 3)), .01)
    assert result is None


def test_image_pose_recovers_known_pose_with_pixel_noise_and_outliers():
    rng = np.random.default_rng(8)
    points = rng.uniform(-.025, .025, (100, 3))
    k = np.array([[700., 0, 416], [0, 700, 232], [0, 0, 1]])
    rvec = np.array([.2, -.3, .1])
    translation = np.array([.03, -.02, .5])
    pixels = cv2.projectPoints(points, rvec, translation, k, None)[0].reshape(-1, 2)
    pixels += rng.normal(0, .2, pixels.shape)
    pixels[:20] += rng.normal(0, 35, (20, 2))
    cv2.setRNGSeed(0)
    result, reason = estimate_pose(points, pixels, k)
    assert reason is None
    assert np.linalg.norm(result["tvec"] - translation) < .002
    delta = cv2.Rodrigues(result["rvec"])[0] @ cv2.Rodrigues(rvec)[0].T
    assert Rotation.from_matrix(delta).magnitude() < .02


def test_image_pose_rejects_insufficient_and_degenerate_points():
    k = np.eye(3)
    result, _ = estimate_pose(np.zeros((4, 3)), np.zeros((4, 2)), k)
    assert result is None
    result, _ = estimate_pose(np.zeros((20, 3)), np.zeros((20, 2)), k)
    assert result is None


def test_window_composition_uses_the_overlap_anchor_frame():
    object_r = Rotation.from_rotvec([.2, -.1, .3]).as_matrix()
    object_t = np.array([.02, -.03, .04])
    a = dict(rotation=Rotation.from_rotvec([-.1, .4, .2]).as_matrix(), translation=np.array([.1, .2, -.1]))
    b = dict(rotation=Rotation.from_rotvec([.3, .2, -.4]).as_matrix(), translation=np.array([-.2, .1, .3]))
    anchor_r, anchor_t = a["rotation"] @ object_r, a["rotation"] @ object_t + a["translation"]
    r, t = anchored_pose(b, a, anchor_r, anchor_t)
    np.testing.assert_allclose(r, b["rotation"] @ object_r, atol=1e-12)
    np.testing.assert_allclose(t, b["rotation"] @ object_t + b["translation"], atol=1e-12)


def test_image_refinement_aligns_visible_edges_under_partial_occlusion():
    from itertools import product
    surface = np.array(list(product([-.03, .03], [-.022, .022], [-.012, .012])))
    k = np.array([[700., 0, 416], [0, 700, 232], [0, 0, 1]])
    rotation = Rotation.from_rotvec([.2, -.15, .1]).as_matrix()
    translation = np.array([0., 0., .5])
    pixels = (surface @ rotation.T + translation) @ k.T
    pixels = pixels[:, :2] / pixels[:, 2:]
    case = np.zeros((464, 832), np.uint8)
    cv2.fillConvexPoly(case, cv2.convexHull(pixels.astype(np.float32)).astype(int), 255)
    occluder = np.zeros_like(case)
    occluder[160:300, 425:500] = 255
    case[occluder > 0] = 0
    initial_rotation = Rotation.from_rotvec([0., 0., .05]).as_matrix() @ rotation
    result = refine_pose(surface, initial_rotation, translation + [.004, -.003, .002],
                         np.eye(3), np.zeros(3), k, case, occluder)
    assert result is not None
    assert result["median_visible_edge_error_px"] < 1.5
    assert np.linalg.det(result["rotation"]) > .999
