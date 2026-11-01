"""Exercise input rejection and temporal reasoning without model downloads."""

import cv2
import numpy as np

from reconstruction.scene_geometry import fit_support, rank_objects, visible_extent
from reconstruction.video_checks import assess, inspect_video


def facts():
    """Supply a decodable continuous clip for isolated gate tests."""
    return dict(incomplete_decode=False, nonmonotonic_timestamps=False, cuts=[], duration_s=3)


def rows(visible):
    """Supply explicitly observed presence; this does not simulate model accuracy."""
    return [
        dict(hands=[{}] if present else [], quality=dict(blur_variance=80, clipped_fraction=0.01))
        for present in visible
    ]


def test_no_hand_rejects_but_brief_occlusion_does_not():
    missing = assess(facts(), rows([False] * 12))
    assert missing["status"] == "blocked"
    assert missing["reasons"][0]["code"] == "hand_not_detected"
    assert assess(facts(), rows([True, True, False, True, True]))["status"] == "pass"


def test_one_detection_is_insufficient_and_cut_requires_review():
    assert assess(facts(), rows([False, True, False]))["status"] == "review"
    video = facts()
    video["cuts"] = [dict(frame=2)]
    assert assess(video, rows([True] * 5))["status"] == "review"


def test_quality_and_timing_reject_unusable_evidence():
    video = facts()
    video["nonmonotonic_timestamps"] = True
    assert assess(video, rows([True] * 5))["status"] == "blocked"
    evidence = rows([True] * 5)
    for row in evidence:
        row["quality"]["blur_variance"] = 0
    assert assess(facts(), evidence)["reasons"][0]["code"] == "severe_blur"


def test_invalid_video_is_not_misreported_as_missing_hand(tmp_path):
    path = tmp_path / "bad.mp4"
    path.write_bytes(b"not a video")
    video, samples = inspect_video(path)
    result = assess(video, samples)
    assert result["reasons"][0]["code"] == "invalid_video"
    assert samples == []


def test_decode_catches_an_abrupt_scene_change(tmp_path):
    path = tmp_path / "cut.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 90))
    assert writer.isOpened()
    rng = np.random.default_rng(3)
    for i in range(20):
        im = rng.integers(0, 20, (90, 160, 3), dtype=np.uint8)
        im += np.array([170, 0, 0] if i < 10 else [0, 170, 0], np.uint8)
        writer.write(im)
    writer.release()
    video, samples = inspect_video(path)
    assert video["decoded_frames"] == 20
    assert any(abs(c["frame"] - 10) <= 1 for c in video["cuts"])
    assert len(samples) == 4


def test_plane_fit_recovers_support_with_outliers():
    rng = np.random.default_rng(5)
    x, y = np.meshgrid(np.linspace(-0.2, 0.2, 80), np.linspace(-0.15, 0.15, 60))
    points = np.stack([x, y, 1 + 0.0005 * rng.normal(size=x.shape)], axis=-1)
    points[:6, :, 2] += 0.12
    plane = fit_support(points, np.ones(x.shape, bool), np.zeros(x.shape, bool))
    assert plane["residual_m"] < 0.002
    assert plane["inlier_fraction"] > 0.9
    assert plane["rotation"][2] @ np.array([0, 0, -1]) > 0.99
    cloud, lo, hi = visible_extent(points, np.ones(x.shape, bool), plane)
    assert cloud.shape == (4800, 3)
    assert np.all(hi >= lo)


def test_background_mask_is_not_accepted_as_an_object():
    image = np.zeros((120, 160, 3), np.uint8)
    candidate = dict(mask=np.ones((120, 160), bool), score=0.99)
    assert rank_objects([candidate], [], image) == []


def test_scene_export_produces_free_loadable_object(tmp_path):
    """A synthetic known plane/object tests file and frame conventions, not model accuracy."""
    import mujoco

    from reconstruction.scene_geometry import export_scene

    h, w = 80, 120
    x, y = np.meshgrid(np.linspace(-0.2, 0.2, w), np.linspace(-0.15, 0.15, h))
    z = np.ones((h, w))
    mask = (abs(x) < 0.04) & (abs(y) < 0.04)
    z[mask] = 0.94 + 0.003 * np.sin(x[mask] * 100)
    points = np.stack([x, y, z], axis=-1)
    image = np.full((h, w, 3), 120, np.uint8)
    result = export_scene(tmp_path, image, dict(points=points, mask=np.ones((h, w), bool)), mask, [])
    assert result["scene_check"]["finite_state"]
    assert result["scene_check"]["stable_under_assumed_physics"]
    assert result["scene_check"]["loads_in_mujoco"]
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "scene.xml"))
    assert model.nu == 0 and model.neq == 0
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE


def test_missing_models_are_environment_errors_not_absent_hands(tmp_path):
    from reconstruction.scene_check import run

    path = tmp_path / "valid.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    for _ in range(10):
        writer.write(frame)
    writer.release()
    result = run(path, tmp_path / "result", tmp_path / "missing-models")
    assert result["status"] == "blocked"
    assert result["error_kind"] == "environment"
    assert result["reasons"][0]["code"] == "inference_environment_error"
    assert (tmp_path / "result/report.json").exists()


def test_brightness_discontinuity_is_detected_even_with_similar_hue(tmp_path):
    path = tmp_path / "invert.avi"
    rng = np.random.default_rng(9)
    frame = rng.integers(80, 120, (90, 160, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 90))
    for i in range(20):
        writer.write(frame if i < 10 else 255 - frame)
    writer.release()
    video, _ = inspect_video(path)
    assert any(abs(c["frame"] - 10) <= 1 for c in video["cuts"])
