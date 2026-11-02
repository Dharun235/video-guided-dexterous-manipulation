"""Known-transform and moving-surface checks independent of recorded-video calibration."""

import numpy as np
from scipy.spatial.transform import Rotation

from reconstruction.fusion_geometry import register_similarity, static_consensus


def test_similarity_registration_rejects_moving_object_outliers():
    rng = np.random.default_rng(8)
    source = rng.uniform([-0.3, -0.2, 0.5], [0.3, 0.2, 1.5], (100, 3))
    r = Rotation.from_rotvec([0.05, -0.1, 0.2]).as_matrix()
    t = np.array([0.04, 0.02, -0.03])
    scale = 1.15
    target = scale * source @ r.T + t + rng.normal(0, 0.0003, source.shape)
    target[:25] = rng.uniform(-1, 1, (25, 3))
    result = register_similarity(source, target, threshold=0.003)
    assert result is not None
    assert abs(result["scale"] - scale) < 0.002
    np.testing.assert_allclose(result["camera_to_world"][:3, :3], r, atol=0.002)
    np.testing.assert_allclose(result["camera_to_world"][:3, 3], t, atol=0.001)
    assert result["inliers"][:25].sum() == 0


def test_inadequate_registration_is_not_replaced_with_identity():
    rng = np.random.default_rng(2)
    assert register_similarity(rng.normal(size=(8, 3)), rng.normal(size=(8, 3))) is None
    assert register_similarity(rng.normal(size=(80, 3)), rng.normal(size=(80, 3))) is None


def view(depth, image=None, excluded=None):
    """Construct a small synchronized pinhole RGB-D observation."""
    h, w = depth.shape
    k = np.array([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1]])
    y, x = np.mgrid[:h, :w]
    points = np.stack([(x - w * 0.5) / w * depth, (y - h * 0.5) / h * depth, depth], axis=-1)
    return dict(
        depth=depth,
        points=points,
        intrinsics=k,
        scale=1.0,
        camera_to_world=np.eye(4),
        mask=np.ones((h, w), bool),
        excluded=np.zeros((h, w), bool) if excluded is None else excluded,
        image=np.full((h, w, 3), 150, np.uint8) if image is None else image,
    )


def test_temporal_consensus_removes_departed_foreground_but_keeps_occluded_background():
    normal = np.ones((40, 60), np.float32)
    moved = normal.copy()
    moved[12:28, 20:40] = 0.9
    observations = [view(moved), *[view(normal) for _ in range(4)]]
    filtered = static_consensus(observations)
    assert filtered[0]["dynamic"][16:24, 24:36].all()
    assert not filtered[1]["dynamic"].any()
    assert not filtered[0]["dynamic"][:8].any()


def test_excluded_human_pixels_cannot_become_static_surface():
    excluded = np.zeros((30, 40), bool)
    excluded[8:20, 10:25] = True
    observations = [view(np.ones((30, 40), np.float32), excluded=excluded) for _ in range(4)]
    filtered = static_consensus(observations)
    for result in filtered:
        assert not result["static"][excluded].any()
        assert not result["dynamic"][excluded].any()


def test_every_frame_camera_pose_uses_static_landmarks(monkeypatch):
    import cv2

    from reconstruction import video_fusion

    rng = np.random.default_rng(11)
    world = rng.uniform([-0.3, -0.2, 0.5], [0.3, 0.2, 1.0], (80, 3))
    rvec = np.array([0.04, -0.03, 0.02])
    translation = np.array([0.01, 0.02, 0.03])
    k = np.array([[500.0, 0, 256], [0, 500.0, 144], [0, 0, 1.0]])
    pixels = cv2.projectPoints(world, rvec, translation, k, None)[0].reshape(-1, 2)
    pixels[:12] += 80
    monkeypatch.setattr(video_fusion, "correspondences", lambda *_: (pixels, world))
    result = video_fusion.track_camera(None, [dict(frame=0)], k)
    target = np.eye(4)
    target[:3, :3] = cv2.Rodrigues(rvec)[0]
    target[:3, 3] = translation
    np.testing.assert_allclose(result["camera_to_world"], np.linalg.inv(target), atol=1e-5)
    assert video_fusion.track_camera(None, [], k) is None


def test_forearm_is_excluded_when_person_segmentation_misses_it():
    from reconstruction.video_fusion import excluded_people

    points = np.tile([0.5, 0.4, 0.0], (21, 1))
    points[0] = [0.5, 0.6, 0.0]
    points[5] = [0.4, 0.4, 0.0]
    points[17] = [0.6, 0.4, 0.0]
    mask = excluded_people(
        np.zeros((100, 100, 3), np.uint8), [dict(landmarks=points)], lambda _: np.zeros((100, 100), bool)
    )
    assert mask[90, 50] and mask[50, 50]
    assert not mask[10, 10]


def test_wrong_video_depth_cache_is_rejected_before_inference(tmp_path):
    import json

    import pytest

    from reconstruction.video_fusion import run

    path = tmp_path / "clip.mp4"
    path.write_bytes(b"clip one")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "report.json").write_text(json.dumps(dict(input_sha256="wrong")))
    with pytest.raises(ValueError, match="source video differs"):
        run(path, tmp_path / "output", tmp_path / "models", depth_cache=cache)


def test_cpu_tsdf_preserves_plane_and_exports_loadable_scene(tmp_path):
    import pytest

    pytest.importorskip("open3d")
    import mujoco

    from reconstruction.fusion_geometry import export_environment, fuse_views

    observations = [view(np.ones((60, 80), np.float32)) for _ in range(4)]
    filtered = static_consensus(observations)
    mesh = fuse_views(observations, filtered, voxel_size=0.02)
    assert len(mesh.triangles) > 100
    np.testing.assert_allclose(np.asarray(mesh.vertices)[:, 2], 1, atol=0.015)
    result = export_environment(mesh, tmp_path)
    assert result["visual_mesh_collision"] is False
    assert result["hidden_surfaces_filled"] is False
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "scene.xml"))
    assert model.nu == 0 and model.neq == 0


def test_bidirectional_flow_recovers_pose_from_static_image_evidence():
    import cv2

    from reconstruction.video_fusion import track_flow_camera

    rng = np.random.default_rng(4)
    image = rng.integers(0, 255, (288, 512, 3), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 2], [0, 1, -1]]), (512, 288))
    y, x = np.meshgrid(np.linspace(60, 230, 6), np.linspace(80, 430, 10))
    pixels = np.c_[x.ravel(), y.ravel()]
    world = np.c_[(pixels[:, 0] - 256) / 500, (pixels[:, 1] - 144) / 500, np.ones(len(pixels))]
    previous = dict(image=image, pixels=pixels, world=world)
    k = np.array([[500.0, 0, 256], [0, 500.0, 144], [0, 0, 1.0]])
    result = track_flow_camera(previous, shifted, np.zeros((288, 512), bool), k)
    assert result is not None and result["method"] == "optical_flow_pnp"
    np.testing.assert_allclose(result["camera_to_world"][:3, 3], [-0.004, 0.002, 0], atol=0.001)
    assert track_flow_camera(previous, shifted, np.ones((288, 512), bool), k) is None


def test_fusion_review_stops_downstream_execution(monkeypatch):
    import sys

    import pytest

    from reconstruction import video_fusion

    monkeypatch.setattr(sys, "argv", ["video_fusion", "input.mp4", "--output", "result"])
    monkeypatch.setattr(video_fusion, "run", lambda *a, **k: dict(status="review"))
    with pytest.raises(SystemExit) as stopped:
        video_fusion.main()
    assert stopped.value.code == 3


def test_corrupt_fusion_input_stops_before_model_loading(tmp_path):
    from reconstruction.video_fusion import run

    video = tmp_path / "broken.mp4"
    video.write_bytes(b"broken")
    report = run(video, tmp_path / "result", tmp_path / "models-missing")
    assert report["status"] == "blocked" and report["reason"] == "invalid_video"
    assert (tmp_path / "result/report.json").exists()


def test_texture_baking_preserves_source_detail_and_rejects_excluded_pixels():
    from reconstruction.fusion_geometry import source_texture

    vertices = np.array([[0.0, 0.0, 1.0], [0.1, 0, 1], [0, 0.1, 1]])
    faces = np.array([[0, 1, 2]])
    fallback = np.zeros((3, 3))
    y, x = np.mgrid[:100, :100]
    image = np.stack([np.zeros_like(x), y * 2, x * 2], axis=-1).astype(np.uint8)
    observation = dict(
        image=image,
        static=np.ones((100, 100), bool),
        depth=np.ones((100, 100)),
        scale=1.0,
        intrinsics=np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]]),
        camera_to_scene=np.eye(4),
    )
    atlas, fraction = source_texture(vertices, faces, fallback, [observation], side=1, tile=8)
    np.testing.assert_array_equal(atlas[0, 0], [100, 100, 0])
    np.testing.assert_array_equal(atlas[0, -1], [120, 100, 0])
    assert fraction == 1
    observation["static"][:, 55:] = False
    masked, fraction = source_texture(vertices, faces, fallback, [observation], side=1, tile=8)
    assert not masked.any() and fraction == 0


def test_published_fused_scene_assets_are_consistent_and_static():
    import hashlib
    import json
    from pathlib import Path

    import mujoco

    folder = Path(__file__).resolve().parents[2] / "examples/fused-scene"
    report = json.loads((folder / "report.json").read_text())
    for name, expected in report["artifact_sha256"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    assert model.nu == 0 and model.neq == 0
    visual = model.geom("environment_visual").id
    assert model.geom_contype[visual] == 0 and model.geom_conaffinity[visual] == 0
    assert model.geom_contype.sum() > 0 and model.ncam == 1
