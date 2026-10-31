"""Exercise public calibration inputs through a loadable, free-object scene."""

import json
from pathlib import Path

import mujoco
import numpy as np

from video_g1.calibration import run
from video_g1.scene import build

ROOT = Path(__file__).resolve().parents[2]


def test_public_inputs_generate_measured_free_object_scene(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    cameras = tmp_path / "cameras"
    scene = tmp_path / "scene"
    run(ROOT / "configs/book5.json", ROOT / "examples/full-video/camera-observations.json", cameras)
    build(
        ROOT / "configs/book5.json",
        cameras,
        ROOT / "examples/object-motion/scene",
        scene,
        ROOT / "examples/full-video/case-masks",
    )
    model = mujoco.MjModel.from_xml_path(str(scene / "scene.xml"))
    np.testing.assert_allclose(model.geom_size[model.geom("case_solid").id], [0.03, 0.03, 0.01])
    np.testing.assert_allclose(model.geom_size[model.geom("book_collision").id], [0.11, 0.11, 0.01])
    assert model.jnt_type[model.joint("case_free").id] == mujoco.mjtJoint.mjJNT_FREE
    assert model.nu == 0  # The object has no actuator.
    assert model.neq == 0  # The object has no hidden attachment.
    report = json.loads((cameras / "report.json").read_text())
    assert report["source_frames"] == 748
    assert report["mean_corner_residual_px"] < 2
