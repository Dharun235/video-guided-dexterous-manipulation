"""Ensure the published physical evidence matches its recorded checksums."""

import hashlib
from pathlib import Path


def test_published_sequence_bundle_checksums():
    folder = Path(__file__).resolve().parents[2] / "examples/video-to-g1"
    for line in (folder / "SHA256SUMS").read_text().splitlines():
        expected, filename = line.split("  ", 1)
        assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == expected, filename


def test_published_grasp_has_no_parked_fist_and_loads_all_fingers():
    """The reported pre-contact fist and unloaded curled fingers must not return."""
    import numpy as np

    folder = Path(__file__).resolve().parents[2] / "examples/video-to-g1"
    reference = np.load(folder / "reference.npz")
    run = np.load(folder / "rollout.npz")
    np.testing.assert_array_equal(reference["grasp_finger_ids"], np.arange(5))
    np.testing.assert_array_equal(reference["confirmation_finger_ids"], np.arange(5))
    approach = (run["source_frame"] >= 225) & (run["source_frame"] <= 255)
    assert np.max(run["actions"][approach, 7:11]) < 0.5
    turn = (run["source_frame"] >= 405) & (run["source_frame"] <= 570)
    contact_fraction = (run["case_finger_resultant_normal_N"][turn] > 0.05).mean(axis=0)
    assert np.all(contact_fraction > 0.95)
