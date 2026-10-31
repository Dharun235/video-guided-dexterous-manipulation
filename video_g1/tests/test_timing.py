"""Protect full-frame coverage and distinguish inferred commands from observations."""

import numpy as np

from video_g1.timing import fill_control_gaps, retime_controls


def test_gap_commands_keep_observation_provenance():
    times = np.arange(5, dtype=float)
    observed = np.array([False, True, False, True, False])
    controls = np.array([[np.nan], [0.0], [np.nan], [2.0], [np.nan]])
    filled, origin = fill_control_gaps(times, controls, observed)
    np.testing.assert_array_equal(filled[:, 0], [0, 0, 1, 2, 2])
    np.testing.assert_array_equal(origin, [2, 0, 1, 0, 2])
    assert np.isnan(controls[~observed]).all()


def test_retiming_preserves_samples_and_respects_speed():
    source = np.array([0.0, 0.1, 0.2, 0.3])
    controls = np.array([[0.0, 0], [0.5, 0], [0.5, 0.8], [0.5, 0.8]])
    speeds = np.array([1.0, 2.0])
    execution = retime_controls(source, controls, speeds)
    assert execution.shape == source.shape
    assert (np.diff(execution) >= np.diff(source)).all()
    actual = abs(np.diff(controls, axis=0)) / np.diff(execution)[:, None]
    assert (actual <= speeds + 1e-12).all()


def test_nonfinite_commands_are_rejected_before_physics():
    import pytest

    with pytest.raises(ValueError, match="finite"):
        retime_controls([0, 1], [[0], [np.nan]], 1)
    with pytest.raises(ValueError, match="finite"):
        fill_control_gaps([0, 1], [[0], [np.nan]], [True, True])
