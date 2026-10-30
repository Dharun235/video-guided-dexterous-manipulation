import numpy as np
import pytest

from manipulation.targets import task_targets


def test_control_proposal_keeps_uncertain_goal_explicit_and_corrects_support():
    n = 748
    position = np.tile([0.0, 0.0, 0.035], (n, 1))
    position[200:431, 2] = 0.08
    position[700:] = [0.18, -0.02, 0.03]
    track = dict(
        valid=np.ones(n, dtype=bool),
        requires_review=np.arange(n) > 430,
        source_frame=np.arange(n),
        translation_world_m=position,
        rotation_world=np.tile(np.eye(3), (n, 1, 1)),
    )
    surface = np.array([[-0.03, -0.02, -0.015], [0.03, 0.02, 0.015]])
    original = position.copy()
    result = task_targets(track, surface)
    assert result["goal_position_requires_review"]
    assert result["rotation_policy"].startswith("Hold initial")
    assert result["goal_height_correction_m"] == pytest.approx(-0.0145)
    np.testing.assert_equal(position, original)


def test_unreviewed_initial_anchor_is_rejected():
    track = dict(valid=np.ones(748, dtype=bool), requires_review=np.ones(748, dtype=bool))
    with pytest.raises(ValueError, match="anchor"):
        task_targets(track, np.zeros((2, 3)))
