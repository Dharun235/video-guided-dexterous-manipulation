"""Regressions for parked fingers and support-plane force mistaken for object contact."""

import numpy as np

from video_g1.sequence import contact_requirements, force_step


def opposed_wrap():
    """Use one opposed body with four independently loaded non-thumb fingers."""
    return np.array([[1.2, 0, 0], [-0.48, 0, 0], [-0.36, 0, 0], [-0.24, 0, 0], [-0.12, 0, 0]])


def test_pressure_on_book_cannot_replace_missing_grasp_finger():
    case = opposed_wrap()
    case[4] = 0
    book = np.zeros((5, 3))
    book[4] = [0, 0, 2]
    target = np.array([1.2, 0.48, 0.36, 0.24, 0.12])
    candidate, quality = contact_requirements({17: case, 42: book}, np.arange(5), np.arange(5), target)
    assert candidate == 17
    assert quality[17] == 0
    assert quality[42] == 0


def test_all_requested_fingers_are_required_on_same_body():
    target = np.array([1.2, 0.48, 0.36, 0.24, 0.12])
    candidate, quality = contact_requirements({17: opposed_wrap()}, np.arange(5), np.arange(5), target)
    assert candidate == 17
    assert quality[17] > 0.5


def test_unloaded_finger_recloses_instead_of_latching_old_touch():
    assert force_step(0.3, 0, 0.12, 0.01, 0.9) > 0.3
    assert force_step(0.3, 0.12, 0.12, 0.01, 0.9) == 0.3
    assert force_step(0.3, 0.4, 0.12, 0.01, 0.9) < 0.3
    assert force_step(0.9, 0, 0.12, 0.01, 0.9) == 0.9
