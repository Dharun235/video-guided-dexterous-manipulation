import numpy as np

from manipulation.control import ContactFeedback, grasp_detected
from manipulation.metrics import score


def test_feedback_closes_without_pressure_and_backs_off_excess_force():
    controller = ContactFeedback()
    assert np.all(controller.update(np.zeros((5, 2)), 0.1) > 0)
    initial = controller.offsets.copy()
    assert np.all(controller.update(np.full((5, 2), [3.0, 0.0]), 0.1) < initial)
    np.testing.assert_equal(controller.update(np.ones((5, 2)), 0.1, enabled=False), np.zeros(5))


def test_shear_feedback_increases_closure_and_is_bounded():
    quiet = ContactFeedback()
    slipping = ContactFeedback()
    a = quiet.update(np.tile([0.6, 0], (5, 1)), 0.1)
    b = slipping.update(np.tile([0.6, 0.5], (5, 1)), 0.1)
    assert np.all(b > a)
    for _ in range(100):
        slipping.update(np.zeros((5, 2)), 0.1)
    assert np.all(slipping.offsets <= 0.55)


def test_grasp_requires_thumb_and_opposing_finger():
    x = np.zeros((5, 2))
    x[1, 0] = 1
    assert not grasp_detected(x)
    x[0, 0] = 1
    assert grasp_detected(x)
    x[1, 0] = 0
    x[4, 0] = 1
    assert not grasp_detected(x)


def test_scoring_rejects_lift_without_case_contact():
    t = np.arange(100) * 0.02
    pose = np.zeros((100, 7))
    pose[20:60, 2] = 0.05
    result = score(
        t,
        pose,
        np.zeros((100, 6)),
        np.zeros((100, 5, 2)),
        ["retreat"] * 100,
        np.zeros(3),
        np.zeros(3),
        np.zeros((100, 3)),
    )
    assert result["maximum_object_lift_m"] == 0.05
    assert not result["task_success"]
    assert not result["ready_for_training"]
