"""Check confirmation and loss handling without object-class information."""

from video_g1.sequence import ContactGate


def test_parallel_pressure_never_allows_carry():
    gate = ContactGate()
    for _ in range(100):
        assert not gate.update({17: 0.0}, 0.01)
    assert not gate.confirmed


def test_confirmation_requires_same_body_then_loss_aborts():
    gate = ContactGate()
    for _ in range(10):
        gate.update({17: 1.0}, 0.01)
    for _ in range(10):
        gate.update({42: 1.0}, 0.01)
    assert not gate.confirmed
    for _ in range(6):
        gate.update({42: 1.0}, 0.01)
    assert gate.confirmed
    for _ in range(16):
        gate.update({17: 1.0}, 0.01, carrying=True)
    assert gate.aborted
    assert not gate.update({42: 1.0}, 1.0, carrying=True)


def test_brief_loss_recovers_and_release_does_not_abort():
    gate = ContactGate()
    gate.update({900: 1.0}, 0.16)
    assert gate.update({}, 0.1, carrying=True)
    assert gate.update({900: 1.0}, 0.1, carrying=True)
    assert gate.lost_s == 0
    assert gate.update({}, 10, carrying=False)
    assert not gate.aborted
