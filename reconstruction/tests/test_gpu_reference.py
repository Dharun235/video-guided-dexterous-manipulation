import numpy as np
import pytest

from scripts.gpu.video_reference import source_segment


def test_reference_rejects_unobserved_frames_between_samples():
    track = dict(valid=np.ones(20, dtype=bool), requires_review=np.zeros(20, dtype=bool))
    np.testing.assert_equal(source_segment(track, 0, 18), [0, 6, 12, 18])
    track["valid"][7] = False
    with pytest.raises(ValueError, match="observations"):
        source_segment(track, 0, 18)


def test_reference_rejects_unreviewed_or_out_of_range_intervals():
    track = dict(valid=np.ones(20, dtype=bool), requires_review=np.zeros(20, dtype=bool))
    track["requires_review"][5] = True
    with pytest.raises(ValueError, match="observations"):
        source_segment(track, 0, 18)
    with pytest.raises(ValueError, match="interval"):
        source_segment(track, 0, 20)
