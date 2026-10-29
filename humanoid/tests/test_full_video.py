import json

import numpy as np
import pytest

from reconstruction.video_hands import export_track
from humanoid.retarget_hand import camera_track


def test_hand_export_keeps_missing_frames_without_fabricating_poses(tmp_path):
    hand = dict(is_right=True, keypoints_3d=np.zeros((21, 3)).tolist(),
                keypoints_2d=np.zeros((21, 2)).tolist(), vertices=np.zeros((778, 3)).tolist(),
                cam_t=[1, 2, 3], global_orient=np.eye(3).tolist())
    rows = [dict(source_frame=i, seconds=i / 60, hands=[hand] if i == 1 else []) for i in range(3)]
    (tmp_path / 'hands.json').write_text(json.dumps(rows))
    export_track(tmp_path)
    x = np.load(tmp_path / 'hand_track.npz')
    assert x['valid'].tolist() == [False, True, False]
    assert np.isnan(x['joints_camera_m'][0]).all()
    np.testing.assert_equal(x['joints_camera_m'][1], np.tile([1, 2, 3], (21, 1)))
    assert len(json.loads((tmp_path / 'coverage.json').read_text())['missing_intervals']) == 2


def test_hand_export_rejects_sampled_frames_as_full_video(tmp_path):
    (tmp_path / 'hands.json').write_text(json.dumps([dict(source_frame=5, seconds=.1, hands=[])]))
    with pytest.raises(ValueError, match='every frame'):
        export_track(tmp_path)


def test_camera_interpolation_preserves_orientation_and_translation():
    rows = [dict(seconds=t, rotation=[0, 0, 0], translation=[t, 0, 0]) for t in (0, 1)]
    rotations, positions = camera_track(rows, np.array([0, .5, 1]))
    np.testing.assert_allclose(rotations, np.tile(np.eye(3), (3, 1, 1)))
    np.testing.assert_allclose(positions[:, 0], [0, .5, 1])
