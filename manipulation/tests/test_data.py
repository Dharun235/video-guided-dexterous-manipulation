import json

import cv2
import numpy as np
import pytest

from manipulation.data import validate_episode


def episode(tmp_path):
    data = dict(
        time_s=np.array([0.0, 0.1]),
        joint_positions=np.zeros((2, 19)),
        joint_velocities=np.zeros((2, 19)),
        actions=np.zeros((2, 13)),
        touch=np.zeros((2, 5, 2)),
        object_pose=np.tile([0, 0, 0, 1, 0, 0, 0], (2, 1)),
        feedforward_joint_torque_Nm=np.zeros((2, 19)),
    )
    for camera in ["front_dataset", "inspection_dataset"]:
        (tmp_path / camera).mkdir()
        paths = []
        for i in range(2):
            path = f"{camera}/{i}.jpg"
            cv2.imwrite(str(tmp_path / path), np.zeros((8, 8, 3), np.uint8))
            paths.append(path)
        data[f"image_{camera}"] = paths
    (tmp_path / "manifest.json").write_text(json.dumps(dict(frames=2, image_shape=[8, 8, 3])))
    np.savez(tmp_path / "episode.npz", **data)
    return data


def test_episode_checks_images_and_synced_states(tmp_path):
    episode(tmp_path)
    result = validate_episode(tmp_path)
    assert result["synchronized_rows"]
    assert not result["ready_for_training"]
    (tmp_path / "front_dataset/0.jpg").unlink()
    with pytest.raises(ValueError, match="image"):
        validate_episode(tmp_path)


def test_episode_rejects_wrong_action_interface(tmp_path):
    data = episode(tmp_path)
    data["actions"] = np.zeros((2, 19))
    np.savez(tmp_path / "episode.npz", **data)
    with pytest.raises(ValueError, match="actions"):
        validate_episode(tmp_path)
