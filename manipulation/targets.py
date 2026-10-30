"""Create explicit control proposals while retaining raw pose uncertainty."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from humanoid.book_g1 import ORIGIN, SCENE_ROTATION


def task_targets(track, surface):
    valid = np.asarray(track["valid"], bool)
    trusted = valid & ~np.asarray(track["requires_review"], bool)
    if not trusted[0]:
        raise ValueError("Initial object anchor must be unflagged")
    end = np.flatnonzero(valid & (track["source_frame"] >= len(valid) - 30))
    if len(end) < 10:
        raise ValueError("Insufficient end-position observations")
    position = track["translation_world_m"][0]
    rotation = track["rotation_world"][0]
    goal = np.median(track["translation_world_m"][end], axis=0)
    # For this recording the final case is outside the book footprint.
    if abs(goal[0]) < 0.099 and abs(goal[1]) < 0.0645:
        raise ValueError("Goal is over the book; table-plane placement assumption invalid")
    bounds = np.array([surface.min(0), surface.max(0)])
    corners = np.array([[x, y, z] for x in bounds[:, 0] for y in bounds[:, 1] for z in bounds[:, 2]])
    goal[2] = -np.min((corners @ rotation.T)[:, 2]) + 0.0005
    lift_ids = np.flatnonzero(trusted & (track["source_frame"] >= 200) & (track["source_frame"] <= 430))
    height = float(
        np.clip(np.percentile(track["translation_world_m"][lift_ids, 2], 90) - position[2], 0.035, 0.08)
    )
    return dict(
        initial_position_world_m=(SCENE_ROTATION @ position + ORIGIN).tolist(),
        goal_position_world_m=(SCENE_ROTATION @ goal + ORIGIN).tolist(),
        transport_delta_world_m=(SCENE_ROTATION @ (goal - position)).tolist(),
        video_estimated_lift_m=height,
        robot_clearance_margin_m=0.025,
        lift_height_m=height + 0.025,
        goal_source_frames=track["source_frame"][end].tolist(),
        goal_position_requires_review=bool(track["requires_review"][end].any()),
        goal_height_correction_m=float(goal[2] - np.median(track["translation_world_m"][end, 2])),
        rotation_policy="Hold initial orientation; uncertain flip is not reproduced",
        planned_controls_are_observations=False,
        original_pose_flags_preserved=True,
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--track", type=Path, default=Path("examples/object-motion"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    x = np.load(a.track / "object_track.npz")
    surface = np.load(a.track / "case_surface.npy")
    task = task_targets(x, surface)
    # Only supported-position proposals change; validity and review flags are copied.
    raw = {key: x[key].copy() for key in x.files}
    proposed = raw["translation_world_m"].copy()
    correction = np.zeros(len(proposed))
    for i in np.flatnonzero(x["valid"]):
        bottom = float((surface @ x["rotation_world"][i].T + proposed[i])[:, 2].min())
        on_book = abs(proposed[i, 0]) < 0.099 and abs(proposed[i, 1]) < 0.0645
        plane = 0.018 if on_book else 0.0
        correction[i] = max(0, plane - bottom)
        proposed[i, 2] += correction[i]
    raw["proposed_translation_world_m"] = proposed
    raw["support_correction_m"] = correction
    np.savez_compressed(a.output / "review.npz", **raw)
    task["source_track_sha256"] = hashlib.sha256((a.track / "object_track.npz").read_bytes()).hexdigest()
    task["uncertain_pose_frames"] = int((x["requires_review"] & x["valid"]).sum())
    task["missing_pose_frames"] = int((~x["valid"]).sum())
    (a.output / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    print(json.dumps(task, indent=2))


if __name__ == "__main__":
    main()
