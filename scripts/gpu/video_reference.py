"""Build a grasp-conditioned 6-DoF video reference, without prescribed task phases.

Only an unflagged source segment is used. Interpolated robot/object references
are control proposals, not added human observations. The optimizer must move a
free object through contact; this file does not execute object pose playback.
"""

import argparse
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from humanoid.book_g1 import ORIGIN, SCENE_ROTATION
from humanoid.kinematics import set_configuration
from manipulation.fit import case_pose


def source_segment(track, start, end, stride=6):
    """Require every source frame to be valid and unflagged before decimation."""
    if start < 0 or end <= start or end >= len(track["valid"]) or stride < 1:
        raise ValueError("Invalid source interval")
    all_frames = np.arange(start, end + 1)
    if not np.all(track["valid"][all_frames] & ~track["requires_review"][all_frames]):
        raise ValueError("Reference interval includes invalid or flagged observations")
    return all_frames[::stride]


def main():
    """Build smoothed, retimed control references from an unflagged in-hand video segment."""
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--objects", type=Path, default=Path("examples/object-motion"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start", type=int, default=228)
    p.add_argument("--end", type=int, default=426)
    p.add_argument("--retime", type=float, default=2.0)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(a.bundle / "assets", a.output / "assets")
    shutil.copy2(a.bundle / "scene.xml", a.output / "scene.xml")
    m = mujoco.MjModel.from_xml_path(str((a.output / "scene.xml").resolve()))
    d = mujoco.MjData(m)
    original = np.load(a.bundle / "reference.npz")
    track = np.load(a.objects / "object_track.npz")
    ids = source_segment(track, a.start, a.end)
    object_adr = m.jnt_qposadr[m.joint("case_free").id]
    expected_height = ORIGIN[2] + track["translation_world_m"][ids[0], 2]
    eligible = np.flatnonzero(
        (original["time_s"] >= 5)
        & (original["time_s"] <= 8)
        & (original["qpos"][:, object_adr + 2] >= expected_height)
    )
    if not len(eligible):
        raise ValueError("No validated in-hand state reaches the source segment's initial height")
    seed = int(eligible[0])
    d.qpos[:] = original["qpos"][seed]
    mujoco.mj_forward(m, d)
    active = original["active_actuators"]
    control = original["ctrl"][seed].copy()
    sites = [m.site(f"landmark_{i}").id for i in range(21)]
    initial_landmarks = d.site_xpos[sites].copy()
    object_adr = m.jnt_qposadr[m.joint("case_free").id]
    initial_position = d.qpos[object_adr : object_adr + 3].copy()
    initial_rotation = d.xmat[m.body("case").id].reshape(3, 3).copy()
    source_position = track["translation_world_m"][ids[0]]
    source_rotation = track["rotation_world"][ids[0]]
    lo, hi = m.actuator_ctrlrange[active[:7]].T
    window = min(7, len(ids) if len(ids) % 2 else len(ids) - 1)
    positions = savgol_filter(track["translation_world_m"][ids], window, 2, axis=0)
    raw_delta = Rotation.from_matrix(track["rotation_world"][ids] @ source_rotation.T).as_rotvec()
    smoothed = Rotation.from_rotvec(savgol_filter(raw_delta, window, 2, axis=0)).as_matrix()
    smoothed = smoothed @ smoothed[0].T
    source_position = positions[0].copy()
    qs = []
    cs = []
    errors = []
    for k, i in enumerate(ids):
        delta_rotation = SCENE_ROTATION @ smoothed[k] @ SCENE_ROTATION.T
        position = initial_position + SCENE_ROTATION @ (positions[k] - source_position)
        rotation = delta_rotation @ initial_rotation
        target = (initial_landmarks - initial_position) @ delta_rotation.T + position
        previous = control[active[:7]].copy()

        def residual(q):
            control[active[:7]] = q
            set_configuration(m, d, control)
            return np.r_[(d.site_xpos[sites] - target).ravel(), 0.001 * (q - previous)]

        result = least_squares(
            residual, np.clip(previous, lo + 1e-6, hi - 1e-6), bounds=(lo + 1e-7, hi - 1e-7), max_nfev=60
        )
        residual(result.x)
        errors.append(float(np.linalg.norm(d.site_xpos[sites] - target, axis=1).mean()))
        case_pose(m, d, position, rotation)
        qs.append(d.qpos.copy())
        cs.append(control.copy())
    source_times = track["timestamp_s"][ids] - track["timestamp_s"][ids[0]]
    max_velocity = float(
        np.max(np.abs(np.diff(np.asarray(cs)[:, active[:7]], axis=0)) / np.diff(source_times)[:, None])
    )
    retime = max(a.retime, max_velocity / 1.0)
    times = source_times * retime
    grid = np.arange(0, times[-1] + 1e-8, 0.02)
    qposes = np.column_stack([np.interp(grid, times, np.asarray(qs)[:, j]) for j in range(m.nq)])
    quat = np.asarray(qs)[:, object_adr + 3 : object_adr + 7][:, [1, 2, 3, 0]]
    qposes[:, object_adr + 3 : object_adr + 7] = Slerp(times, Rotation.from_quat(quat))(grid).as_quat()[
        :, [3, 0, 1, 2]
    ]
    controls = np.column_stack([np.interp(grid, times, np.asarray(cs)[:, j]) for j in range(m.nu)])
    qvel = np.zeros((len(grid), m.nv))
    for i in range(1, len(grid)):
        mujoco.mj_differentiatePos(m, qvel[i], 0.02, qposes[i - 1], qposes[i])
    qvel[0] = qvel[1]
    # Initial state is the actual successful grasp, not an imposed penetrated IK pose.
    qposes[0] = original["qpos"][seed]
    qvel[0] = 0
    np.savez_compressed(
        a.output / "reference.npz",
        qpos=qposes,
        qvel=qvel,
        ctrl=controls,
        time_s=grid,
        active_actuators=active,
        source_frames=ids,
    )
    meta = json.loads((a.bundle / "manifest.json").read_text())
    meta.update(
        reference="Trusted video object-pose segment with grasp-conditioned arm IK; initialized in hand",
        source_frames=ids.tolist(),
        source_retime_factor=retime,
        source_requested_minimum_retime=a.retime,
        automatic_reference_arm_velocity_limit_rad_s=1.0,
        initial_grasp_time_s=float(original["time_s"][seed]),
        smoothing="Savitzky-Golay, seven source samples, order two; positions and relative rotation vectors",
        mean_smoothing_position_change_m=float(
            np.linalg.norm(positions - track["translation_world_m"][ids], axis=1).mean()
        ),
        reference_interpolation="Control proposals only; original observations and validity flags unchanged",
        initial_grasp_source="Validated in-hand baseline state selected by source object height; originally conditioned on frame-240 hand",
        holds_grip_shape=True,
        has_prescribed_approach_carry_release_phases=False,
        mean_kinematic_target_error_m=float(np.mean(errors)),
        maximum_kinematic_target_error_m=float(max(errors)),
        full_video_imitation=False,
    )
    (a.output / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: meta[k]
                for k in [
                    "reference",
                    "mean_kinematic_target_error_m",
                    "maximum_kinematic_target_error_m",
                    "source_retime_factor",
                    "initial_grasp_time_s",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
