"""Execute the full reference with a free object and record frame-aligned evidence."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from humanoid.kinematics import set_configuration
from manipulation.control import external_contacts
from manipulation.fit import control_vector

from .provenance import write_manifest


def run(reference_path, output):
    """Apply only actuator targets and robot bias compensation after a single reset."""
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    write_manifest(output, [reference_path / "reference.npz", reference_path / "scene.xml"])
    reference = np.load(reference_path / "reference.npz")
    model = mujoco.MjModel.from_xml_path(str((reference_path / "scene.xml").resolve()))
    data = mujoco.MjData(model)
    active = reference["active_actuators"]
    state = reference["state_joints"]
    actions = reference["actions"]
    times = reference["execution_time_s"]
    joint = model.joint("case_free").id
    object_dof = model.jnt_dofadr[joint]
    robot = np.ones(model.nv, bool)
    robot[object_dof : object_dof + 6] = False
    data.ctrl[:] = control_vector(model, active, actions[0])
    set_configuration(model, data, data.ctrl)
    mujoco.mj_forward(model, data)
    sites = [model.site(f"landmark_{i}").id for i in range(21)]
    states = []
    velocities = []
    positions = []
    landmarks = []
    forces = []
    commands = []
    actual_times = []
    executed_controls = []
    executed_times = []
    for time_goal in times:
        while data.time + 1e-10 < time_goal:
            command = np.array([np.interp(data.time, times, actions[:, j]) for j in range(actions.shape[1])])
            data.ctrl[:] = control_vector(model, active, command)
            data.qfrc_applied[:] = 0
            data.qfrc_applied[robot] = data.qfrc_bias[robot]
            executed_controls.append(data.ctrl[active].copy())
            executed_times.append(data.time)
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all():
                raise RuntimeError("Non-finite simulator state")
        mujoco.mj_forward(model, data)
        states.append(data.qpos[model.jnt_qposadr[state]].copy())
        velocities.append(data.qvel[model.jnt_dofadr[state]].copy())
        positions.append(data.qpos.copy())
        landmarks.append(data.site_xpos[sites].copy())
        forces.append(external_contacts(model, data))
        commands.append(data.ctrl[active].copy())
        actual_times.append(data.time)
    landmarks = np.array(landmarks)
    error = np.linalg.norm(landmarks - reference["target_landmarks_m"], axis=2)
    np.savez_compressed(
        output / "rollout.npz",
        source_frame=reference["source_frame"],
        source_time_s=reference["source_time_s"],
        execution_time_s=actual_times,
        joint_states=states,
        joint_velocities=velocities,
        actions=commands,
        qpos=positions,
        landmarks_m=landmarks,
        external_finger_forces_N=forces,
        observed_hand=reference["observed_hand"],
        executed_controls=executed_controls,
        control_time_s=executed_times,
    )
    report = {
        "source_frames": len(times),
        "all_source_frames_preserved": True,
        "initialization": "Robot pose set once; case starts on book",
        "object_pose_playback": False,
        "object_forces_or_welds": False,
        "robot_bias_compensation": True,
        "controller": "Open-loop fitted motor targets; physical feedback not enabled in this baseline",
        "mean_wrist_error_m": float(np.nanmean(error[:, 0])),
        "mean_tip_error_m": float(np.nanmean(error[:, [4, 8, 12, 16, 20]])),
        "maximum_joint_tracking_error_rad": float(
            np.max(
                abs(
                    np.array(commands)
                    - np.array(states)[:, [list(state).index(model.actuator_trnid[a, 0]) for a in active]]
                )
            )
        ),
        "duration_s": float(data.time),
        "elapsed_s": time.monotonic() - started,
        "physical_success_verified": False,
        "ready_for_training": False,
        "mujoco_version": mujoco.__version__,
    }
    (output / "scene-path.txt").write_text(str((reference_path / "scene.xml").resolve()))
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    """Record complete source-aligned physical execution without hidden object control."""
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.reference, a.output)


if __name__ == "__main__":
    main()
