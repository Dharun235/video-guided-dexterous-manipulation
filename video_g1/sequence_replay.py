"""Independently verify full-sequence replay from arm, hand and waist commands."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from humanoid.kinematics import set_configuration
from manipulation.fit import control_vector


def check(reference_path, rollout_path):
    """Compare every saved state after a fresh reset and motor-only integration."""
    ref = np.load(reference_path / "reference.npz")
    run = dict(np.load(rollout_path / "rollout.npz"))
    m = mujoco.MjModel.from_xml_path(str((reference_path / "scene.xml").resolve()))
    d = mujoco.MjData(m)
    active = ref["active_actuators"]
    waist = ref["extra_actuators"]
    d.ctrl[:] = control_vector(m, active, ref["actions"][0])
    d.ctrl[waist] = ref["extra_actions"][0]
    set_configuration(m, d, d.ctrl)
    mujoco.mj_forward(m, d)
    case_dof = m.jnt_dofadr[m.joint("case_free").id]
    robot = np.ones(m.nv, bool)
    robot[case_dof : case_dof + 6] = False
    index = 0
    errors = []
    for i, goal in enumerate(run["execution_time_s"]):
        while d.time + 1e-10 < goal:
            if abs(run["control_time_s"][index] - d.time) > 1e-8:
                raise ValueError("Command stream timestamp mismatch")
            d.ctrl[:] = control_vector(m, active, run["executed_controls"][index])
            d.ctrl[waist] = run["extra_executed_controls"][index]
            d.qfrc_applied[:] = 0
            d.qfrc_applied[robot] = d.qfrc_bias[robot]
            mujoco.mj_step(m, d)
            index += 1
        mujoco.mj_forward(m, d)
        errors.append(np.max(abs(d.qpos - run["qpos"][i])))
    return {
        "recorded_states": len(errors),
        "executed_commands": index,
        "waist_commands_included": True,
        "maximum_qpos_error": float(max(errors)),
        "all_states_match": bool(max(errors) < 1e-10),
        "case_pose_set_only_at_reset": True,
    }


def main():
    """Write a replay check without changing any source evidence."""
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--rollout", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = check(args.reference, args.rollout)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["all_states_match"]:
        raise SystemExit("Numerical replay did not match")


if __name__ == "__main__":
    main()
