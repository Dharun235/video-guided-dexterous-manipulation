"""Independently verify a saved GPU-adapter command stream in CPU MuJoCo."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def main():
    """Reset once, apply every saved motor command and compare recorded states."""
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    reference = np.load(a.input / "reference.npz")
    saved = np.load(a.trajectory / "trajectory.npz")
    model = mujoco.MjModel.from_xml_path(str((a.input / "scene.xml").resolve()))
    data = mujoco.MjData(model)
    data.qpos[:] = reference["qpos"][0]
    data.qvel[:] = reference["qvel"][0]
    data.ctrl[:] = reference["ctrl"][0]
    mujoco.mj_forward(model, data)
    sample_steps = np.rint(saved["time_s"] / model.opt.timestep).astype(int) - 1
    rows = {int(step): i for i, step in enumerate(sample_steps)}
    error = np.full(len(sample_steps), np.nan)
    for step, command in enumerate(saved["executed_ctrl_2ms"]):
        data.ctrl[:] = command
        mujoco.mj_step(model, data)
        if step in rows:
            i = rows[step]
            error[i] = np.max(abs(data.qpos - saved["qpos"][i]))
    report = {
        "mujoco": mujoco.__version__,
        "recorded_states": len(error),
        "matched_sample_count": int(np.isfinite(error).sum()),
        "maximum_qpos_component_error": float(np.max(error)),
        "all_states_match": bool(np.isfinite(error).all() and np.max(error) < 1e-10),
        "object_pose_set_only_at_reset": True,
        "validates": "Command replay determinism, not manipulation success",
    }
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["all_states_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
