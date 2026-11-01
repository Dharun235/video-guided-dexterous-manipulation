"""Centre the native finger row over the case and construct an open-to-wrap prior.

This is a calibrated Book5 adaptation of the saved RH56 palm path, not a claim of
recovering hidden contact forces from RGB. Actual load is verified in simulation.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from humanoid.book_g1 import make_model
from humanoid.kinematics import control_vector, set_configuration
from video_g1.ik import fit_palm
from video_g1.provenance import write_manifest


def build(scene, source, output):
    """Fit the centred palm path and coordinated five-finger approach."""
    output.mkdir(parents=True, exist_ok=False)
    model, active, _ = make_model(scene, output, scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    case_axis = data.xmat[model.body("case").id].reshape(3, 3)[:, 0].copy()
    hand = model.body("R_hand_base_link").id
    original = dict(np.load(source))
    controls = []
    errors = []
    for arm, index, thumb in zip(
        original["arm_commands_rad"], original["index_flexion_rad"], original["thumb_flexion_rad"]
    ):
        action = np.r_[arm, [0.0, 0.0, 0.0, index, thumb, model.actuator_ctrlrange[active[12], 1]]]
        set_configuration(model, data, control_vector(model, active, action))
        position = data.xpos[hand].copy()
        rotation = data.xmat[hand].reshape(3, 3).copy()
        finger = action[7:].copy()
        # Native length differences need different pre-shapes; none is a parked fist.
        finger[:3] = [0.5 * action[10], 0.10 + 0.6 * action[10], 0.16 + 0.6 * action[10]]
        finger[5] = model.actuator_ctrlrange[active[12], 1]
        fitted, p_error, r_error = fit_palm(
            model, active, action, finger, position - 0.025 * case_axis, rotation, True
        )
        controls.append(fitted)
        errors.append([p_error, r_error])
    original = dict(
        time_s=original["time_s"],
        controls=np.array(controls),
        action_names=[model.actuator(int(i)).name for i in active],
        grasp_fingers=5,
        grasp_force_target_N=1.2,
    )
    np.savez_compressed(output / "reference.npz", **original)
    report = {
        "source": "RH56-derived palm path with native five-finger wrap pre-shapes",
        "centering_offset_m": 0.025,
        "force_targets_N": [1.2, 0.48, 0.36, 0.24, 0.12],
        "maximum_palm_position_error_m": float(np.max(np.array(errors)[:, 0])),
        "maximum_palm_rotation_error_rad": float(np.max(np.array(errors)[:, 1])),
        "physical_success_claimed_by_this_stage": False,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_manifest(output, [source, scene / "scene.xml"])


def main():
    """Build a native five-finger proposal; physical verification is a separate stage."""
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--source", type=Path, default=Path("data/book5/inputs/grasp-prior.npz"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    build(args.scene, args.source, args.output)


if __name__ == "__main__":
    main()
