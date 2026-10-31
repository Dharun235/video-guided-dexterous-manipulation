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
from humanoid.kinematics import set_configuration
from manipulation.fit import control_vector
from video_g1.provenance import write_manifest
from video_g1.rh56_native import fit_palm


def build(scene, source, output):
    """Keep native geometry while replacing parked fingers with a shared wrap approach."""
    output.mkdir(parents=True, exist_ok=False)
    model, active, _ = make_model(scene, output, scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    case_axis = data.xmat[model.body("case").id].reshape(3, 3)[:, 0].copy()
    hand = model.body("R_hand_base_link").id
    original = dict(np.load(source))
    controls = []
    errors = []
    for action in original["controls"]:
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
    original.update(controls=np.array(controls), grasp_fingers=5, grasp_force_target_N=1.2)
    np.savez_compressed(output / "reference.npz", **original)
    report = {
        "source": "Saved native RH56 width plan; original finger parking replaced",
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
    p.add_argument("--source", type=Path, default=Path("examples/rh56-evaluation/reference.npz"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    build(args.scene, args.source, args.output)


if __name__ == "__main__":
    main()
