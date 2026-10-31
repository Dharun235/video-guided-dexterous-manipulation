"""Verify normal opposition, lift and slip using native free-object dynamics.

The contact gate considers any shared external body, without a case label.
Case identity appears only in scoring. The robot uses bias compensation; the
object receives neither external forces nor attachments.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from humanoid.book_g1 import make_model
from humanoid.kinematics import set_configuration
from manipulation.control import ContactFeedback, finger_geoms
from manipulation.fit import control_vector
from manipulation.metrics import longest_interval
from manipulation.rollout import arm_target

from .provenance import write_manifest


class ContactReadout:
    """Cache topology and return external forces plus forces directed onto each body."""

    def __init__(self, model):
        self.model = model
        self.fingers = {g: i for i, geoms in enumerate(finger_geoms(model)) for g in geoms}
        pelvis = model.body("pelvis").id
        self.robot = set()
        for i in range(1, model.nbody):
            parent = i
            while parent:
                if parent == pelvis:
                    self.robot.add(i)
                    break
                parent = int(model.body_parentid[parent])

    def read(self, data):
        """Return (5,2) normal/shear sums and per-body finger normal-force vectors."""
        touches = np.zeros((5, 2))
        vectors = {}
        for index, contact in enumerate(data.contact):
            if contact.geom1 in self.fingers:
                finger = self.fingers[contact.geom1]
                other = int(self.model.geom_bodyid[contact.geom2])
                sign = 1
            elif contact.geom2 in self.fingers:
                finger = self.fingers[contact.geom2]
                other = int(self.model.geom_bodyid[contact.geom1])
                sign = -1
            else:
                continue
            if other in self.robot:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, index, force)
            normal = max(float(force[0]), 0)
            touches[finger] += [normal, np.linalg.norm(force[1:3])]
            vectors.setdefault(other, np.zeros((5, 3)))[finger] += sign * normal * contact.frame[:3]
        return touches, vectors


def opposition(vectors, minimum_force=0.08):
    """Return thumb/index-or-middle opposition in [0,1]; parallel pressure is zero."""
    lengths = np.linalg.norm(vectors, axis=1)
    if lengths[0] <= minimum_force:
        return 0.0
    quality = [
        max(0.0, -np.dot(vectors[0], vectors[i]) / (lengths[0] * lengths[i]))
        for i in [1, 2]
        if lengths[i] > minimum_force
    ]
    return float(np.clip(max(quality, default=0.0), 0, 1))


def verify(model, active, state, grasp, duration=9.0, gated=True):
    """Execute closure, require sustained normal opposition, then attempt a small lift."""
    data = mujoco.MjData(model)
    workspace = mujoco.MjData(model)
    set_configuration(model, workspace, control_vector(model, active, grasp))
    mujoco.mj_forward(model, workspace)
    sites = [model.site(f"landmark_{i}").id for i in range(21)]
    high = arm_target(model, active, grasp, workspace.site_xpos[sites].copy(), np.array([0, 0, 0.05]))
    initial = grasp.copy()
    initial[:7] = high
    initial[7:11] = np.maximum(0, grasp[7:11] - 0.35)
    initial[11] = max(0, grasp[11] - 0.15)
    controls = initial.copy()
    data.ctrl[:] = control_vector(model, active, controls)
    set_configuration(model, data, data.ctrl)
    mujoco.mj_forward(model, data)
    case = model.body("case").id
    hand = model.body("R_hand_base_link").id
    joint = model.joint("case_free").id
    adr = model.jnt_qposadr[joint]
    dof = model.jnt_dofadr[joint]
    robot = np.ones(model.nv, bool)
    robot[dof : dof + 6] = False
    origin = data.qpos[adr : adr + 3].copy()
    reader = ContactReadout(model)
    feedback = ContactFeedback()
    stable = 0.0
    confirmed = None
    records = []
    command_stream = []
    aborted = False
    lost = 0.0
    for step in range(int(duration / model.opt.timestep)):
        t = data.time
        touches, vectors = reader.read(data)
        quality = max((opposition(v, minimum_force=0.3) for v in vectors.values()), default=0.0)
        stable = stable + model.opt.timestep if quality > 0.5 else 0.0
        if t >= 3.5 and stable >= 0.15 and confirmed is None:
            confirmed = float(t)
        start = max(5.0, confirmed) if confirmed is not None else None
        if start is not None and t > start and gated:
            lost = lost + model.opt.timestep if quality <= 0.5 else 0.0
            if lost > 0.15:
                aborted = True

        def smooth(first, last, t=t):
            x = np.clip((t - first) / (last - first), 0, 1)
            return x * x * (3 - 2 * x)

        desired = grasp.copy()
        if t < 2:
            desired = initial.copy()
            s = smooth(0, 2)
            desired[:7] = high * (1 - s) + grasp[:7] * s
        elif t < 4:
            s = smooth(2, 4)
            desired[7:] = initial[7:] * (1 - s) + grasp[7:] * s
        elif (not gated or start is not None) and not aborted:
            lift_start = 5.0 if not gated else start
            s = smooth(lift_start, lift_start + 2)
            desired[:7] = grasp[:7] * (1 - s) + high * s
        offset = feedback.update(touches, model.opt.timestep, enabled=t >= 3.5)
        desired[[7, 8, 9, 10]] += offset[[4, 3, 2, 1]]
        desired[11] += 0.4 * offset[0]
        desired = np.clip(desired, *model.actuator_ctrlrange[active].T)
        controls += np.clip(desired - controls, -1.2 * model.opt.timestep, 1.2 * model.opt.timestep)
        data.ctrl[:] = control_vector(model, active, controls)
        data.qfrc_applied[:] = 0
        data.qfrc_applied[robot] = data.qfrc_bias[robot]
        command_stream.append(controls.copy())
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError("Nonfinite physical state")
        if step % 20 == 0:
            mujoco.mj_forward(model, data)
            external, forces = reader.read(data)
            case_quality = opposition(forces.get(case, np.zeros((5, 3))))
            relative = data.xmat[hand].reshape(3, 3).T @ (data.qpos[adr : adr + 3] - data.xpos[hand])
            records.append(
                (
                    data.time,
                    data.qpos.copy(),
                    data.qvel.copy(),
                    controls.copy(),
                    external,
                    quality,
                    case_quality,
                    relative,
                    data.qpos[model.jnt_qposadr[state]].copy(),
                    data.xmat[hand].reshape(3, 3).T @ data.xmat[case].reshape(3, 3),
                )
            )
    times = np.array([r[0] for r in records])
    qpos = np.array([r[1] for r in records])
    lift = qpos[:, adr + 2] - origin[2]
    case_quality = np.array([r[6] for r in records])
    supported = (case_quality > 0.5) & (lift > 0.025)
    supported_time = longest_interval(supported, 0.02)
    relative = np.array([r[7] for r in records])
    selected = np.flatnonzero(supported)
    slip = (
        float(np.max(np.linalg.norm(relative[selected] - relative[selected[0]], axis=1)))
        if len(selected)
        else None
    )
    relative_rotation = np.array([r[9] for r in records])
    angular_slip = (
        float(
            np.max(
                Rotation.from_matrix(
                    relative_rotation[selected[0]].T @ relative_rotation[selected]
                ).magnitude()
            )
        )
        if len(selected)
        else None
    )
    final_hold = bool(supported[-25:].all())
    report = {
        "gate_enabled": gated,
        "normal_opposition_threshold": 0.5,
        "required_stable_contact_s": 0.15,
        "gate_minimum_normal_force_N": 0.3,
        "contact_confirmed_time_s": confirmed,
        "lift_aborted_after_contact_loss": aborted,
        "maximum_lift_m": float(lift.max()),
        "normal_opposition_supported_lift_s": supported_time,
        "relative_slip_during_supported_lift_m": slip,
        "angular_slip_during_supported_lift_rad": angular_slip,
        "final_half_second_supported": final_hold,
        "success": bool(
            supported_time >= 0.5
            and slip is not None
            and slip < 0.01
            and angular_slip is not None
            and angular_slip < np.deg2rad(10)
            and final_hold
            and not aborted
        ),
        "acceptance": {
            "lift_m": 0.025,
            "supported_duration_s": 0.5,
            "maximum_slip_m": 0.01,
            "maximum_angular_slip_deg": 10,
            "final_hold_s": 0.5,
        },
        "object_pose_set_only_at_reset": True,
        "object_attachments_or_forces": False,
        "robot_bias_compensation": True,
        "controller_knows_case_identity": False,
        "ready_for_training": False,
    }
    arrays = {
        "time_s": times,
        "qpos": qpos,
        "qvel": np.array([r[2] for r in records]),
        "actions": np.array([r[3] for r in records]),
        "external_finger_forces_N": np.array([r[4] for r in records]),
        "external_opposition": np.array([r[5] for r in records]),
        "case_opposition": case_quality,
        "object_in_hand_m": relative,
        "object_in_hand_rotation": relative_rotation,
        "joint_states": np.array([r[8] for r in records]),
        "executed_controls": np.array(command_stream),
    }
    return report, arrays


def main():
    """Compare ungated replay with the stricter contact gate for supplied grasp candidates."""
    p = argparse.ArgumentParser(__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--search", type=Path, help="Existing native-model candidate search directory")
    source.add_argument("--candidates", type=Path, help="Portable saved candidate controls JSON")
    p.add_argument("--scene", type=Path, help="Measured scene directory for portable candidates")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if a.search:
        model = mujoco.MjModel.from_xml_path(str((a.search / "scene.xml").resolve()))
        mapping = np.load(a.search / "grasp.npz")
        active, state = mapping["active"], mapping["state"]
        candidates = [
            json.loads(path.read_text())["actions"]
            for path in sorted(a.search.glob("candidate-*/report.json"))
        ]
    else:
        if a.scene is None:
            p.error("--scene is required with --candidates")
        model_folder = a.output / "model"
        model_folder.mkdir()
        model, active, state = make_model(a.scene, model_folder, a.scene)
        saved = json.loads(a.candidates.read_text())
        names = [model.actuator(int(i)).name for i in active]
        if names != saved["action_names"]:
            raise ValueError("Candidate command order differs from native model")
        candidates = [row["actions"] for row in saved["candidates"]]
    write_manifest(
        a.output,
        [a.candidates, a.scene / "scene.xml"]
        if a.candidates
        else [a.search / "grasp.npz", a.search / "scene.xml"],
    )
    if not candidates:
        raise ValueError("No grasp candidates supplied")
    rows = []
    for index, values in enumerate(candidates):
        grasp = np.asarray(values, dtype=float)
        if grasp.shape != (len(active),) or not np.isfinite(grasp).all():
            raise ValueError("Candidate must contain finite native joint targets")
        low, high = model.actuator_ctrlrange[active].T
        if (grasp < low).any() or (grasp > high).any():
            raise ValueError("Candidate exceeds native actuator limits")
        for gated in [False, True]:
            report, arrays = verify(model, active, state, grasp, gated=gated)
            report["candidate"] = index
            folder = a.output / (f"candidate-{index:02d}-" + ("gated" if gated else "replay"))
            folder.mkdir()
            np.savez_compressed(folder / "rollout.npz", **arrays)
            (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            rows.append(report)
            print(json.dumps(report), flush=True)
    (a.output / "report.json").write_text(
        json.dumps(
            {
                "tested": rows,
                "any_success": any(r["success"] for r in rows),
                "elapsed_s": time.monotonic() - started,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
