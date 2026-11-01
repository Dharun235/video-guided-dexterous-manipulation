"""Compose source-aligned gestures, native grasp, carried motion and release.

This controller constructs proposals from reviewed source phases and retains raw
observations separately. Only robot motor commands act on the physical plant.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from humanoid.kinematics import control_vector, set_configuration
from video_g1.contacts import ContactReadout, opposition
from video_g1.provenance import write_manifest


@dataclass
class ContactGate:
    """Require persistent opposition on one external body before carrying it.

    Body identifiers are used for continuity, never interpreted as case labels.
    Contact loss stops arm/waist motion; it does not move or attach the object.
    """

    stable_s: float = 0.0
    lost_s: float = 0.0
    body: int | None = None
    confirmed: bool = False
    aborted: bool = False

    def update(self, qualities, dt, carrying=False):
        """Accumulate 150 ms confirmation/loss windows using external contact only."""
        if not self.confirmed:
            best = max(qualities, key=qualities.get) if qualities else None
            if best is not None and qualities[best] > 0.5:
                if self.body != best:
                    self.stable_s = 0.0
                self.body = best
                self.stable_s += dt
                self.confirmed = self.stable_s >= 0.15
            else:
                self.stable_s = 0.0
        elif carrying:
            self.lost_s = self.lost_s + dt if qualities.get(self.body, 0.0) <= 0.5 else 0.0
            self.aborted |= self.lost_s > 0.15
        return self.confirmed and not self.aborted


def contact_requirements(vectors, fingers, confirmation_fingers, targets):
    """Select an opposed external body; require load on every requested finger.

    A finger pressing a different body, such as the book, cannot satisfy the
    grasp. No body identifier is interpreted as an object-class label.
    """
    qualities = {
        b: opposition(v, minimum_force=min(0.3, 0.4 / max(len(fingers) - 1, 1))) for b, v in vectors.items()
    }
    candidate = max(qualities, key=qualities.get) if qualities and max(qualities.values()) > 0.5 else None
    minimum = np.maximum(0.04, 0.4 * targets[: len(confirmation_fingers)])
    loaded = {
        b: q if np.all(np.linalg.norm(vectors[b][confirmation_fingers], axis=1) > minimum) else 0.0
        for b, q in qualities.items()
    }
    return candidate, loaded


def force_step(current, measured, target, dt, maximum):
    """Close toward contact, then regulate pressure without latching an unloaded finger."""
    error = target - measured
    rate = np.clip(0.5 * error, -0.08, 0.22)
    if measured < min(0.02, 0.25 * target):
        rate = 0.2
    if abs(error) < min(0.08, 0.15 * target):
        rate = 0.0
    return min(current + rate * dt, maximum)


def simulate(model, active, reference, output, force_target=None, feedback=True):
    """Execute all source frames with external-contact feedback and no object forcing."""
    output.mkdir(parents=True, exist_ok=False)
    reference = dict(reference)
    force_target = float(reference.get("grasp_force_target_N", 0.8)) if force_target is None else force_target
    data = mujoco.MjData(model)
    times = reference["execution_time_s"]
    actions = reference["actions"]
    frames = reference["source_frame"]
    case = model.body("case").id
    hand = model.body("R_hand_base_link").id
    dof = model.jnt_dofadr[model.joint("case_free").id]
    robot = np.ones(model.nv, bool)
    robot[dof : dof + 6] = False
    data.ctrl[:] = control_vector(model, active, actions[0])
    waist = reference["extra_actuators"]
    data.ctrl[waist] = reference["extra_actions"][0]
    set_configuration(model, data, data.ctrl)
    mujoco.mj_forward(model, data)
    reader = ContactReadout(model)
    control = actions[0].copy()
    phases = reference.get("control_phase_frames", [310, 330, 350, 600, 610])
    force_start, carry_start, confirmation_deadline, carry_end, force_end = phases
    fingers = np.asarray(reference.get("grasp_finger_ids", [0, 1]), dtype=int)
    confirmation_fingers = np.asarray(reference.get("confirmation_finger_ids", fingers), dtype=int)
    target_forces = np.array(
        [force_target if i == 0 else force_target / max(len(fingers) - 1, 1) for i in fingers]
    )
    if len(fingers) == 5:
        target_forces = force_target * np.array([1, 0.4, 0.3, 0.2, 0.1])
    held = np.zeros(5, bool)
    gate = ContactGate()
    carry_snapshot = None
    extra_stream = []
    gate_rows = []
    extra_states, extra_velocities = [], []
    extra_joints = model.actuator_trnid[waist, 0]
    lo, hi = model.actuator_ctrlrange[active].T
    records, stream, stream_times = [], [], []
    sites = [model.site(f"landmark_{i}").id for i in range(21)]
    states = reference["state_joints"]
    if len(times) != len(frames) or actions.shape != (len(frames), len(active)):
        raise ValueError("Source and control timelines must align")
    if (
        not np.isfinite(actions).all()
        or not np.isfinite(reference["extra_actions"]).all()
        or not (np.diff(times) > 0).all()
    ):
        raise ValueError("Controls must be finite and execution time strictly increasing")
    for goal in times:
        while data.time + 1e-10 < goal:
            sf = np.interp(data.time, times, frames)
            desired = np.array([np.interp(data.time, times, actions[:, j]) for j in range(13)])
            extra = np.array(
                [np.interp(data.time, times, reference["extra_actions"][:, j]) for j in range(len(waist))]
            )
            if force_start <= sf < force_end:
                touches, vectors = reader.read(data)
                candidate_body, confirmed_qualities = contact_requirements(
                    vectors, fingers, confirmation_fingers, target_forces
                )
                allowed = gate.update(
                    confirmed_qualities,
                    model.opt.timestep,
                    carrying=carry_start <= sf < carry_end,
                )
                if sf > confirmation_deadline and not gate.confirmed:
                    gate.aborted = True
                if sf >= carry_start and not allowed:
                    if carry_snapshot is None:
                        carry_snapshot = np.r_[control[:7], data.ctrl[waist]]
                    desired[:7] = carry_snapshot[:7]
                    extra = carry_snapshot[7:]
                else:
                    carry_snapshot = None
            if feedback and force_start <= sf < force_end:
                ramp = np.clip((sf - force_start) / (carry_start - force_start), 0, 1)
                for finger, target_force in zip(fingers, target_forces):
                    motor = [11, 10, 9, 8, 7][finger]
                    desired[motor] += ramp * (hi[motor] - desired[motor])
                    if len(fingers) > 2:
                        # A support-plane touch must not latch an unloaded grasp finger.
                        measured = (
                            np.linalg.norm(vectors[candidate_body][finger])
                            if candidate_body is not None
                            else touches[finger, 0]
                        )
                        maximum = min(hi[motor], 0.9) if finger > 0 else hi[motor]
                        desired[motor] = force_step(
                            control[motor], measured, target_force, model.opt.timestep, maximum
                        )
                        continue
                    if touches[finger, 0] >= target_force:
                        held[finger] = True
                    if held[finger]:
                        desired[motor] = control[motor]
            if gate.aborted and carry_snapshot is not None:
                desired[:7] = carry_snapshot[:7]
                extra = carry_snapshot[7:]
            desired = np.clip(desired, lo, hi)
            control += np.clip(desired - control, -1.2 * model.opt.timestep, 1.2 * model.opt.timestep)
            data.ctrl[:] = control_vector(model, active, control)
            data.ctrl[waist] = extra
            extra_stream.append(extra.copy())
            data.qfrc_applied[:] = 0
            data.qfrc_applied[robot] = data.qfrc_bias[robot]
            stream.append(control.copy())
            stream_times.append(data.time)
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all():
                raise RuntimeError("Nonfinite physical state")
        mujoco.mj_forward(model, data)
        external, vectors = reader.read(data)
        gate_rows.append([gate.confirmed, gate.aborted])
        extra_states.append(data.qpos[model.jnt_qposadr[extra_joints]].copy())
        extra_velocities.append(data.qvel[model.jnt_dofadr[extra_joints]].copy())
        records.append(
            (
                data.time,
                data.qpos.copy(),
                data.qpos[model.jnt_qposadr[states]].copy(),
                data.qvel[model.jnt_dofadr[states]].copy(),
                data.site_xpos[sites].copy(),
                external,
                opposition(vectors.get(case, np.zeros((5, 3)))),
                data.xpos[case].copy(),
                data.xmat[case].reshape(3, 3).copy(),
                data.xpos[hand].copy(),
                data.xmat[hand].reshape(3, 3).copy(),
                control.copy(),
                np.linalg.norm(vectors.get(case, np.zeros((5, 3))), axis=1),
            )
        )
    arrays = {
        "source_frame": frames,
        "source_time_s": reference["source_time_s"],
        "execution_time_s": np.array([r[0] for r in records]),
        "qpos": np.array([r[1] for r in records]),
        "joint_states": np.array([r[2] for r in records]),
        "joint_velocities": np.array([r[3] for r in records]),
        "landmarks_m": np.array([r[4] for r in records]),
        "external_finger_forces_N": np.array([r[5] for r in records]),
        "case_opposition": np.array([r[6] for r in records]),
        "case_position_m": np.array([r[7] for r in records]),
        "case_rotation": np.array([r[8] for r in records]),
        "hand_position_m": np.array([r[9] for r in records]),
        "hand_rotation": np.array([r[10] for r in records]),
        "actions": np.array([r[11] for r in records]),
        "case_finger_resultant_normal_N": np.array([r[12] for r in records]),
        "executed_controls": np.array(stream),
        "control_time_s": np.array(stream_times),
        "extra_executed_controls": np.array(extra_stream),
        "extra_actuators": waist,
        "contact_gate": np.array(gate_rows, dtype=bool),
        "observed_hand": reference["observed_hand"],
        "extra_joint_states": np.array(extra_states),
        "extra_joint_velocities": np.array(extra_velocities),
        "state_names": np.array([model.joint(int(i)).name for i in states]),
        "extra_state_names": np.array([model.joint(int(i)).name for i in extra_joints]),
        "action_names": np.array([model.actuator(int(i)).name for i in active]),
        "extra_action_names": np.array([model.actuator(int(i)).name for i in waist]),
    }
    np.savez_compressed(output / "rollout.npz", **arrays)
    return arrays


def main():
    """Execute a separately generated full-timeline motor reference."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--no-feedback", action="store_true", help="Negative control; still enforces the contact gate"
    )
    args = parser.parse_args()
    reference = dict(np.load(args.reference / "reference.npz"))
    model = mujoco.MjModel.from_xml_path(str((args.reference / "scene.xml").resolve()))
    arrays = simulate(
        model, reference["active_actuators"], reference, args.output, feedback=not args.no_feedback
    )
    write_manifest(args.output, [args.reference / "reference.npz", args.reference / "scene.xml"])
    (args.output / "scene-path.txt").write_text(str((args.reference / "scene.xml").resolve()))
    report = score(reference, arrays)
    report["force_feedback_enabled"] = not args.no_feedback
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["full_sequence_success"]:
        raise SystemExit("Full-sequence checks failed; inspect report.json")


def score(reference, arrays):
    """Score carried pose tracking, rotation, release and stable placement separately.

    Targets are embodiment-adapted proposals, not monocular ground truth. A task
    pass never certifies exact video reconstruction or statistical robustness.
    """
    ids = arrays["source_frame"]
    carried = (ids >= 350) & (ids <= 595)
    final = ids >= 700
    positions = arrays["case_position_m"]
    rotations = arrays["case_rotation"]
    position_error = np.linalg.norm(positions - reference["target_case_position_m"], axis=1)
    rotation_error = Rotation.from_matrix(
        np.swapaxes(reference["target_case_rotation"], 1, 2) @ rotations
    ).magnitude()
    turn = Rotation.from_matrix(rotations @ rotations[0].T).magnitude()
    final_error = float(np.linalg.norm(positions[-1] - reference["target_case_position_m"][-1]))
    drift = float(np.max(np.linalg.norm(positions[final] - positions[final][0], axis=1)))
    released = bool(np.max(arrays["external_finger_forces_N"][final, :, 0]) < 0.08)
    support_fraction = float(np.mean(arrays["case_opposition"][carried] > 0.5))
    lift = float(np.max(positions[:, 2] - positions[0, 2]))
    from video_g1.geometry import palm_frame

    gesture = (ids <= 180) & reference["observed_hand"]
    gesture_error = [
        Rotation.from_matrix(
            palm_frame(reference["target_landmarks_m"][i]).T @ palm_frame(arrays["landmarks_m"][i])
        ).magnitude()
        for i in np.flatnonzero(gesture)
    ]
    final_gesture = (ids >= 680) & reference["observed_hand"]
    final_gesture_error = (
        [
            Rotation.from_matrix(
                palm_frame(reference["planned_landmarks_m"][i]).T @ palm_frame(arrays["landmarks_m"][i])
            ).magnitude()
            for i in np.flatnonzero(final_gesture)
        ]
        if "planned_landmarks_m" in reference
        else [np.nan]
    )
    full_wrap = len(reference.get("grasp_finger_ids", [0, 1])) == 5
    loads = arrays.get("case_finger_resultant_normal_N")
    review = (ids >= 405) & (ids <= 570)
    fractions = (loads[review] > 0.05).mean(axis=0) if loads is not None else np.zeros(5)
    approach = (ids >= 225) & (ids <= 255)
    pre_curl = float(np.max(arrays["actions"][approach, 7:11]))
    wrap_ok = bool(full_wrap and pre_curl < 0.5 and np.all(fractions > 0.95))
    timeline_ok = np.array_equal(ids, np.arange(748))
    valid = bool(
        timeline_ok
        and lift > 0.05
        and turn.max() > np.deg2rad(100)
        and support_fraction > 0.99
        and position_error[carried].mean() < 0.02
        and position_error[carried].max() < 0.04
        and rotation_error[carried].mean() < np.deg2rad(5)
        and rotation_error[carried].max() < np.deg2rad(20)
        and final_error < 0.01
        and drift < 0.002
        and released
        and not arrays["contact_gate"][:, 1].any()
    )
    return {
        "grasp_style": "five-finger object wrap" if full_wrap else "pinch baseline",
        "wrap_grasp_checks_passed": wrap_ok,
        "turn_case_load_fraction_by_finger": dict(
            zip(["thumb", "index", "middle", "ring", "pinky"], fractions.tolist())
        ),
        "precontact_maximum_finger_flexion_rad": pre_curl,
        "source_frames": len(ids),
        "all_source_frames_preserved": bool(timeline_ok),
        "duration_s": float(arrays["execution_time_s"][-1]),
        "maximum_lift_m": lift,
        "maximum_case_turn_deg": float(np.rad2deg(turn.max())),
        "carried_opposition_fraction": support_fraction,
        "final_gesture_mean_palm_tracking_error_deg": float(np.rad2deg(np.mean(final_gesture_error))),
        "final_gesture_maximum_palm_tracking_error_deg": float(np.rad2deg(np.max(final_gesture_error))),
        "initial_gesture_mean_palm_error_deg": float(np.rad2deg(np.mean(gesture_error))),
        "initial_gesture_maximum_palm_error_deg": float(np.rad2deg(np.max(gesture_error))),
        "carried_mean_position_error_m": float(position_error[carried].mean()),
        "carried_maximum_position_error_m": float(position_error[carried].max()),
        "carried_mean_rotation_error_deg": float(np.rad2deg(rotation_error[carried].mean())),
        "carried_maximum_rotation_error_deg": float(np.rad2deg(rotation_error[carried].max())),
        "final_position_error_m": final_error,
        "final_rotation_error_deg": float(np.rad2deg(rotation_error[-1])),
        "final_position_drift_m": drift,
        "released_without_finger_contact": released,
        "contact_loss_abort": bool(arrays["contact_gate"][:, 1].any()),
        "grasp_confirmed": bool(arrays["contact_gate"][:, 0].any()),
        "physical_task_success": valid,
        "full_sequence_success": bool(
            valid
            and np.rad2deg(np.mean(gesture_error)) < 10
            and np.rad2deg(np.max(gesture_error)) < 30
            and np.rad2deg(np.mean(final_gesture_error)) < 10
            and np.rad2deg(np.max(final_gesture_error)) < 30
            and (not full_wrap or wrap_ok)
        ),
        "initial_case_drift_m": float(np.max(np.linalg.norm(positions[ids <= 180] - positions[0], axis=1))),
        "exact_video_imitation_verified": False,
        "object_forces_or_welds": False,
        "robot_bias_compensation": True,
        "waist_joints_used": 3,
        "pelvis_supported": True,
        "target_provenance": "Reviewed video phases, observed hand turn, fitted finger gestures, source-derived placement; adapted workspace, timing and clearance",
        "ready_for_training": False,
    }


if __name__ == "__main__":
    main()
