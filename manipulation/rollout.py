"""Actuator-only contact experiment. Object pose is set once at reset, then free."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from humanoid.kinematics import set_configuration

from .control import ContactFeedback, external_contacts, grasp_detected
from .fit import case_pose, control_vector
from .metrics import case_contacts, score


def arm_target(model, active, seed, landmarks, translation_m):
    """Fit seven arm targets for a translation, in metres, of the existing grasp."""
    workspace = mujoco.MjData(model)
    sites = [model.site(f"landmark_{j}").id for j in [0, 4, 8, 12]]
    lo, hi = model.actuator_ctrlrange[active[:7]].T

    def residual(arm):
        q = seed.copy()
        q[:7] = arm
        set_configuration(model, workspace, control_vector(model, active, q))
        return np.r_[
            (workspace.site_xpos[sites] - landmarks[[0, 4, 8, 12]] - translation_m).ravel(),
            0.002 * (arm - seed[:7]),
        ]

    result = least_squares(residual, seed[:7], bounds=(lo + 1e-6, hi - 1e-6), max_nfev=45)
    return result.x


def run(a):
    """Execute bounded motor targets with a free case and save states, contacts and metrics."""
    started = time.monotonic()
    a.output.mkdir(parents=True, exist_ok=False)
    m = mujoco.MjModel.from_xml_path(str((a.fit / "scene.xml").resolve()))
    case_id = m.body("case").id
    m.body_mass[case_id] *= a.mass_scale
    m.body_inertia[case_id] *= a.mass_scale
    m.geom_friction[:, 0] *= a.friction_scale
    d = mujoco.MjData(m)
    mujoco.mj_setConst(m, d)
    g = np.load(a.fit / "grasp.npz")
    active = g["active"]
    state = g["state"]
    grasp = g["actions"]
    position = g["position"]
    rotation = g["rotation"]
    # All kinematic operations are confined to reset and separate IK workspaces.
    initial = grasp.copy()
    initial[7:11] = np.maximum(0, initial[7:11] - 0.3)
    initial[11] = max(0, initial[11] - 0.12)
    joint = m.joint("case_free").id
    adr = m.jnt_qposadr[joint]
    dof = m.jnt_dofadr[joint]
    robot_dofs = np.ones(m.nv, dtype=bool)
    robot_dofs[dof : dof + 6] = False
    state_adr = m.jnt_qposadr[state]
    controller = ContactFeedback(target_force=a.force)
    task = json.loads(a.task.read_text()) if a.task else None
    lift = task["lift_height_m"] if task else 0.065
    goal = np.asarray(task["goal_position_world_m"]) if task else position + np.array([0, 0, 0.065])
    offsets_path = np.array(
        [
            [0, 0, 0.08],
            [0, 0, 0],
            [0, 0, lift],
            goal - position + [0, 0, lift],
            goal - position + [0, 0, 0.012],
            goal - position + [0, 0, 0.08],
        ]
    )
    target_arms = [arm_target(m, active, grasp, g["landmarks"], delta) for delta in offsets_path]
    initial[:7] = target_arms[0]
    set_configuration(m, d, control_vector(m, active, initial))
    case_pose(m, d, position, rotation)
    mujoco.mj_forward(m, d)
    contact_hold = 0.0
    contact_confirmed = False
    phases = []
    refiner = None
    mpc_delta = np.zeros(len(active))
    if a.mpc:
        from .mpc import PhysicsRefiner

        np.random.seed(0)

        def reference(t):
            times = [0, 5, 8, 12, 15, 17, 21]
            positions = np.vstack(
                [
                    position,
                    position,
                    position + offsets_path[2],
                    position + offsets_path[3],
                    position + offsets_path[4],
                    goal,
                    goal,
                ]
            )
            return np.array([np.interp(t, times, positions[:, i]) for i in range(3)])

        refiner = PhysicsRefiner(m, active, robot_dofs, reference, samples=a.mpc_samples)
    ctrl = initial.copy()
    records = []
    qposes = []
    all_forces = []
    case_forces = []
    relative = []
    joint_velocities = []
    torques = []
    feedforward = []
    grasp_steps = 0
    steps = int(a.duration / m.opt.timestep)
    sample_every = 20
    for step in range(steps):
        t = d.time

        # Smooth scheduled reference derived from the demonstration's source,
        # lift and destination; contact confirmation gates all transport.
        def blend(first, last, start, end):
            u = np.clip((t - start) / (end - start), 0, 1)
            u = u * u * (3 - 2 * u)
            return (1 - u) * first + u * last

        nominal = grasp.copy()
        if t < 2:
            phase = "approach"
            nominal = initial.copy()
            nominal[:7] = blend(target_arms[0], target_arms[1], 0, 2)
        elif t < 4:
            phase = "close"
            nominal[:7] = target_arms[1]
            nominal[7:] = blend(initial[7:], grasp[7:], 2, 4)
        elif t < 5 or not contact_confirmed:
            phase = "confirm"
            nominal[:7] = target_arms[1]
        elif t < 8:
            phase = "lift"
            nominal[:7] = blend(target_arms[1], target_arms[2], 5, 8)
        elif t < 12:
            phase = "carry"
            nominal[:7] = blend(target_arms[2], target_arms[3], 8, 12)
        elif t < 15:
            phase = "place"
            nominal[:7] = blend(target_arms[3], target_arms[4], 12, 15)
        elif t < 17:
            phase = "release"
            nominal[:7] = target_arms[4]
            nominal[7:] = blend(grasp[7:], np.r_[np.zeros(5), grasp[12]], 15, 17)
        else:
            phase = "retreat"
            nominal = initial.copy()
            nominal[7:12] = 0.0
            nominal[:7] = blend(target_arms[4], target_arms[5], 17, 20)
        touches = external_contacts(m, d)
        offsets = controller.update(touches, m.opt.timestep, enabled=4 <= t < 15 and not a.no_feedback)
        desired = nominal.copy()
        # Active order is pinky/ring/middle/index/thumb-pitch/thumb-yaw.
        desired[[7, 8, 9, 10]] += offsets[[4, 3, 2, 1]]
        desired[11] += 0.4 * offsets[0]
        desired = np.clip(desired, *m.actuator_ctrlrange[active].T)
        if refiner is not None and 8 <= t < 12:
            if step % 80 == 0:
                selected = refiner.improve(d, control_vector(m, active, desired))
                mpc_delta = selected - desired
            desired = np.clip(desired + mpc_delta, *m.actuator_ctrlrange[active].T)
        ctrl += np.clip(desired - ctrl, -a.velocity * m.opt.timestep, a.velocity * m.opt.timestep)
        d.ctrl[:] = control_vector(m, active, ctrl)
        # Robot bias-force compensation (gravity/Coriolis) only: never apply external force to the case.
        d.qfrc_applied[:] = 0
        d.qfrc_applied[robot_dofs] = d.qfrc_bias[robot_dofs]
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all():
            raise RuntimeError("Non-finite simulation state")
        if grasp_detected(touches):
            grasp_steps += 1
            if 4 <= t < 5:
                contact_hold += m.opt.timestep
        elif 4 <= t < 5:
            contact_hold = 0.0
        if contact_hold > 0.15:
            contact_confirmed = True
        if step % sample_every == 0:
            mujoco.mj_forward(m, d)
            qposes.append(d.qpos.copy())
            all_forces.append(external_contacts(m, d))
            phases.append(phase)
            case_forces.append(case_contacts(m, d))
            hand = m.body("R_hand_base_link").id
            relative.append(d.xmat[hand].reshape(3, 3).T @ (d.qpos[adr : adr + 3] - d.xpos[hand]))
            joint_velocities.append(d.qvel[m.jnt_dofadr[state]].copy())
            torques.append(d.actuator_force[active].copy())
            feedforward.append(d.qfrc_applied[m.jnt_dofadr[state]].copy())
            records.append(
                (
                    d.time,
                    d.qpos[state_adr].copy(),
                    ctrl.copy(),
                    d.qpos[adr : adr + 7].copy(),
                    d.qvel[dof : dof + 6].copy(),
                )
            )
    np.savez_compressed(
        a.output / "rollout.npz",
        time_s=[r[0] for r in records],
        joint_positions=[r[1] for r in records],
        actions=[r[2] for r in records],
        object_pose=[r[3] for r in records],
        object_velocity=[r[4] for r in records],
        touch=all_forces,
        qpos=qposes,
        phase=phases,
        case_touch=case_forces,
        object_in_hand_m=relative,
        joint_velocities=joint_velocities,
        actuator_force=torques,
        feedforward_joint_torque_Nm=feedforward,
    )
    z = np.array([r[3][2] for r in records])
    actual = np.array([r[3][:3] for r in records])
    report = dict(
        upstream_mpc_enabled=a.mpc,
        mpc_updates=refiner.updates if refiner else 0,
        mpc_mean_control_change_rad=float(np.mean(refiner.change)) if refiner and refiner.change else 0.0,
        feedback_enabled=not a.no_feedback,
        mass_scale=a.mass_scale,
        friction_scale=a.friction_scale,
        actuator_driven=True,
        object_pose_set_only_at_reset=True,
        hidden_attachment=False,
        robot_bias_force_compensation=True,
        duration_s=float(d.time),
        source_frame=int(json.loads((a.fit / "report.json").read_text())["source_frame"]),
        source_sha256=json.loads((a.fit / "report.json").read_text()).get("source_sha256"),
        maximum_object_lift_m=float(z.max() - position[2]),
        final_object_displacement_m=(actual[-1] - position).tolist(),
        opposing_contact_duration_s=grasp_steps * m.opt.timestep,
        maximum_finger_force_N=np.asarray(all_forces)[:, :, 0].max(0).tolist(),
        lifted=bool(z.max() - position[2] > 0.025),
        ready_for_training=False,
        contact_confirmed=contact_confirmed,
        goal_position_m=goal.tolist(),
        task_reference=task,
        limitations=[
            "Initial supported robot configuration is prescribed; walking/balance are not evaluated",
            "Estimated box collision geometry, mass/friction and scale are assumptions",
            "Contact feedback uses all external finger contacts",
            "Translation-task adaptation holds initial orientation; the recorded flip and early gestures are not reproduced",
            "Single nominal trial is not a generalization benchmark",
        ],
        elapsed_s=time.monotonic() - started,
    )
    report.update(
        score(
            [r[0] for r in records],
            [r[3] for r in records],
            [r[4] for r in records],
            case_forces,
            phases,
            position,
            goal,
            relative,
        )
    )
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (a.output / "scene-path.txt").write_text(str((a.fit / "scene.xml").resolve()))
    (a.output / "schema.json").write_text(
        json.dumps(
            dict(
                action_names=[m.actuator(int(i)).name for i in active],
                joint_names=[m.joint(int(i)).name for i in state],
                touch_fingers=["thumb", "index", "middle", "ring", "pinky"],
                touch_channels=["normal_N", "shear_N"],
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(report, indent=2))


def main():
    """Run the transfer or a configured contact/physics diagnostic."""
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--fit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", type=Path)
    p.add_argument("--duration", type=float, default=21)
    p.add_argument("--force", type=float, default=0.6)
    p.add_argument("--mass-scale", type=float, default=1.0)
    p.add_argument("--friction-scale", type=float, default=1.0)
    p.add_argument("--no-feedback", action="store_true")
    p.add_argument("--mpc", action="store_true")
    p.add_argument("--mpc-samples", type=int, default=12)
    p.add_argument("--velocity", type=float, default=0.8)
    args = p.parse_args()
    if (
        args.duration < 1
        or args.mass_scale <= 0
        or args.friction_scale <= 0
        or args.velocity <= 0
        or args.mpc_samples < 1
    ):
        p.error("Positive scales, velocity and sample count; duration >= 1 required")
    run(args)


if __name__ == "__main__":
    main()
