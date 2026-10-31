"""Adapt upstream width planning to the unchanged native Unitree Inspire geometry.

Only names and a rigid coordinate-frame transform change in the standalone FK
fixture. Native meshes, joint limits and URDF mimic equations remain intact.
"""

import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from humanoid.book_g1 import DRIVEN, HAND, convert_hand

ACT_NAMES = ["pinky", "ring", "middle", "index", "thumb_proximal", "thumb_yaw"]


def standalone_native(source_xml, output):
    """Export the native hand in the reference planner's coordinate frame for FK."""
    output.mkdir(parents=True, exist_ok=False)
    converted, body, mimics = convert_hand(output)
    source = ET.parse(source_xml).getroot()
    visual = source.find(".//geom[@mesh='right_base_link_visual']")
    quat = np.fromstring(visual.get("quat"), sep=" ")
    rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    assets = ET.SubElement(root, "asset")
    for asset in converted.find("asset"):
        assets.append(copy.deepcopy(asset))
    world = ET.SubElement(root, "worldbody")
    body = copy.deepcopy(body)
    body.set("pos", "0 0 0")
    body.set("quat", " ".join(map(str, quat)))
    world.append(body)
    equal = ET.SubElement(root, "equality")
    for dependent, driver, mult, offset in mimics:
        ET.SubElement(equal, "joint", joint1=dependent, joint2=driver, polycoef=f"{offset} {mult} 0 0 0")
    actuators = ET.SubElement(root, "actuator")
    urdf = ET.parse(HAND / "inspire_hand_right.urdf")
    for alias, joint in zip(ACT_NAMES, DRIVEN):
        limit = urdf.find(f"joint[@name='{joint}']/limit")
        ET.SubElement(
            actuators,
            "position",
            name=alias,
            joint=joint,
            kp="4",
            kv=".12",
            ctrlrange=f"{limit.get('lower')} {limit.get('upper')}",
            forcerange="-1 1",
        )
    for element in root.iter():
        for key, value in list(element.attrib.items()):
            if value.startswith("R_"):
                element.set(key, value[2:])
    body.set("name", "base")
    for i, name in zip([4, 8, 12, 16, 20], ["thumb", "index", "middle", "ring", "pinky"]):
        body.find(f".//site[@name='landmark_{i}']").set("name", f"right_{name}_tip")
    ET.ElementTree(root).write(output / "native.xml")
    return output / "native.xml", rotation


def native_fk_class(upstream_fk):
    """Subclass FK table construction to read the native model's own coupling equations."""

    class NativeFK(upstream_fk):
        def _drive(self, joint, value):
            driver = self._model.joint(joint).id
            self._data.qpos[self._model.jnt_qposadr[driver]] = float(value)
            for i in range(self._model.neq):
                if self._model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT and self._model.eq_obj2id[i] == driver:
                    dependent = self._model.eq_obj1id[i]
                    self._data.qpos[self._model.jnt_qposadr[dependent]] = sum(
                        self._model.eq_data[i, k] * float(value) ** k for k in range(5)
                    )

        def _set_finger_qpos(self, name, ctrl):
            self._drive(f"{name}_proximal_joint", ctrl)

        def _set_thumb_qpos(self, ctrl_pitch, ctrl_yaw):
            self._drive("thumb_proximal_yaw_joint", ctrl_yaw)
            self._drive("thumb_proximal_pitch_joint", ctrl_pitch)

    return NativeFK


def model_spec(model, prefix=""):
    """Read control and coupling parameters rather than assuming model equivalence."""
    coupling = {
        model.joint(int(model.eq_obj1id[i])).name: model.eq_data[i, :5].tolist()
        for i in range(model.neq)
        if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT
    }
    return {
        "control_ranges": {
            model.actuator(i).name: model.actuator_ctrlrange[i].tolist() for i in range(model.nu)
        },
        "position_gains": {
            model.actuator(i).name: float(model.actuator_gainprm[i, 0]) for i in range(model.nu)
        },
        "joint_coupling": coupling,
        "total_mass_kg": float(model.body_mass.sum()),
    }


def audit(reference_fk, native_fk, closure_class, output):
    """Compare common-control fingertip geometry and record both planner outputs."""
    rows = []
    for amount in [0.0, 0.25, 0.5, 0.75, 1.0]:
        values = {}
        for key in ACT_NAMES:
            low = max(reference_fk.ctrl_min[key], native_fk.ctrl_min[key])
            high = min(reference_fk.ctrl_max[key], native_fk.ctrl_max[key])
            values[key] = low + amount * (high - low)
        a = reference_fk.all_tips(values)
        b = native_fk.all_tips(values)
        rows.append(
            {
                "fraction": amount,
                "control": values,
                "tip_differences_m": {key: float(np.linalg.norm(a[key] - b[key])) for key in a},
            }
        )
    closure = closure_class(native_fk)
    plans = []
    for n in [2, 3, 4, 5]:
        for width in [0.06, 0.08]:
            result = closure.line(width) if n == 2 else closure.plane(width, n_fingers=n)
            plans.append(
                {
                    "fingers": n,
                    "requested_site_width_m": width,
                    "achieved_site_width_m": result.width,
                    "tilt_deg": result.tilt_deg,
                    "controls": result.ctrl_values,
                }
            )
    report = {
        "reference_model": model_spec(reference_fk._model),
        "native_model": model_spec(native_fk._model),
        "same_control_geometry_comparison": rows,
        "native_fk_plans": plans,
        "native_model_replaced": False,
        "upstream_closure_solver_modified": False,
    }
    (output / "model-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def fit_palm(model, active, seed, fingers, position, rotation, collision_aware=False):
    """Fit the G1 arm to a native hand-base target without changing the model."""
    from scipy.optimize import least_squares

    from humanoid.kinematics import set_configuration
    from manipulation.fit import control_vector

    data = mujoco.MjData(model)
    hand = model.body("R_hand_base_link").id
    control = seed.copy()
    control[7:] = fingers
    lo, hi = model.actuator_ctrlrange[active[:7]].T
    torso = [
        g
        for g in range(model.ngeom)
        if model.geom_contype[g] and model.body(model.geom_bodyid[g]).name == "torso_link"
    ]
    arm_geoms = [
        g
        for g in range(model.ngeom)
        if model.geom_contype[g]
        and model.body(model.geom_bodyid[g]).name
        in (
            "right_shoulder_roll_link",
            "right_shoulder_yaw_link",
            "right_elbow_link",
            "right_wrist_roll_link",
        )
    ]

    def residual(arm):
        control[:7] = arm
        set_configuration(model, data, control_vector(model, active, control))
        current = data.xmat[hand].reshape(3, 3)
        collision = (
            np.array([mujoco.mj_geomDistance(model, data, a, b, 0.1, None) for a in arm_geoms for b in torso])
            if collision_aware
            else np.empty(0)
        )
        return np.r_[
            10 * np.minimum(collision - 0.002, 0),
            data.xpos[hand] - position,
            0.05 * Rotation.from_matrix(rotation.T @ current).as_rotvec(),
            0.0001 * (arm - seed[:7]),
        ]

    fit = least_squares(
        residual, np.clip(seed[:7], lo + 1e-6, hi - 1e-6), bounds=(lo + 1e-7, hi - 1e-7), max_nfev=150
    )
    control[:7] = fit.x
    residual(fit.x)
    return (
        control.copy(),
        float(np.linalg.norm(data.xpos[hand] - position)),
        float(Rotation.from_matrix(rotation.T @ data.xmat[hand].reshape(3, 3)).magnitude()),
    )


def g1_plan(model, active, seed, closure, frame_rotation, mode, width, yaw_offset=0.0, collision_aware=False):
    """Build coordinated arm/finger commands using the unchanged upstream closure solver."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    case = model.body("case").id
    centre = data.xpos[case].copy()
    yaw = Rotation.from_matrix(data.xmat[case].reshape(3, 3)).as_euler("xyz")[2] + np.pi / 2 + yaw_offset
    symmetry = Rotation.from_euler("z", yaw).as_matrix()
    widths = np.linspace(0.11, width, 31)
    values = []
    errors = []
    positions = []
    rotations = []
    control = seed.copy()
    for w in widths:
        result = closure.line(w) if mode == 2 else closure.plane(w, n_fingers=mode)
        position = centre + symmetry @ result.world_base(0)
        rotation = symmetry @ result._rot_matrix(result.base_tilt_y) @ frame_rotation
        fingers = np.array([result.ctrl_values[a] for a in ACT_NAMES])
        control, p_error, r_error = fit_palm(
            model, active, control, fingers, position, rotation, collision_aware=collision_aware
        )
        values.append(control.copy())
        errors.append([p_error, r_error])
        positions.append(position)
        rotations.append(rotation)
    pre, p_error, r_error = fit_palm(
        model,
        active,
        values[0],
        values[0][7:],
        positions[0] + [0, 0, 0.08],
        rotations[0],
        collision_aware=collision_aware,
    )
    errors.append([p_error, r_error])
    high, p_error, r_error = fit_palm(
        model,
        active,
        values[-1],
        values[-1][7:],
        positions[-1] + [0, 0, 0.05],
        rotations[-1],
        collision_aware=collision_aware,
    )
    errors.append([p_error, r_error])
    times = np.r_[0, np.linspace(2, 5, len(widths)), 7, 9, 10]
    controls = np.vstack([pre, values, values[-1], high, high])
    return (
        times,
        controls,
        {
            "maximum_hand_base_position_error_m": float(np.max(np.array(errors)[:, 0])),
            "maximum_hand_base_rotation_error_deg": float(np.max(np.array(errors)[:, 1]) * 180 / np.pi),
            "waypoints": len(times),
            "box_symmetry_rotation_rad": float(yaw),
        },
    )


def execute_g1(model, active, plan_times, plan_controls, mode, feedback=False):
    """Apply coordinated motor commands on the G1; the case remains dynamically free."""
    from humanoid.kinematics import set_configuration
    from manipulation.fit import control_vector
    from manipulation.metrics import longest_interval
    from video_g1.grasp_verify import ContactReadout, opposition

    data = mujoco.MjData(model)
    data.ctrl[:] = control_vector(model, active, plan_controls[0])
    set_configuration(model, data, data.ctrl)
    mujoco.mj_forward(model, data)
    case = model.body("case").id
    hand = model.body("R_hand_base_link").id
    joint = model.joint("case_free").id
    adr = model.jnt_qposadr[joint]
    dof = model.jnt_dofadr[joint]
    origin = data.qpos[adr : adr + 3].copy()
    robot = np.ones(model.nv, bool)
    robot[dof : dof + 6] = False
    reader = ContactReadout(model)
    control = plan_controls[0].copy()
    held = np.zeros(5, bool)
    last = control.copy()
    records = []
    stream = []
    lo, hi = model.actuator_ctrlrange[active].T
    for step in range(int(10 / model.opt.timestep)):
        t = data.time
        desired = np.array([np.interp(t, plan_times, plan_controls[:, i]) for i in range(13)])
        if feedback and t >= 5:
            touches, _ = reader.read(data)
            targets = np.r_[0.8, np.full(4, 0.8 / max(mode - 1, 1))]
            ramp = np.clip((t - 5) / 2, 0, 1)
            active_motors = [10, 11] if mode == 2 else list(range(12 - mode, 12))
            for motor in active_motors:
                desired[motor] += ramp * (hi[motor] - desired[motor])
            for finger, motor in enumerate([11, 10, 9, 8, 7]):
                if touches[finger, 0] >= targets[finger]:
                    held[finger] = True
                if held[finger]:
                    desired[motor] = last[motor]
        desired = np.clip(desired, lo, hi)
        control += np.clip(desired - control, -1.2 * model.opt.timestep, 1.2 * model.opt.timestep)
        last = control.copy()
        data.ctrl[:] = control_vector(model, active, control)
        data.qfrc_applied[:] = 0
        data.qfrc_applied[robot] = data.qfrc_bias[robot]
        stream.append(control.copy())
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError("Nonfinite G1 plant")
        if step % 20 == 0:
            mujoco.mj_forward(model, data)
            external, vectors = reader.read(data)
            quality = opposition(vectors.get(case, np.zeros((5, 3))))
            relative = data.xmat[hand].reshape(3, 3).T @ (data.qpos[adr : adr + 3] - data.xpos[hand])
            relative_r = data.xmat[hand].reshape(3, 3).T @ data.xmat[case].reshape(3, 3)
            records.append((data.time, data.qpos.copy(), quality, external, relative, relative_r))
    qpos = np.array([r[1] for r in records])
    height = qpos[:, adr + 2] - origin[2]
    quality = np.array([r[2] for r in records])
    supported = (height > 0.025) & (quality > 0.5)
    indices = np.flatnonzero(supported)
    relative = np.array([r[4] for r in records])
    relative_r = np.array([r[5] for r in records])
    slip = (
        float(np.max(np.linalg.norm(relative[indices] - relative[indices[0]], axis=1)))
        if len(indices)
        else None
    )
    angle = (
        float(np.max(Rotation.from_matrix(relative_r[indices[0]].T @ relative_r[indices]).magnitude()))
        if len(indices)
        else None
    )
    report = {
        "robot": "Unchanged native G1 + Unitree Inspire right hand",
        "mode_fingers": mode,
        "force_stop_adapter": feedback,
        "maximum_lift_m": float(height.max()),
        "opposition_supported_lift_s": longest_interval(supported, 0.02),
        "relative_slip_m": slip,
        "angular_slip_rad": angle,
        "success": bool(
            supported[-25:].all()
            and longest_interval(supported, 0.02) >= 0.5
            and slip is not None
            and slip < 0.01
            and angle < np.deg2rad(10)
        ),
        "case_external_forces_or_attachments": False,
        "robot_bias_compensation": True,
        "model_replaced": False,
    }
    arrays = {
        "time_s": np.array([r[0] for r in records]),
        "qpos": qpos,
        "case_opposition": quality,
        "external_forces_N": np.array([r[3] for r in records]),
        "object_in_hand_m": relative,
        "executed_controls": np.array(stream),
    }
    return report, arrays


def main():
    """Generate or replay a width-planned native G1 grasp with explicit physical checks."""
    import argparse
    import os
    import subprocess
    import sys
    import time

    from humanoid.book_g1 import make_model
    from video_g1.provenance import write_manifest

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Replay a saved control plan, not robot/object states")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--fingers", type=int, choices=[2, 3, 4, 5])
    parser.add_argument(
        "--width",
        type=float,
        default=0.08,
        help="Internal site width in metres; not automatically object width",
    )
    parser.add_argument("--azimuth-deg", type=float, default=180.0)
    parser.add_argument("--collision-aware", action="store_true")
    parser.add_argument("--park-inactive", action="store_true")
    parser.add_argument("--force-stop", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    model, active, _ = make_model(args.scene, args.output, args.scene)
    inputs = [args.scene / "scene.xml"]
    if args.reference:
        saved = np.load(args.reference)
        names = [model.actuator(int(i)).name for i in active]
        if names != saved["action_names"].tolist():
            raise ValueError("Saved control order differs from native model")
        args.fingers = int(saved["grasp_fingers"])
        args.width = float(saved["requested_internal_width_m"])
        times = saved["time_s"]
        controls = saved["controls"]
        inputs.append(args.reference)
        ik = {"source": "Previously generated control plan; no state playback"}
    else:
        args.fingers = args.fingers or 2
        if args.upstream is None or args.seed is None:
            parser.error("--upstream and --seed are required when generating")
        spec = json.loads((Path(__file__).parent / "RH56_UPSTREAM.json").read_text())
        for path, key in [(args.upstream, "controller"), (args.upstream / "h1_mujoco", "reference_hand")]:
            revision = subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
            ).strip()
            if revision != spec[key]["revision"]:
                raise ValueError(f"Unpinned upstream revision: {path}")
        os.environ["RH56_CACHE_DIR"] = str((args.output / "cache").resolve())
        sys.path.insert(0, str(args.upstream.resolve()))
        from rh56_controller.grasp_geometry import ClosureGeometry, InspireHandFK

        source = args.upstream / "h1_mujoco" / spec["reference_hand"]["source"]
        xml, rotation = standalone_native(source, args.output / "native-fk")
        fk = native_fk_class(InspireHandFK)(str(xml.resolve()), rebuild=True)
        closure = ClosureGeometry(fk)
        seed = np.load(args.seed)["initial_actions"]
        times, controls, ik = g1_plan(
            model,
            active,
            seed,
            closure,
            rotation,
            args.fingers,
            args.width,
            yaw_offset=np.deg2rad(args.azimuth_deg),
            collision_aware=args.collision_aware,
        )
        if args.park_inactive:
            controls[:, 7 : 12 - args.fingers] = (
                model.actuator_ctrlrange[active[7 : 12 - args.fingers], 1] * 0.9
            )
        inputs.extend([args.seed, source])
    if (
        len(times) != len(controls)
        or controls.shape[1] != len(active)
        or not np.isfinite(controls).all()
        or not (np.diff(times) > 0).all()
    ):
        raise ValueError("Invalid native control reference")
    write_manifest(args.output, inputs)
    report, arrays = execute_g1(model, active, times, controls, args.fingers, feedback=args.force_stop)
    report.update(
        ik=ik,
        requested_internal_width_m=args.width,
        elapsed_s=time.monotonic() - started,
        case_mass_kg=float(model.body_mass[model.body("case").id]),
        gravity_m_s2=model.opt.gravity.tolist(),
        ready_for_training=False,
    )
    np.savez_compressed(
        args.output / "reference.npz",
        time_s=times,
        controls=controls,
        action_names=[model.actuator(int(i)).name for i in active],
        grasp_fingers=args.fingers,
        requested_internal_width_m=args.width,
    )
    np.savez_compressed(args.output / "rollout.npz", **arrays)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
