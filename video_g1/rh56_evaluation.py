"""Evaluate the unmodified RH56 width planner with a free measured case.

This is an isolated floating-hand fixture, not a G1 demonstration. The upstream
planner is unchanged. Local position-servo and force-stop adapters are evaluated
explicitly; hardware raw-force calibration is never applied to simulator newtons.
"""

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from manipulation.metrics import longest_interval
from video_g1.grasp_verify import opposition


def build_fixture(source, output):
    """Copy MJCF into a local fixture, add table/book/free case and preserve source files."""
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((source.parent / "assets").resolve()))
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", ".001")
    option.set("integrator", "implicitfast")
    world = root.find("worldbody")
    ET.SubElement(
        world,
        "geom",
        name="evaluation_table",
        type="box",
        pos="0 0 -.02",
        size=".3 .3 .02",
        rgba=".65 .58 .46 1",
    )
    ET.SubElement(
        world,
        "geom",
        name="evaluation_book",
        type="box",
        pos="0 0 .01",
        size=".11 .11 .01",
        rgba=".7 .65 .5 1",
    )
    case = ET.SubElement(world, "body", name="case", pos="0 0 .03")
    ET.SubElement(case, "freejoint", name="case_free")
    ET.SubElement(
        case,
        "geom",
        name="case_solid",
        type="box",
        size=".03 .03 .01",
        mass=".055",
        friction=".7 .005 .001",
        condim="4",
        rgba=".035 .04 .035 1",
    )
    ET.SubElement(world, "light", pos=".3 -.3 1", dir="-.2 .2 -1")
    pos = np.array([0.45, -0.45, 0.4])
    z = pos - np.array([0, 0, 0.07])
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    quat = Rotation.from_matrix(np.column_stack([x, np.cross(z, x), z])).as_quat()[[3, 0, 1, 2]]
    ET.SubElement(
        world, "camera", name="evaluation", pos=" ".join(map(str, pos)), quat=" ".join(map(str, quat))
    )
    tree.write(output / "scene.xml")
    return mujoco.MjModel.from_xml_path(str((output / "scene.xml").resolve()))


def fingers(model):
    """Map collision geometry to thumb/index/middle/ring/pinky for the upstream names."""
    names = ["thumb", "index", "middle", "ring", "pinky"]
    return {
        g: i
        for i, name in enumerate(names)
        for g in range(model.ngeom)
        if model.geom_contype[g]
        and model.body(model.geom_bodyid[g]).name in (name + "_intermediate", name + "_distal")
    }


def read_contacts(model, data, mapping, case):
    """Return external force magnitudes for control; case normal vectors for scoring only."""
    pressure = np.zeros(5)
    case_vectors = np.zeros((5, 3))
    case_forces = np.zeros(5)
    robot = set()
    for i in range(model.nbody):
        p = i
        while p:
            if model.body(p).name == "base":
                robot.add(i)
                break
            p = int(model.body_parentid[p])
    for index, c in enumerate(data.contact):
        if c.geom1 in mapping:
            finger = mapping[c.geom1]
            other = int(model.geom_bodyid[c.geom2])
            sign = 1
        elif c.geom2 in mapping:
            finger = mapping[c.geom2]
            other = int(model.geom_bodyid[c.geom1])
            sign = -1
        else:
            continue
        if other in robot:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        normal = max(force[0], 0)
        pressure[finger] += normal
        if other == case:
            case_forces[finger] += normal
            case_vectors[finger] += sign * normal * c.frame[:3]
    return pressure, case_forces, case_vectors


def execute(model, fk, closure, mode, width, feedback=False, coordinated=True, duration=10.0):
    """Run an isolated approach/closure/50 mm lift using actuator commands only."""
    data = mujoco.MjData(model)
    act = ["pinky", "ring", "middle", "index", "thumb_proximal", "thumb_yaw"]
    finger_ids = np.array([model.actuator(name).id for name in act])
    base_names = ["right_pos_x", "right_pos_y", "right_pos_z", "right_rot_x", "right_rot_y", "right_rot_z"]
    base_ids = np.array([model.actuator(name + "_position").id for name in base_names])
    base_q = np.array([model.jnt_qposadr[model.joint(name).id] for name in base_names])
    solve = lambda w: closure.line(w) if mode == 2 else closure.plane(w, n_fingers=mode)
    initial = solve(0.11)
    final = solve(width)

    def transform(result):
        position = result.world_base(0.03)
        rotation = result._rot_matrix(result.base_tilt_y)
        return np.r_[position, Rotation.from_matrix(rotation).as_euler("XYZ")]

    initial_base = transform(initial)
    final_base = transform(final)
    initial_base[2] += 0.08
    initial_base = np.clip(initial_base, *model.actuator_ctrlrange[base_ids].T)
    data.qpos[base_q] = initial_base
    # Set the robot at reset using the upstream coupling implementation.
    for name in ["index", "middle", "ring", "pinky"]:
        fk._set_finger_qpos(name, initial.ctrl_values[name])
    fk._set_thumb_qpos(initial.ctrl_values["thumb_proximal"], initial.ctrl_values["thumb_yaw"])
    for name, address in fk._jnt_adr.items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = fk._data.qpos[address]
    data.ctrl[base_ids] = initial_base
    data.ctrl[finger_ids] = [initial.ctrl_values[a] for a in act]
    mujoco.mj_forward(model, data)
    mapping = fingers(model)
    case = model.body("case").id
    adr = model.jnt_qposadr[model.joint("case_free").id]
    records = []
    stream = []
    last = None
    held = np.zeros(6, bool)
    for step in range(int(duration / model.opt.timestep)):
        t = data.time
        if step % 10 == 0:
            u = np.clip((t - 2) / 3, 0, 1)
            u = u * u * (3 - 2 * u)
            plan = solve(0.11 * (1 - u) + width * u)
            target = transform(plan) if coordinated else final_base.copy()
            target[2] += 0.08 * (1 - min(t / 2, 1))
            target[2] += 0.05 * np.clip((t - 7) / 2, 0, 1)
            command = np.array([plan.ctrl_values[a] for a in act])
            if feedback and t >= 5:
                pressure, _, _ = read_contacts(model, data, mapping, case)
                # Thumb receives 0.8 N; active non-thumb fingers share 0.8 N.
                targets = np.r_[0.8, np.full(4, 0.8 / max(mode - 1, 1))]
                order = [4, 3, 2, 1, 0]
                maximum = np.array([fk.ctrl_max[a] for a in act])
                ramp = np.clip((t - 5) / 2, 0, 1)
                active_motors = [3, 4] if mode == 2 else list(range(5 - mode, 5))
                for motor in active_motors:
                    command[motor] += ramp * (maximum[motor] - command[motor])
                for motor, finger in enumerate(order):
                    if t >= 2 and pressure[finger] >= targets[finger]:
                        held[motor] = True
                    if held[motor] and last is not None:
                        command[motor] = last[motor]
            target = np.clip(target, *model.actuator_ctrlrange[base_ids].T)
            data.ctrl[base_ids] = target
            data.ctrl[finger_ids] = command
            last = command.copy()
        stream.append(data.ctrl.copy())
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError("Nonfinite plant")
        if step % 20 == 0:
            mujoco.mj_forward(model, data)
            pressure, forces, vectors = read_contacts(model, data, mapping, case)
            records.append((data.time, data.qpos.copy(), forces, opposition(vectors), pressure))
    qpos = np.array([r[1] for r in records])
    height = qpos[:, adr + 2] - 0.03
    quality = np.array([r[3] for r in records])
    supported = (height > 0.025) & (quality > 0.5)
    report = {
        "mode_fingers": mode,
        "requested_internal_width_m": width,
        "nominal_object_width_after_upstream_20mm_offset_m": width - 0.02,
        "coordinated_palm": coordinated,
        "force_stop_adapter": feedback,
        "maximum_lift_m": float(height.max()),
        "opposition_supported_lift_s": longest_interval(supported, 0.02),
        "final_half_second_supported": bool(supported[-25:].all()),
        "success": bool(longest_interval(supported, 0.02) >= 0.5 and supported[-25:].all()),
        "maximum_case_force_N": np.array([r[2] for r in records]).max(axis=0).tolist(),
        "case_external_forces_or_attachments": False,
        "robot": "Actuated floating RH56 fixture; no G1 arm",
        "upstream_planner_unmodified": True,
    }
    return report, {
        "time_s": np.array([r[0] for r in records]),
        "qpos": qpos,
        "case_forces_N": np.array([r[2] for r in records]),
        "case_opposition": quality,
        "external_normal_forces_N": np.array([r[4] for r in records]),
        "executed_controls": np.array(stream),
    }


def main():
    """Evaluate source-planner geometry separately from physical performance."""
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--modes", type=int, nargs="+", default=[2, 3, 4, 5])
    p.add_argument("--widths", type=float, nargs="+", default=[0.06, 0.07, 0.08])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    os.environ["RH56_CACHE_DIR"] = str((a.output / "cache").resolve())
    sys.path.insert(0, str(a.upstream.resolve()))
    from rh56_controller.grasp_geometry import ClosureGeometry, InspireHandFK

    source = a.upstream / "h1_mujoco/inspire/inspire_grasp_scene.xml"
    fk = InspireHandFK(str(source.resolve()), rebuild=True)
    closure = ClosureGeometry(fk)
    model = build_fixture(source, a.output)
    rows = []
    started = time.monotonic()
    for mode in a.modes:
        for width in a.widths:
            for feedback in [False, True]:
                report, arrays = execute(model, fk, closure, mode, width, feedback=feedback)
                folder = a.output / f"mode-{mode}-width-{int(width * 1000)}-feedback-{int(feedback)}"
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
