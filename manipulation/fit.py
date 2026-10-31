"""Fit a feasible grasp from observed landmarks and the reconstructed case surface."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from humanoid.book_g1 import BACKGROUND_CONTROL, ORIGIN, SCENE_ROTATION, make_model
from humanoid.kinematics import set_configuration

from .control import finger_geoms


def control_vector(model, active, values):
    """Embed 13 arm/hand targets into the model controls and fixed background posture."""
    control = np.zeros(model.nu)
    control[active] = values
    for name, value in BACKGROUND_CONTROL.items():
        control[model.actuator(name).id] = value
    return control


def case_pose(model, data, position, rotation):
    """Set the free case pose in a reset or IK workspace, never during execution."""
    joint = model.joint("case_free").id
    adr = model.jnt_qposadr[joint]
    q = Rotation.from_matrix(rotation).as_quat()
    data.qpos[adr : adr + 7] = np.r_[position, q[[3, 0, 1, 2]]]


def fit_grasp(model, active, target, position, rotation, seed, iterations=120):
    """Fit bounded joints against landmark and surface-distance targets in an IK workspace."""
    data = mujoco.MjData(model)
    sites = [model.site(f"landmark_{i}").id for i in range(21)]
    fingers = finger_geoms(model)
    case_geom = model.geom("case_solid").id
    lo, hi = model.actuator_ctrlrange[active].T
    case_pose(model, data, position, rotation)

    def residual(u):
        set_configuration(model, data, control_vector(model, active, u))
        mujoco.mj_forward(model, data)
        points = data.site_xpos[sites]
        distances = [
            min(mujoco.mj_geomDistance(model, data, g, case_geom, 0.1, None) for g in gs) for gs in fingers
        ]
        # Contact objectives are geometric proposals in a separate IK workspace.
        # No object pose or robot qpos is overwritten in a running rollout.
        return np.r_[
            0.5 * (points - target).ravel(),
            2 * (points[[4, 8]] - target[[4, 8]]).ravel(),
            8 * (np.array(distances[:3]) - 0.0008),
            15 * np.minimum(distances, 0),
            0.001 * (u - seed),
        ]

    fit = least_squares(
        residual,
        np.clip(seed, lo + 1e-6, hi - 1e-6),
        bounds=(lo + 1e-7, hi - 1e-7),
        max_nfev=iterations,
        diff_step=1e-4,
        ftol=1e-7,
        xtol=1e-7,
    )
    residual(fit.x)
    return fit.x, data


def main():
    """Write a grasp proposal from recorded landmarks and approximate collision geometry."""
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--scene", type=Path, default=Path("examples/object-motion/scene"))
    p.add_argument("--calibration", type=Path, default=Path("examples/physics/calibration"))
    p.add_argument("--hand-fit", type=Path)
    p.add_argument("--seed", type=Path, default=Path("examples/physics/hand-seed.npz"))
    p.add_argument("--object-track", type=Path, default=Path("examples/object-motion"))
    p.add_argument("--frame", type=int, default=240)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    m, active, state = make_model(a.scene, a.output, a.calibration)
    surface = np.load(a.object_track / "case_surface.npy")
    bounds = np.array([surface.min(0), surface.max(0)])
    # A stable box is an explicit collision approximation of the small rigid case.
    # The neural mesh remains visual only; its unseen uneven base is unverified.
    xml = ET.parse(a.output / "scene.xml")
    geom = xml.find(".//geom[@name='case_solid']")
    geom.attrib.pop("mesh", None)
    geom.set("type", "box")
    geom.set("size", " ".join(map(str, (bounds[1] - bounds[0]) / 2)))
    geom.set("pos", " ".join(map(str, bounds.mean(0))))
    xml.write(a.output / "scene.xml")
    m = mujoco.MjModel.from_xml_path(str((a.output / "scene.xml").resolve()))
    if a.hand_fit:
        h = np.load(a.hand_fit / "retarget.npz")
        observed = bool(h["valid"][a.frame])
        landmarks = h["target_landmarks_m"][a.frame]
        initial_actions = h["actions"][a.frame]
    else:
        h = np.load(a.seed)
        if int(h["source_frame"]) != a.frame:
            raise ValueError("Saved hand seed uses a different source frame")
        observed = True
        landmarks = h["target_landmarks_m"]
        initial_actions = h["initial_actions"]
    o = np.load(a.object_track / "object_track.npz")
    if not observed or not o["valid"][a.frame] or o["requires_review"][a.frame]:
        raise ValueError("Grasp seed requires observed hand and unflagged object pose")
    pos = SCENE_ROTATION @ o["translation_world_m"][0] + ORIGIN
    rot = SCENE_ROTATION @ o["rotation_world"][0]
    source_pos = SCENE_ROTATION @ o["translation_world_m"][a.frame] + ORIGIN
    source_rot = SCENE_ROTATION @ o["rotation_world"][a.frame]
    local = (landmarks - source_pos) @ source_rot
    surface = np.load(a.object_track / "case_surface.npy")
    # Snap only observed contact candidates to estimated geometry; keep all raw
    # observations untouched in their source archives.
    for j in [4, 8, 12, 16, 20]:
        point = np.clip(local[j], bounds[0], bounds[1])
        if np.all(point > bounds[0]) and np.all(point < bounds[1]):
            axis = np.argmin(np.minimum(point - bounds[0], bounds[1] - point))
            point[axis] = bounds[int(point[axis] > bounds[:, axis].mean()), axis]
        local[j] = point
    target = local @ rot.T + pos
    q, d = fit_grasp(m, active, target, pos, rot, initial_actions)
    np.savez_compressed(
        a.output / "grasp.npz",
        actions=q,
        active=active,
        state=state,
        target=target,
        position=pos,
        rotation=rot,
        landmarks=d.site_xpos[[m.site(f"landmark_{i}").id for i in range(21)]],
    )
    distances = [
        min(mujoco.mj_geomDistance(m, d, g, m.geom("case_solid").id, 0.1, None) for g in gs)
        for gs in finger_geoms(m)
    ]
    report = dict(
        source_sha256=json.loads((a.calibration / "reconstruction.json").read_text())["source_sha256"],
        source_frame=a.frame,
        method="Observed human landmarks, nearest estimated surface contacts and bounded SciPy contact IK",
        finger_distances_m=distances,
        actions=q.tolist(),
        requires_review=True,
        physical_grasp_verified=False,
        mean_landmark_error_m=float(
            np.linalg.norm(
                d.site_xpos[[m.site(f"landmark_{i}").id for i in range(21)]] - target, axis=1
            ).mean()
        ),
        collision_proxy="Box fitted to estimated case extents; neural mesh visual only",
        collision_bounds_local_m=bounds.tolist(),
    )
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
