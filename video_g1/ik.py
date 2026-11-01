"""Fit native G1 arm and waist joints to palm targets in a separate IK workspace."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from humanoid.kinematics import control_vector, set_configuration


def fit_palm(model, active, seed, fingers, position, rotation, collision_aware=False):
    """Fit the G1 arm to a native hand-base target without changing the model."""
    from scipy.optimize import least_squares

    from humanoid.kinematics import set_configuration

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


def fit_upper_body(model, active, seed, fingers, position, rotation):
    """Fit arm and native waist joints with explicit arm/torso clearance penalties."""
    from scipy.optimize import least_squares

    data = mujoco.MjData(model)
    waist = np.array([model.actuator(f"waist_{a}_joint").id for a in ["yaw", "roll", "pitch"]])
    driven = np.r_[active[:7], waist]
    lo, hi = model.actuator_ctrlrange[driven].T
    hand = model.body("R_hand_base_link").id
    torso = [
        g
        for g in range(model.ngeom)
        if model.geom_contype[g] and model.body(model.geom_bodyid[g]).name == "torso_link"
    ]
    arm = [
        g
        for g in range(model.ngeom)
        if model.geom_contype[g]
        and model.body(model.geom_bodyid[g]).name
        in [
            "right_shoulder_roll_link",
            "right_shoulder_yaw_link",
            "right_elbow_link",
            "right_wrist_roll_link",
        ]
    ]

    def residual(value):
        command = np.r_[value[:7], fingers]
        controls = control_vector(model, active, command)
        controls[waist] = value[7:]
        set_configuration(model, data, controls)
        distances = np.array(
            [mujoco.mj_geomDistance(model, data, a, b, 0.1, None) for a in arm for b in torso]
        )
        return np.r_[
            data.xpos[hand] - position,
            0.05 * Rotation.from_matrix(rotation.T @ data.xmat[hand].reshape(3, 3)).as_rotvec(),
            10 * np.minimum(distances - 0.002, 0),
            0.00005 * (value - seed),
        ]

    result = least_squares(
        residual, np.clip(seed, lo + 1e-6, hi - 1e-6), bounds=(lo + 1e-7, hi - 1e-7), max_nfev=160
    )
    residual(result.x)
    return (
        result.x,
        float(np.linalg.norm(data.xpos[hand] - position)),
        float(Rotation.from_matrix(rotation.T @ data.xmat[hand].reshape(3, 3)).magnitude()),
    )
