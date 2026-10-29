"""Bounded right-arm kinematics for the supported G1."""
import mujoco
import numpy as np


def arm_indices(model, side):
    names = [
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    ]
    joints = [model.joint(f"{side}_{name}_joint").id for name in names]
    return (
        model.jnt_qposadr[joints],
        model.jnt_dofadr[joints],
        [model.actuator(f"{side}_{name}_joint").id for name in names],
        joints,
    )


def solve_arm(model, data, side, target, seed, rotation=None, *, update_dynamics=True, site_name=None, axis=None):
    """Bounded damped-Jacobian position IK; it does not mutate live physics data."""
    qadr, dofs, _, joints = arm_indices(model, side)
    data.qpos[qadr] = seed
    sid = model.site(site_name or f"{side}_grasp_site").id
    jac = np.zeros((3, model.nv))
    jacr = np.zeros_like(jac)
    lower, upper = model.jnt_range[joints, 0].copy(), model.jnt_range[joints, 1].copy()
    # Conservative shoulder envelope avoids driving the roll housing into torso.
    if side == "right":
        upper[1] = 0.10
    else:
        lower[1] = -0.10
    for _ in range(100):
        # IK needs transforms and Jacobians, not contact solving at every iterate.
        # Retain a full forward pass below for callers inspecting the final pose.
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        error = target - data.site_xpos[sid]
        if np.linalg.norm(error) < 0.003 and rotation is None and axis is None:
            break
        mujoco.mj_jacSite(model, data, jac, jacr, sid)
        j = jac[:, dofs]
        if rotation is not None:
            current, desired, angular = np.zeros(4), np.zeros(4), np.zeros(3)
            mujoco.mju_mat2Quat(current, data.site_xmat[sid])
            mujoco.mju_mat2Quat(desired, np.asarray(rotation).ravel())
            mujoco.mju_subQuat(angular, desired, current)
            angular = data.site_xmat[sid].reshape(3, 3) @ angular
            error = np.r_[error, 0.20 * angular]
            j = np.vstack([j, 0.20 * jacr[:, dofs]])
        elif axis is not None:
            local_axis,desired_axis=axis
            current=data.site_xmat[sid].reshape(3,3)@np.asarray(local_axis)
            x,y,z=current
            cross=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
            error=np.r_[error,.08*(np.asarray(desired_axis)-current)]
            j=np.vstack([j,-.08*cross@jacr[:,dofs]])
        delta = j.T @ np.linalg.solve(j @ j.T + 0.0005 * np.eye(len(error)), error)
        data.qpos[qadr] = np.clip(data.qpos[qadr] + np.clip(delta, -0.12, 0.12), lower, upper)
    if update_dynamics:
        mujoco.mj_forward(model, data)
    else:
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
    return data.qpos[qadr].copy(), float(np.linalg.norm(target - data.site_xpos[sid]))
