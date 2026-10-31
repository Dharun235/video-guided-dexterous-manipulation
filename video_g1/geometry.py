"""Native-hand landmark geometry and control Jacobians for trajectory fitting."""

import mujoco
import numpy as np


def palm_frame(points, jacobian=None):
    """Return a right-handed palm basis and optional derivatives with respect to controls.

    Columns point from pinky to index, wrist to middle knuckle, and their cross
    product. Positions are metres; Jacobians have shape (21, 3, controls).
    """
    a = points[5] - points[17]
    length = np.linalg.norm(a)
    if length < 1e-8:
        raise ValueError("Degenerate palm width")
    x = a / length
    b = points[9] - points[0]
    b = b - x * np.dot(x, b)
    length_b = np.linalg.norm(b)
    if length_b < 1e-8:
        raise ValueError("Degenerate palm length")
    y = b / length_b
    basis = np.column_stack([x, y, np.cross(x, y)])
    if jacobian is None:
        return basis
    dx = (np.eye(3) - np.outer(x, x)) @ (jacobian[5] - jacobian[17]) / length
    raw = points[9] - points[0]
    draw = jacobian[9] - jacobian[0]
    db = draw - dx * np.dot(x, raw) - x[:, None] * (raw @ dx + x @ draw)[None, :]
    dy = (np.eye(3) - np.outer(y, y)) @ db / length_b
    dz = np.cross(dx.T, y).T + np.cross(x, dy.T).T
    return basis, np.stack([dx, dy, dz], axis=1)


def coupled_control_map(model, active):
    """Map joint-target changes to native DOFs, retaining linear Inspire coupling."""
    mapping = np.zeros((model.nv, len(active)))
    for column, actuator in enumerate(active):
        if model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError("Trajectory fitting currently requires joint actuators")
        joint = model.actuator_trnid[actuator, 0]
        mapping[model.jnt_dofadr[joint], column] = 1
    for i in range(model.neq):
        if model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        if np.any(model.eq_data[i, 2:5]):
            raise ValueError("Nonlinear joint coupling needs a different Jacobian")
        dependent, driver = model.eq_obj1id[i], model.eq_obj2id[i]
        mapping[model.jnt_dofadr[dependent]] = model.eq_data[i, 1] * mapping[model.jnt_dofadr[driver]]
    return mapping


def site_jacobians(model, data, sites, mapping):
    """Return native site derivatives (sites, 3, controls) at an IK configuration."""
    mujoco.mj_comPos(model, data)
    result = np.empty((len(sites), 3, mapping.shape[1]))
    translation = np.zeros((3, model.nv))
    for i, site in enumerate(sites):
        mujoco.mj_jacSite(model, data, translation, None, int(site))
        result[i] = translation @ mapping
    return result
