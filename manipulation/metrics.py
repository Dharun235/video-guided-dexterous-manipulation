"""Task scoring uses object identity; the contact controller does not."""

import mujoco
import numpy as np

from .fit import finger_geoms


def case_contacts(model, data):
    result = np.zeros((5, 2))
    case = model.body("case").id
    for index, contact in enumerate(data.contact):
        b1, b2 = model.geom_bodyid[[contact.geom1, contact.geom2]]
        if case not in (b1, b2):
            continue
        other = contact.geom2 if b1 == case else contact.geom1
        for i, geoms in enumerate(finger_geoms(model)):
            if other in geoms:
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, index, force)
                result[i] += [max(force[0], 0), np.linalg.norm(force[1:3])]
    return result


def longest_interval(mask, dt):
    padded = np.r_[False, mask, False].astype(int)
    start = np.flatnonzero(np.diff(padded) == 1)
    end = np.flatnonzero(np.diff(padded) == -1)
    return float(max(end - start, default=0) * dt)


def score(time, pose, velocity, touches, phase, position, goal, relative):
    time = np.asarray(time)
    phase = np.asarray(phase)
    pose = np.asarray(pose)
    touches = np.asarray(touches)
    dt = float(np.median(np.diff(time)))
    opposing = (touches[:, 0, 0] > 0.08) & (touches[:, 1:3, 0].max(1) > 0.08)
    carry = (phase == "lift") | (phase == "carry")
    lift = pose[:, 2] - position[2]
    final_error = float(np.linalg.norm(pose[-1, :3] - goal))
    final = time > time[-1] - 0.5
    v = np.asarray(velocity)[final]
    settled = bool(
        np.max(np.linalg.norm(v[:, :3], axis=1)) < 0.03 and np.max(np.linalg.norm(v[:, 3:], axis=1)) < 0.3
    )
    released = bool(phase[-1] == "retreat" and not opposing[final].any())
    relative = np.asarray(relative)
    selected = np.flatnonzero(carry)
    slip = (
        float(np.max(np.linalg.norm(relative[selected] - relative[selected[0]], axis=1)))
        if len(selected)
        else None
    )
    supported_lift = longest_interval(opposing & (lift > 0.025), dt)
    success = bool(supported_lift > 0.3 and final_error < 0.04 and settled and released)
    return dict(
        task_success=success,
        maximum_object_lift_m=float(lift.max()),
        longest_opposing_case_contact_s=longest_interval(opposing, dt),
        contact_supported_lift_s=supported_lift,
        final_position_error_m=final_error,
        maximum_relative_translation_during_carry_m=slip,
        released=released,
        settled=settled,
        maximum_case_contact_force_N=touches[:, :, 0].max(0).tolist(),
        acceptance=dict(
            minimum_lift_m=0.025,
            minimum_contact_supported_lift_s=0.3,
            maximum_placement_error_m=0.04,
            maximum_final_linear_speed_m_s=0.03,
            maximum_final_angular_speed_rad_s=0.3,
        ),
        ready_for_training=False,
    )
