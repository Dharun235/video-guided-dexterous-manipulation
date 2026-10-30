"""External finger-contact feedback, independent of the contacted object's identity."""

from dataclasses import dataclass

import mujoco
import numpy as np

from .fit import finger_geoms


def external_contacts(model, data):
    groups = finger_geoms(model)
    pelvis = model.body("pelvis").id
    robot = set()
    for i in range(1, model.nbody):
        parent = i
        while parent:
            if parent == pelvis:
                robot.add(i)
                break
            parent = int(model.body_parentid[parent])
    normal = np.zeros(5)
    shear = np.zeros(5)
    for contact_id, contact in enumerate(data.contact):
        for i, geoms in enumerate(groups):
            if contact.geom1 in geoms:
                other = contact.geom2
            elif contact.geom2 in geoms:
                other = contact.geom1
            else:
                continue
            if model.geom_bodyid[other] in robot:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, contact_id, force)
            normal[i] += max(force[0], 0)
            shear[i] += np.linalg.norm(force[1:3])
    return np.column_stack([normal, shear])


@dataclass
class ContactFeedback:
    target_force: float = 0.6
    closure_rate: float = 0.4
    offsets: np.ndarray = None

    def __post_init__(self):
        self.offsets = np.zeros(5)

    def update(self, touches, dt, enabled=True):
        if not enabled:
            self.offsets[:] = 0
            return self.offsets.copy()
        normal, shear = np.asarray(touches).T
        slipping = (shear > 0.65 * np.maximum(normal, 0.05)) & (normal > 0.05)
        rate = np.clip(self.target_force - normal, -2, 1) + 0.5 * slipping
        self.offsets = np.clip(self.offsets + self.closure_rate * rate * dt, -0.15, 0.55)
        return self.offsets.copy()


def grasp_detected(touches, threshold=0.08):
    force = np.asarray(touches)[:, 0]
    return bool(force[0] > threshold and np.any(force[1:3] > threshold))
