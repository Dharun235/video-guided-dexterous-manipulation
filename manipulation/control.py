"""External finger-contact feedback, independent of the contacted object's identity."""

from dataclasses import dataclass, field

import mujoco
import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def finger_geoms(model):
    """Group native collision geometry by thumb, index, middle, ring and pinky."""
    return [
        [
            g
            for g in range(model.ngeom)
            if model.geom_contype[g]
            and model.body(model.geom_bodyid[g]).name in (f"R_{f}_intermediate", f"R_{f}_distal")
        ]
        for f in FINGERS
    ]


def external_contacts(model, data):
    """Return external normal/shear force sums in newtons, shaped (5, 2)."""
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
    """Integrate bounded finger-closure corrections from pressure and shear."""

    target_force: float = 0.6
    closure_rate: float = 0.4
    offsets: np.ndarray = field(default_factory=lambda: np.zeros(5), init=False)

    def update(self, touches, dt, enabled=True):
        """Return five closure offsets in radians; reset them when disabled."""
        if not enabled:
            self.offsets[:] = 0
            return self.offsets.copy()
        normal, shear = np.asarray(touches).T
        slipping = (shear > 0.65 * np.maximum(normal, 0.05)) & (normal > 0.05)
        rate = np.clip(self.target_force - normal, -2, 1) + 0.5 * slipping
        self.offsets = np.clip(self.offsets + self.closure_rate * rate * dt, -0.15, 0.55)
        return self.offsets.copy()


def grasp_detected(touches, threshold=0.08):
    """Detect thumb and index/middle opposition from external normal forces."""
    force = np.asarray(touches)[:, 0]
    return bool(force[0] > threshold and np.any(force[1:3] > threshold))
