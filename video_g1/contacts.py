"""Read external finger contacts and evaluate opposed normal forces."""

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


class ContactReadout:
    """Cache topology and return external forces plus forces directed onto each body."""

    def __init__(self, model):
        self.model = model
        self.fingers = {g: i for i, geoms in enumerate(finger_geoms(model)) for g in geoms}
        pelvis = model.body("pelvis").id
        self.robot = set()
        for i in range(1, model.nbody):
            parent = i
            while parent:
                if parent == pelvis:
                    self.robot.add(i)
                    break
                parent = int(model.body_parentid[parent])

    def read(self, data):
        """Return (5,2) normal/shear sums and per-body finger normal-force vectors."""
        touches = np.zeros((5, 2))
        vectors = {}
        for index, contact in enumerate(data.contact):
            if contact.geom1 in self.fingers:
                finger = self.fingers[contact.geom1]
                other = int(self.model.geom_bodyid[contact.geom2])
                sign = 1
            elif contact.geom2 in self.fingers:
                finger = self.fingers[contact.geom2]
                other = int(self.model.geom_bodyid[contact.geom1])
                sign = -1
            else:
                continue
            if other in self.robot:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, index, force)
            normal = max(float(force[0]), 0)
            touches[finger] += [normal, np.linalg.norm(force[1:3])]
            vectors.setdefault(other, np.zeros((5, 3)))[finger] += sign * normal * contact.frame[:3]
        return touches, vectors


def opposition(vectors, minimum_force=0.08):
    """Return thumb/index-or-middle opposition in [0,1]; parallel pressure is zero."""
    lengths = np.linalg.norm(vectors, axis=1)
    if lengths[0] <= minimum_force:
        return 0.0
    quality = [
        max(0.0, -np.dot(vectors[0], vectors[i]) / (lengths[0] * lengths[i]))
        for i in [1, 2]
        if lengths[i] > minimum_force
    ]
    return float(np.clip(max(quality, default=0.0), 0, 1))
