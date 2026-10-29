"""Check pose data and imposed-pose collisions, without claiming physical execution."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--track", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--surface", type=Path, required=True)
    args = parser.parse_args()
    track = np.load(args.track / "object_track.npz")
    valid = track["valid"]
    rotations, translations = track["rotation_world"], track["translation_world_m"]
    if not np.isnan(rotations[~valid]).all() or not np.isnan(translations[~valid]).all():
        raise ValueError("Missing poses must remain NaN")
    if not np.isfinite(rotations[valid]).all() or not np.isfinite(translations[valid]).all():
        raise ValueError("Non-finite candidate poses")
    identity = rotations[valid].transpose(0, 2, 1) @ rotations[valid]
    if np.max(abs(identity - np.eye(3))) > 1e-8 or np.max(abs(np.linalg.det(rotations[valid]) - 1)) > 1e-8:
        raise ValueError("Pose rotations must be proper orthonormal matrices")
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "case")
    surface = np.load(args.surface)
    penetration, support_gap = [], []
    for i in np.flatnonzero(valid):
        q = Rotation.from_matrix(rotations[i]).as_quat()
        data.qpos[:] = np.r_[translations[i], q[[3, 0, 1, 2]]]
        mujoco.mj_forward(model, data)
        distances = [contact.dist for contact in data.contact
                     if body in [model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]]]
        penetration.append(max(0., -min(distances, default=0.)))
        if i >= len(valid) - 30:
            support_gap.append(float((surface @ rotations[i].T + translations[i])[:, 2].min()))
    report = dict(source_frames=len(valid), candidate_frames=int(valid.sum()),
                  invalid_frames=np.flatnonzero(~valid).tolist(),
                  proper_rotations=True, missing_poses_remain_nan=True,
                  frames_with_penetration_over_1mm=int((np.array(penetration) > .001).sum()),
                  maximum_penetration_m=float(max(penetration, default=0.)),
                  final_table_clearance_median_m=float(np.median(support_gap)) if support_gap else None,
                  static_collision_check_only=True, ready_for_training=False,
                  limitations=["Object poses are imposed; no robot or contact-force tracking is evaluated",
                               "Final clearance measures the estimated mesh relative to the table z=0 prior",
                               "Scale, depth and relocalized front/back orientation remain uncertain"])
    (args.track / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
