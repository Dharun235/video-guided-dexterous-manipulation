"""Export the existing case calibration and sampled local surface for pose checks."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from reconstruction.object_motion import Geometry


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    geometry = Geometry(args.scene)
    q = geometry.model.body_quat[geometry.body][[1, 2, 3, 0]]
    pose = dict(rotation=Rotation.from_quat(q).as_matrix().tolist(),
                translation=geometry.model.body_pos[geometry.body].tolist())
    (args.output / "initial_pose.json").write_text(json.dumps(pose, indent=2) + "\n")
    np.save(args.output / "case_surface.npy", geometry.surface)


if __name__ == "__main__":
    main()
