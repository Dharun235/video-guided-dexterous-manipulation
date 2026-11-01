"""Export the existing case calibration and sampled local surface for pose checks."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def case_geometry(scene):
    """Sample the compiled case mesh in its body-local coordinate frame."""
    geometry = SimpleNamespace()
    geometry.model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
    geometry.data = mujoco.MjData(geometry.model)
    geometry.body = mujoco.mj_name2id(geometry.model, mujoco.mjtObj.mjOBJ_BODY, "case")
    geometry.geomgroup = np.array([1, 0, 0, 0, 0, 0], np.uint8)
    # Use compiled geometry so original and neural mesh candidates share
    # exactly the same body-coordinate convention as MuJoCo.
    mujoco.mj_forward(geometry.model, geometry.data)
    geom = mujoco.mj_name2id(geometry.model, mujoco.mjtObj.mjOBJ_GEOM, "taa_case_appearance")
    if geom < 0:
        geom = mujoco.mj_name2id(geometry.model, mujoco.mjtObj.mjOBJ_GEOM, "case_solid")
    mesh = geometry.model.geom_dataid[geom]
    first, count = geometry.model.mesh_vertadr[mesh], geometry.model.mesh_vertnum[mesh]
    vertices = geometry.model.mesh_vert[first : first + count]
    world = vertices @ geometry.data.geom_xmat[geom].reshape(3, 3).T + geometry.data.geom_xpos[geom]
    points = (world - geometry.data.xpos[geometry.body]) @ geometry.data.xmat[geometry.body].reshape(3, 3)
    geometry.surface = points[np.linspace(0, len(points) - 1, 1600, dtype=int)].astype(float)
    return geometry


def main():
    """Write the case reset pose and sampled body-local surface."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    geometry = case_geometry(args.scene)
    q = geometry.model.body_quat[geometry.body][[1, 2, 3, 0]]
    pose = dict(
        rotation=Rotation.from_quat(q).as_matrix().tolist(),
        translation=geometry.model.body_pos[geometry.body].tolist(),
    )
    (args.output / "initial_pose.json").write_text(json.dumps(pose, indent=2) + "\n")
    np.save(args.output / "case_surface.npy", geometry.surface)


if __name__ == "__main__":
    main()
