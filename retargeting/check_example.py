"""Run a small, headless portion of the upstream H2O example on CPU."""
import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
import trimesh
from yourdfpy import URDF

from .upstream import load, default_weights, REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    module, utils, source = load()
    start = time.monotonic()
    path = source / 'assets/g1_with_inspire_hand/g1_29dof_rev_1_0_with_inspire_hands.urdf'
    print('Loading bundled G1/Inspire model', flush=True)
    urdf = URDF.load(path)
    robot = pk.Robot.from_urdf(urdf)
    collision = pk.collision.RobotCollision.from_urdf(urdf)
    data = source / 'example_data/h2o'
    hand = np.load(data / 'mano/subject1/o1/milk.mano.npy')[:args.frames]
    poses = jaxlie.SE3.from_matrix(jnp.array(np.load(data / 'object_pose/subject1/o1/milk.object.npy')[:args.frames]))
    mesh = trimesh.load(data / 'object_mesh/milk/milk.obj', force='mesh')
    rng = np.random.default_rng(0)
    points = np.asarray(mesh.vertices)[rng.choice(len(mesh.vertices), 128, replace=False)]
    points = jnp.array(np.tile(points[None], (len(hand), 1, 1)), dtype=jnp.float32)
    source_indices, robot_indices = utils.get_mapping_from_mano_to_inspire(robot)
    print('Building upstream interaction mesh', flush=True)
    laplacian, neighbor_idx, neighbor_mask, degree = utils.create_laplacian_interaction_mesh(
        points, poses, jnp.array(hand), source_indices)
    print('Running upstream trajectory solver', flush=True)
    root, joints = module.solve_retargeting(robot, collision, points, poses, robot_indices,
                                           default_weights(), laplacian, neighbor_idx, neighbor_mask, degree)
    joints = np.array(joints)
    root = np.array(root.wxyz_xyz)
    if not np.isfinite(joints).all() or not np.isfinite(root).all():
        raise ValueError('Nonfinite upstream solution')
    np.savez_compressed(args.output / 'example.npz', joints=joints, root_wxyz_xyz=root)
    report = dict(upstream_revision=REVISION, example='H2O milk', frames=len(hand),
                  controlled_joints=list(robot.joints.actuated_names), devices=[str(x) for x in jax.devices()],
                  elapsed_seconds=time.monotonic()-start, finite=True,
                  scope='Headless upstream-example smoke test; not our video or physical grasp verification.')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
