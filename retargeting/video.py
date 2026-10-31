"""Use the upstream Laplacian trajectory solver for our observed right hand."""

import argparse
import json
import time
from pathlib import Path

import cv2
import jax.numpy as jnp
import jaxlie
import jaxls
import mujoco
import numpy as np
import pyroki as pk
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from yourdfpy import URDF

from humanoid.book_g1 import BACKGROUND_CONTROL, ORIGIN, SCENE_ROTATION, make_model
from humanoid.kinematics import set_configuration
from humanoid.retarget_hand import camera_track
from reconstruction.book_space import BOOK_SIZE, corners

from .model_bridge import export_robot
from .upstream import REVISION, default_weights, load


def install_supported_root(root_position):
    """Add a numerical support constraint to the unchanged upstream objective."""
    original = jaxls.LeastSquaresProblem
    reference = jaxlie.SE3.from_translation(jnp.asarray(root_position, dtype=jnp.float32))

    @jaxls.Cost.create_factory
    def support(values, variable):
        return (reference.inverse() @ values[variable]).log() * 10000.0

    def supported_problem(costs, variables, *args, **kwargs):
        return original([*costs, support(variables[1])], variables, *args, **kwargs)

    jaxls.LeastSquaresProblem = supported_problem
    return original


def verify_fk(model, active, robot, root_position):
    """Check exported URDF landmarks against native MuJoCo forward kinematics."""
    rng = np.random.default_rng(1)
    name_to_index = {n: i for i, n in enumerate(robot.joints.actuated_names)}
    links = [robot.links.names.index(f"landmark_{i}") for i in range(21)]
    data = mujoco.MjData(model)
    maximum = 0.0
    for _ in range(5):
        control = np.zeros(model.nu)
        control[active] = rng.uniform(*model.actuator_ctrlrange[active].T)
        for name, value in BACKGROUND_CONTROL.items():
            control[model.actuator(name).id] = value
        set_configuration(model, data, control)
        cfg = np.zeros(len(name_to_index))
        for a in active:
            cfg[name_to_index[model.actuator(int(a)).name]] = control[a]
        points = (
            np.asarray(jaxlie.SE3(robot.forward_kinematics(jnp.asarray(cfg))).translation())[links]
            + root_position
        )
        actual = data.site_xpos[[model.site(f"landmark_{i}").id for i in range(21)]]
        maximum = max(maximum, float(np.max(np.linalg.norm(points - actual, axis=1))))
    if maximum > 1e-5:
        raise ValueError(f"URDF/MuJoCo landmark geometry differs by {maximum} m")
    return maximum


def run(args):
    """Fit selected observed frames with the upstream kinematic interaction objective."""
    module, utils, _ = load()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = json.loads((args.hands / "hands.json").read_text())
    source = json.loads((args.hands / "manifest.json").read_text())
    scene = json.loads((args.scene / "reconstruction.json").read_text())
    camera_report = json.loads((args.cameras.parent / "report.json").read_text())
    if scene["source_sha256"] != source["source_sha256"] or not camera_report["complete"]:
        raise ValueError("Require matching source hashes and complete reviewed camera registration")
    times = np.array([r["seconds"] for r in rows])
    rotations, translations = camera_track(json.loads(args.cameras.read_text()), times)
    world = np.full((len(rows), 21, 3), np.nan)
    intrinsics = np.array(scene["camera"]["intrinsics"])
    for i, row in enumerate(rows):
        hands = [h for h in row["hands"] if h["is_right"]]
        if not hands:
            continue
        h = hands[0]
        local, uv = np.asarray(h["keypoints_3d"]), np.asarray(h["keypoints_2d"])
        ok, rvec, tvec = cv2.solvePnP(local, uv, intrinsics, None)
        if ok and tvec[2, 0] > 0:
            camera = local @ cv2.Rodrigues(rvec)[0].T + tvec.ravel()
            world[i] = ((camera - translations[i]) @ rotations[i]) @ SCENE_ROTATION.T + ORIGIN
    model, active, state = make_model(args.scene, args.output)
    urdf_path, root_position = export_robot(model, active, args.output)
    urdf = URDF.load(urdf_path)
    robot = pk.Robot.from_urdf(urdf)
    if args.initial_reference:
        baseline = np.load(args.initial_reference / "retarget.npz")
        baseline_names = json.loads((args.initial_reference / "joint-map.json").read_text())["action_names"]
        baseline_report = json.loads((args.initial_reference / "report.json").read_text())
        if baseline_report["source_sha256"] != source["source_sha256"]:
            raise ValueError("Initialization must use the same source recording")
        first = int(np.flatnonzero(baseline["valid"] & (np.arange(len(rows)) >= args.start_frame))[0])
        initial = np.array(
            [baseline["actions"][first, baseline_names.index(name)] for name in robot.joints.actuated_names]
        )
        robot = pk.Robot.from_urdf(urdf, default_joint_cfg=initial)
    else:
        # This one-frame IK supplies an initial configuration, not the output
        # trajectory. All selected motion frames are optimized by upstream.
        first = int(
            np.flatnonzero(np.isfinite(world).all(axis=(1, 2)) & (np.arange(len(rows)) >= args.start_frame))[
                0
            ]
        )
        workspace = mujoco.MjData(model)
        sites = [model.site(f"landmark_{i}").id for i in range(21)]
        low, high = model.actuator_ctrlrange[active].T

        def initial_error(values):
            control = np.zeros(model.nu)
            control[active] = values
            for name, value in BACKGROUND_CONTROL.items():
                control[model.actuator(name).id] = value
            set_configuration(model, workspace, control)
            return (workspace.site_xpos[sites] - world[first]).ravel()

        seed = np.clip(
            np.array([-0.5, -0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.3, 0.2, 0.5]),
            low + 1e-6,
            high - 1e-6,
        )
        initial_fit = least_squares(initial_error, seed, bounds=(low + 1e-7, high - 1e-7), max_nfev=100)
        active_names = [model.actuator(int(a)).name for a in active]
        initial = np.array([initial_fit.x[active_names.index(name)] for name in robot.joints.actuated_names])
        robot = pk.Robot.from_urdf(urdf, default_joint_cfg=initial)
    collision = pk.collision.RobotCollision.from_urdf(urdf)
    fk_error = verify_fk(model, active, robot, root_position)
    print(f"URDF/MuJoCo landmark agreement: {fk_error:.3g} m", flush=True)
    # Use static book anchors by default, or unflagged moving-case anchors below.
    anchors = np.vstack([corners(0), corners(BOOK_SIZE[2])]) @ SCENE_ROTATION.T + ORIGIN
    object_track = None
    if args.object_track:
        object_track = np.load(args.object_track / "object_track.npz")
        if not np.array_equal(object_track["source_frame"], np.arange(len(rows))):
            raise ValueError("Object and hand source timelines must match")
        surface = np.load(args.object_track / "case_surface.npy")
        anchors = surface[np.linspace(0, len(surface) - 1, 16).astype(int)]
    ids = np.array([i for i in range(0, len(rows), args.stride) if np.isfinite(world[i]).all()])
    if object_track is not None:
        ids = ids[object_track["valid"][ids] & ~object_track["requires_review"][ids]]
    ids = ids[ids >= args.start_frame]
    if args.end_frame is not None:
        ids = ids[ids <= args.end_frame]
    if args.maximum_frames:
        ids = ids[: args.maximum_frames]
    if not len(ids):
        raise ValueError("No usable observed right-hand frames")
    split_indices = np.flatnonzero(np.diff(ids) > args.stride) + 1
    segments = np.split(ids, split_indices)
    actions = np.full((len(rows), len(active)), np.nan)
    root_deviation = []
    robot_ids = jnp.array([robot.links.names.index(f"landmark_{i}") for i in range(21)])
    order = [robot.joints.actuated_names.index(model.actuator(int(a)).name) for a in active]
    weights = default_weights()
    weights.update(
        laplacian_deformation=50.0,
        joint_smoothness=0.03,
        root_smoothness=0.0,
        floor_contact=0.0,
        foot_skating=0.0,
        posture_prior=0.0,
        arm_prior=0.0,
        root_upright=0.0,
        self_collision=0.2,
    )
    original_problem = install_supported_root(np.zeros(3))
    original_limit = pk.costs.limit_cost
    pk.costs.limit_cost = lambda robot_arg, variable, weight: original_limit(
        robot_arg, variable, weight * 100.0
    )
    try:
        for segment in segments:
            for offset in range(0, len(segment), args.window):
                chosen = segment[offset : offset + args.window]
                padded = np.pad(chosen, (0, args.window - len(chosen)), mode="edge")
                if object_track is None:
                    points = jnp.array(
                        np.tile((anchors - root_position)[None], (len(padded), 1, 1)), dtype=jnp.float32
                    )
                    poses = jaxlie.SE3.identity((len(padded),))
                else:
                    points = jnp.asarray(np.tile(anchors[None], (len(padded), 1, 1)), dtype=jnp.float32)
                    rotations_case = SCENE_ROTATION @ object_track["rotation_world"][padded]
                    xyzw = Rotation.from_matrix(rotations_case).as_quat()
                    positions_case = (
                        object_track["translation_world_m"][padded] @ SCENE_ROTATION.T
                        + ORIGIN
                        - root_position
                    )
                    poses = jaxlie.SE3(
                        jnp.asarray(
                            np.column_stack([xyzw[:, [3, 0, 1, 2]], positions_case]), dtype=jnp.float32
                        )
                    )
                laplacian, neighbor_idx, neighbor_mask, degree = utils.create_laplacian_interaction_mesh(
                    points,
                    poses,
                    jnp.asarray(world[padded] - root_position, dtype=jnp.float32),
                    jnp.arange(21),
                )
                # Keep topology-array shapes constant across CPU windows to
                # avoid recompiling the same solver for different degrees.
                pad = len(anchors) + 21 - neighbor_idx.shape[-1]
                neighbor_idx = jnp.pad(neighbor_idx, ((0, 0), (0, 0), (0, pad)))
                neighbor_mask = jnp.pad(neighbor_mask, ((0, 0), (0, 0), (0, pad)))
                root, result = module.solve_retargeting(
                    robot,
                    collision,
                    points,
                    poses,
                    robot_ids,
                    weights,
                    laplacian,
                    neighbor_idx,
                    neighbor_mask,
                    degree,
                )
                q = np.asarray(result)[: len(chosen)]
                root_values = root.wxyz_xyz[: len(chosen)]
                delta = np.asarray((jaxlie.SE3.identity().inverse() @ jaxlie.SE3(root_values)).log())
                root_deviation.append(float(np.max(np.abs(delta))))
                if root_deviation[-1] > 0.0001:
                    raise ValueError("Supported root moved beyond numerical tolerance")
                if not np.isfinite(q).all():
                    raise ValueError("Nonfinite upstream result")
                # The original limit penalty is soft. Export explicit bounded
                # motor proposals and record the raw excess separately below.
                actions[chosen] = q[:, order]
                print(f"Upstream solver fitted source frames {chosen[0]}..{chosen[-1]}", flush=True)
    finally:
        jaxls.LeastSquaresProblem = original_problem
        pk.costs.limit_cost = original_limit
    valid = np.isfinite(actions).all(axis=1)
    data = mujoco.MjData(model)
    fitted = np.full_like(world, np.nan)
    landmark_ids = [model.site(f"landmark_{i}").id for i in range(21)]
    limit_excess = 0.0
    clipped_components = 0
    for i in np.flatnonzero(valid):
        limit_excess = max(
            limit_excess,
            float(np.maximum(model.actuator_ctrlrange[active, 0] - actions[i], 0).max()),
            float(np.maximum(actions[i] - model.actuator_ctrlrange[active, 1], 0).max()),
        )
        bounded = np.clip(actions[i], *model.actuator_ctrlrange[active].T)
        clipped_components += int(np.count_nonzero(bounded != actions[i]))
        actions[i] = bounded
        control = np.zeros(model.nu)
        control[active] = actions[i]
        for name, value in BACKGROUND_CONTROL.items():
            control[model.actuator(name).id] = value
        set_configuration(model, data, control)
        fitted[i] = data.site_xpos[landmark_ids]
    errors = np.linalg.norm(fitted - world, axis=2)
    np.savez_compressed(
        args.output / "retarget.npz",
        source_frame=[r["source_frame"] for r in rows],
        seconds=times,
        valid=valid,
        observed=np.isfinite(world).all(axis=(1, 2)),
        actions=actions,
        target_landmarks_m=world,
        fitted_landmarks_m=fitted,
        landmark_error_m=errors,
    )
    schema = dict(
        action_names=[model.actuator(int(a)).name for a in active],
        state_joint_names=[model.joint(int(j)).name for j in state],
    )
    (args.output / "joint-map.json").write_text(json.dumps(schema, indent=2) + "\n")
    fitted_ids = np.flatnonzero(valid)
    largest_step = float(np.abs(np.diff(actions[fitted_ids], axis=0)).max()) if len(fitted_ids) > 1 else 0.0
    report = dict(
        method="humanoid-dexart-retarget.solve_retargeting + upstream Laplacian interaction mesh; PyRoki/JAX on CPU",
        upstream_revision=REVISION,
        source_sha256=source["source_sha256"],
        frames=len(rows),
        observed_frames=int(np.isfinite(world).all(axis=(1, 2)).sum()),
        fitted_frames=int(valid.sum()),
        stride=args.stride,
        window=args.window,
        weights=weights,
        mean_wrist_residual_m=float(np.nanmean(errors[:, 0])),
        mean_tip_residual_m=float(np.nanmean(errors[:, [4, 8, 12, 16, 20]])),
        maximum_root_deviation=root_deviation and max(root_deviation),
        maximum_joint_limit_excess_rad=limit_excess,
        fk_agreement_maximum_m=fk_error,
        elapsed_seconds=time.monotonic() - started,
        ready_for_physics=False,
        largest_target_step_rad=largest_step,
        maximum_root_deviation_units="Largest absolute SE3 tangent component: translation metres, rotation radians",
        upstream_files_modified=False,
        limit_penalty_multiplier=100.0,
        clipped_action_components=clipped_components,
        initialization="First valid joint configuration from the supplied baseline only; no baseline trajectory targets"
        if args.initial_reference
        else "Single observed-frame bounded IK seed; final trajectory optimized by upstream",
        reference="Unflagged moving-case poses and local surface anchors"
        if args.object_track
        else "Static reconstructed book anchors",
        limitations=[
            "Kinematic fit, not physical execution.",
            "Capsule self-collision penalty uses box approximations of the original collision shapes.",
            "Only selected observed frames are solved; unsolved and missing frames stay invalid.",
            "Window boundaries are not smoothed across windows; inspect continuity before execution.",
            "Residuals are against estimated human landmarks, not measured ground truth.",
        ],
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    """Retarget recorded right-hand observations and preserve unsolved source rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--maximum-frames", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--initial-reference", type=Path)
    parser.add_argument(
        "--object-track",
        type=Path,
        help="Directory with object_track.npz and case_surface.npy; flagged poses are excluded",
    )
    args = parser.parse_args()
    if args.stride < 1 or args.window < 2:
        parser.error("Positive stride and window >= 2 required")
    run(args)


if __name__ == "__main__":
    main()
