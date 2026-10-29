"""Fit the observed right hand to G1/Inspire joints; kinematics only, without a task script."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from .book_g1 import make_model, ORIGIN, SCENE_ROTATION, BACKGROUND_CONTROL
from .book_transfer import set_configuration


def camera_track(camera_rows, times):
    seconds = np.array([r['seconds'] for r in camera_rows])
    query = np.clip(times, seconds[0], seconds[-1])
    rotations = Slerp(seconds, Rotation.from_rotvec([r['rotation'] for r in camera_rows]))(query).as_matrix()
    positions = np.array([r['translation'] for r in camera_rows])
    translation = np.column_stack([np.interp(query, seconds, positions[:, i]) for i in range(3)])
    return rotations, translation


def run(args):
    rows = json.loads((args.hands / 'hands.json').read_text())
    manifest = json.loads((args.hands / 'manifest.json').read_text())
    scene = json.loads((args.scene / 'reconstruction.json').read_text())
    camera_rows = json.loads(args.cameras.read_text())
    if scene['source_sha256'] != manifest['source_sha256']:
        raise ValueError('Hand estimates and scene must use the same source file')
    camera_report = args.cameras.parent / 'report.json'
    if camera_report.exists() and json.loads(camera_report.read_text()).get('complete') is False:
        raise ValueError('Review missing camera registrations before retargeting')
    args.output.mkdir(parents=True, exist_ok=False)
    model, active, state = make_model(args.scene, args.output)
    data = mujoco.MjData(model)
    lo, hi = model.actuator_ctrlrange[active].T.copy()
    times = np.array([r['seconds'] for r in rows])
    camera_r, camera_t = camera_track(camera_rows, times)
    intrinsics = np.array(scene['camera']['intrinsics'])
    actions = np.full((len(rows), len(active)), np.nan)
    targets = np.full((len(rows), 21, 3), np.nan)
    predicted = np.full_like(targets, np.nan)
    errors = np.full((len(rows), 21), np.nan)
    reprojection = np.full(len(rows), np.nan)
    seed = np.clip(np.array([-.5, -.2, 0, 1, 0, 0, 0, .3, .3, .3, .3, .2, .5]), lo + 1e-6, hi - 1e-6)
    sites = [model.site(f'landmark_{i}').id for i in range(21)]
    for i, row in enumerate(rows):
        hands = [h for h in row['hands'] if h['is_right']]
        if not hands:
            continue
        hand = hands[0]
        local = np.asarray(hand['keypoints_3d'], dtype=float)
        pixels = np.asarray(hand['keypoints_2d'], dtype=float)
        ok, rvec, tvec = cv2.solvePnP(local, pixels, intrinsics, None)
        if not ok or tvec[2, 0] <= 0:
            continue
        camera = local @ cv2.Rodrigues(rvec)[0].T + tvec.ravel()
        world = (camera - camera_t[i]) @ camera_r[i]
        target = world @ SCENE_ROTATION.T + ORIGIN
        uv = cv2.projectPoints(local, rvec, tvec, intrinsics, None)[0].reshape(-1, 2)
        reprojection[i] = np.linalg.norm(uv - pixels, axis=1).mean()
        previous = seed.copy()
        def objective(u):
            control = np.zeros(model.nu)
            control[active] = u
            for name, value in BACKGROUND_CONTROL.items():
                control[model.actuator(name).id] = value
            set_configuration(model, data, control)
            points = data.site_xpos[sites].copy()
            # Absolute landmarks constrain reach and orientation; wrist-local
            # landmarks preserve finger shape when exact human reach is impossible.
            return np.r_[(points - target).ravel(),
                         .5 * ((points[1:] - points[0]) - (target[1:] - target[0])).ravel(),
                         .002 * (u - previous)]
        fit = least_squares(objective, seed, bounds=(lo + 1e-7, hi - 1e-7), max_nfev=args.iterations)
        seed = fit.x
        objective(seed)
        actions[i] = seed
        targets[i] = target
        predicted[i] = data.site_xpos[sites]
        errors[i] = np.linalg.norm(predicted[i] - target, axis=1)
        if i % 60 == 0:
            print(f'Frame {i}: mean landmark residual {errors[i].mean() * 1000:.1f} mm', flush=True)
    valid = np.isfinite(actions).all(axis=1)
    if not valid.any():
        raise ValueError('No usable hand targets')
    np.savez_compressed(args.output / 'retarget.npz', source_frame=[r['source_frame'] for r in rows],
                        seconds=times, valid=valid, actions=actions, target_landmarks_m=targets,
                        fitted_landmarks_m=predicted, landmark_error_m=errors)
    schema = dict(action_names=[model.actuator(int(a)).name for a in active],
                  state_joint_names=[model.joint(int(j)).name for j in state])
    (args.output / 'joint-map.json').write_text(json.dumps(schema, indent=2) + '\n')
    report = dict(source_sha256=manifest['source_sha256'], frames=len(rows), fitted_frames=int(valid.sum()),
                  mean_landmark_residual_m=float(np.nanmean(errors)),
                  mean_wrist_residual_m=float(np.nanmean(errors[:, 0])),
                  mean_tip_residual_m=float(np.nanmean(errors[:, [4, 8, 12, 16, 20]])),
                  mean_predicted_keypoint_reprojection_px=float(np.nanmean(reprojection)),
                  hands_track_sha256=hashlib.sha256((args.hands / 'hands.json').read_bytes()).hexdigest(),
                  method='Bounded joint landmark IK using the existing MuJoCo model and SciPy; not the PyRoki upstream solver.',
                  source_timing='All input frames retained; no scripted grasp/carry/release or time cutoff.',
                  camera_assumption='Uses reviewed Book5 scene calibration and interpolated existing camera estimates.',
                  limitations=['Residuals compare model estimates, not ground-truth human motion.',
                               'NaN actions mark missing/unusable detections; do not execute this file directly.',
                               'Kinematic fit only: no contact optimization, collision acceptance or physical grasp established.',
                               'Camera and scene calibration must be regenerated/reviewed for a different recording.'])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hands', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--cameras', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=35)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
