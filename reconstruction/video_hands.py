"""Extract every source frame into an auditable right-hand track on Apple Silicon."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .book_local import hand_pass, hand_video


def export_track(output):
    rows = json.loads((output / 'hands.json').read_text())
    count = len(rows)
    joints = np.full((count, 21, 3), np.nan)
    pixels = np.full((count, 21, 2), np.nan)
    vertices = np.full((count, 778, 3), np.nan)
    translation = np.full((count, 3), np.nan)
    rotation = np.full((count, 3, 3), np.nan)
    valid = np.zeros(count, dtype=bool)
    for index, row in enumerate(rows):
        hands = [h for h in row['hands'] if h['is_right']]
        if not hands:
            continue
        hand = hands[0]
        joints[index] = hand['keypoints_3d']
        pixels[index] = hand['keypoints_2d']
        vertices[index] = hand['vertices']
        translation[index] = hand['cam_t']
        rotation[index] = np.asarray(hand['global_orient']).reshape(3, 3)
        valid[index] = True
    frames = np.array([r['source_frame'] for r in rows])
    seconds = np.array([r['seconds'] for r in rows])
    if not np.array_equal(frames, np.arange(count)):
        raise ValueError('Full-video extraction must include every frame, including detection gaps')
    np.savez_compressed(output / 'hand_track.npz', source_frame=frames, seconds=seconds,
                        valid=valid, joints_hand_centered_m=joints,
                        joints_camera_m=joints + translation[:, None, :],
                        joints_image_px=pixels, vertices_hand_centered_m=vertices,
                        camera_translation_m=translation, global_orient=rotation)
    runs = []
    for index in np.flatnonzero(~valid):
        if runs and index == runs[-1][-1] + 1:
            runs[-1].append(int(index))
        else:
            runs.append([int(index)])
    summary = dict(frames=count, detected_frames=int(valid.sum()),
                   detection_coverage=float(valid.mean()),
                   missing_intervals=[dict(first_frame=r[0], last_frame=r[-1],
                                           start_s=float(seconds[r[0]]), end_s=float(seconds[r[-1]])) for r in runs],
                   track_sha256=hashlib.sha256((output / 'hand_track.npz').read_bytes()).hexdigest(),
                   coordinates='OpenCV camera: x right, y down, z forward; estimated metres, not calibrated world positions.',
                   landmarks='21 OpenPose-order hand landmarks; these are points, not 21 independent robot controls.',
                   missing_data='NaN with valid=false. No interpolation, invented body pose or hidden-hand filling.',
                   scope='Video hand estimates only; no object tracking, robot retargeting or physical grasp claim.')
    (output / 'coverage.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mano', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('.cache/hand-models'))
    args = parser.parse_args()
    hand_pass(args.video, args.output, args.mano, samples=0, cache=args.cache, right_hand=True)
    export_track(args.output)
    hand_video(args.output)


if __name__ == '__main__':
    main()
