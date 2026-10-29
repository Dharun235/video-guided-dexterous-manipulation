"""Estimate per-frame Book5 camera poses from the reviewed book plane."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .book_space import BOOK_QUAD, fit_camera


def register_camera(image, reference, k, features):
    sift, kp0, des0 = features
    kp, des = sift.detectAndCompute(image, None)
    if des is None:
        return None
    matches = cv2.BFMatcher().knnMatch(des0, des, k=2)
    good = [m for pair in matches if len(pair)==2 for m,n in [pair] if m.distance < .7*n.distance]
    if len(good) < 10:
        return None
    a = np.float32([kp0[m.queryIdx].pt for m in good])
    b = np.float32([kp[m.trainIdx].pt for m in good])
    hom, inliers = cv2.findHomography(a, b, cv2.RANSAC, 2.5)
    if hom is None or inliers.sum() < 10:
        return None
    quad = cv2.perspectiveTransform(BOOK_QUAD[None], hom)[0]
    _, r, t, error = fit_camera(quad, k[0, 0])
    mapped=cv2.perspectiveTransform(a[None],hom)[0]
    residual=np.linalg.norm(mapped-b,axis=1)[inliers.ravel()>0]
    return dict(rotation=r.ravel().tolist(), translation=t.ravel().tolist(),
                inliers=int(inliers.sum()), median_feature_fit_px=float(np.median(residual)),
                corner_fit_px=float(error.mean()), book_quad=quad.tolist())



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hands', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.hands / 'hands.json').read_text())
    scene = json.loads((args.scene / 'reconstruction.json').read_text())
    hand_manifest = json.loads((args.hands / 'manifest.json').read_text())
    if scene['source_sha256'] != hand_manifest['source_sha256']:
        raise ValueError('Scene and hand run must use the same source file')
    args.output.mkdir(parents=True, exist_ok=False)
    reference = cv2.imread(str(args.hands / 'frames' / '000000.jpg'))
    mask = np.zeros(reference.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, BOOK_QUAD.astype(int), 255)
    mask[220:325, 290:425] = 0
    sift = cv2.SIFT_create(nfeatures=2200)
    kp0, des0 = sift.detectAndCompute(reference, mask)
    intrinsics = np.asarray(scene['camera']['intrinsics'])
    cameras, failed = [], []
    for row in rows:
        frame = row['source_frame']
        image = cv2.imread(str(args.hands / 'frames' / f'{frame:06d}.jpg'))
        camera = register_camera(image, reference, intrinsics, (sift, kp0, des0))
        if camera is None:
            failed.append(frame)
        else:
            cameras.append(dict(seconds=row['seconds'], source_frame=frame, **camera))
    (args.output / 'camera_poses.json').write_text(json.dumps(cameras, indent=2) + '\n')
    report = dict(source_sha256=scene['source_sha256'], frames=len(rows), registered_frames=len(cameras),
                  missing_source_frames=failed, complete=not failed,
                  note='Book5 plane-based camera estimates, not ground truth. Missing registrations are not silently filled.')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if failed:
        raise SystemExit('Camera registration gaps require review before retargeting')


if __name__ == '__main__':
    main()
