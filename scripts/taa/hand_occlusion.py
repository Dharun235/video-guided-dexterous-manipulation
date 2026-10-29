"""Rasterise the observed HaMeR hand for occlusion-aware image alignment."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--hands", type=Path, required=True)
    parser.add_argument("--taa", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads((args.hands / "manifest.json").read_text())
    source = json.loads((args.taa / "input-manifest.json").read_text())
    if original["source_sha256"] != source["sha256"]:
        raise ValueError("Hand and case recordings must match")
    args.output.mkdir(exist_ok=False)
    track = np.load(args.hands / "hand_track.npz")
    faces = np.load(args.hands / "hand_faces.npy")
    rows = json.loads((args.hands / "hands.json").read_text())
    shape = cv2.imread(str(args.taa / "frames/000000.jpg")).shape[:2]
    for i, valid in enumerate(track["valid"]):
        mask = np.zeros(shape, np.uint8)
        if valid:
            vertices = track["vertices_hand_centered_m"][i] + track["camera_translation_m"][i]
            xy = vertices[:, :2] / vertices[:, 2:]
            xy = xy * rows[i]["focal"] + [rows[i]["cx"], rows[i]["cy"]]
            for triangle in np.rint(xy[faces]).astype(np.int32):
                cv2.fillConvexPoly(mask, triangle, 255)
            mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
        # The observed visible case always overrides an imperfect hand silhouette.
        case = cv2.imread(str(args.taa / "masks/case" / f"{i:06d}.png"), 0)
        mask[case > 0] = 0
        cv2.imwrite(str(args.output / f"{i:06d}.png"), mask)
    report = dict(source_sha256=source["sha256"], frames=len(rows),
                  observed_hand_frames=int(track["valid"].sum()),
                  method="Projected HaMeR/MANO surface, dilated four pixels, visible case excluded",
                  limitation="Estimated occluder mask; missing hand detections are not filled")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
