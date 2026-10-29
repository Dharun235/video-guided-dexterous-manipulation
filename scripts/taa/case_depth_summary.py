"""Summarise visible-case depth for explicitly uncertain pose reinitialisation."""
import argparse
import io
import json
from pathlib import Path
import tarfile

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--taa", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.taa / "input-manifest.json").read_text())
    values = np.full(manifest["source_frames_read"], np.nan)

    def consume(frame, depth):
        mask = cv2.imread(str(args.taa / "masks/case" / f"{frame:06d}.png"), 0) > 0
        selected = depth[mask & np.isfinite(depth) & (depth > 0)]
        if len(selected):
            values[frame] = np.median(selected)

    if args.archive:
        with tarfile.open(args.archive, "r:gz") as archive:
            for member in archive:
                if member.name.startswith("da3/depth/") and member.name.endswith(".npy"):
                    consume(int(Path(member.name).stem), np.load(io.BytesIO(archive.extractfile(member).read())))
    else:
        for path in sorted((args.taa / "da3/depth").glob("*.npy")):
            consume(int(path.stem), np.load(path))
    np.save(args.taa / "case_depth_camera_m.npy", values)
    notes = dict(source_sha256=manifest["sha256"], frames=len(values), finite_frames=int(np.isfinite(values).sum()),
                 method="Median DA3 book-calibrated z over visible SAM3 case pixels",
                 limitation="Predicted surface depth, not measured object-centre positions")
    (args.taa / "case_depth_notes.json").write_text(json.dumps(notes, indent=2) + "\n")


if __name__ == "__main__":
    main()
