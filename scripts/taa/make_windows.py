"""Prepare overlapping TAA scene windows without duplicating images or depth."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--starts", type=int, nargs="*")
    args = parser.parse_args()
    if not 0 < args.step < args.length:
        parser.error("Require overlapping windows: 0 < step < length")
    args.output.mkdir(exist_ok=False)
    source = args.scene_dir.resolve()
    manifest = json.loads((source / "input-manifest.json").read_text())
    camera = np.load(source / "da3/cameras.npz")
    quality = json.loads((source / "da3/quality.json").read_text())
    count = len(manifest["frames"])
    starts = sorted(set(args.starts if args.starts is not None else [*range(0, count - args.length + 1, args.step), count - args.length]))
    if any(start < 0 or start + args.length > count for start in starts):
        raise ValueError("Window outside source frame range")
    windows, rejected = [], []
    for start in starts:
        indices = np.arange(start, start + args.length)
        if not all(quality[i]["usable"] for i in indices):
            rejected.append(start)
            continue
        destination = args.output / f"window_{start:06d}"
        for child in ["frames", "masks/case", "masks/book", "da3/depth"]:
            (destination / child).mkdir(parents=True)
        for local, frame in enumerate(indices):
            for child, suffix in [("frames", "jpg"), ("masks/case", "png"), ("masks/book", "png"), ("da3/depth", "npy")]:
                (destination / child / f"{local:06d}.{suffix}").symlink_to(source / child / f"{frame:06d}.{suffix}")
        tracking = {"case": dict(prompt="black case", label="case", motion="moving", first_frame=0,
                                  last_frame=args.length-1, n_frames_visible=args.length, keyframe=0)}
        (destination / "masks/tracking.json").write_text(json.dumps(tracking))
        np.savez_compressed(destination / "da3/cameras.npz", frame_indices=np.arange(args.length),
                            cam_quats_xyzw=camera["cam_quats_xyzw"][indices],
                            cam_trans=camera["cam_trans"][indices], intrinsics=camera["intrinsics"][indices])
        np.savez_compressed(destination / "da3/intrinsics.npz", frame_indices=np.arange(args.length),
                            intrinsics=camera["intrinsics"][indices])
        config = dict(metric_book_frame=True, ref_frame=0, geometry_source="DA3 + book-plane metric calibration",
                      source_start_frame=start, source_frames=indices.tolist())
        (destination / "da3/config.json").write_text(json.dumps(config))
        records = [dict(frame_index=j, source_frame=int(i), timestamp_s=manifest["frames"][i]["timestamp_s"])
                   for j, i in enumerate(indices)]
        window_manifest = dict(source=manifest["source"], sha256=manifest["sha256"],
                               source_frames_read=manifest["source_frames_read"], fps=manifest["fps"],
                               frames=records, stride=1, window_start=start)
        (destination / "input-manifest.json").write_text(json.dumps(window_manifest, indent=2))
        windows.append(str(destination.resolve()))
    (args.output / "windows.json").write_text(json.dumps(windows, indent=2))
    (args.output / "report.json").write_text(json.dumps(dict(source_frames=count, length=args.length,
                                                             step=args.step, windows=len(windows),
                                                             rejected_depth_windows=rejected), indent=2))
    print(f"Prepared {len(windows)} windows; {len(rejected)} rejected for depth quality")


if __name__ == "__main__":
    main()
