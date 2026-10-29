"""Write TAA JPEG inputs, timestamps and source checksum without changing playback."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    frames = args.output / "frames"
    frames.mkdir()
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    records = []
    source_index = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if source_index % args.stride == 0 or source_index == count - 1:
            index = len(records)
            path = frames / f"{index:06d}.jpg"
            if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Cannot write {path}")
            records.append(dict(frame_index=index, source_frame=source_index,
                                timestamp_s=source_index / fps))
        source_index += 1
    cap.release()
    manifest = dict(source=args.video.name,
                    sha256=hashlib.sha256(args.video.read_bytes()).hexdigest(),
                    fps=fps, source_frames_read=source_index, stride=args.stride,
                    frames=records)
    (args.output / "input-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    prompts = dict(prompts=[
        dict(text="book", frame_index=0, label="book", motion="static", keyframe_candidates=[0]),
        dict(text="black case", frame_index=0, label="case", motion="moving", keyframe_candidates=[0])])
    (args.output / "prompts.json").write_text(json.dumps(prompts, indent=2) + "\n")
    print(f"Wrote {len(records)} frames from {source_index} source frames")


if __name__ == "__main__":
    main()
