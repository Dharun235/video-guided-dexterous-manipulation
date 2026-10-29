"""Select rejected rigid-fit windows for targeted coherent-depth repair."""
import argparse
import json
from pathlib import Path

from reconstruction.rigid_tracks import window_motion


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    starts = []
    for scene in map(Path, json.loads(args.windows.read_text())):
        motion = window_motion(scene)
        if any(row["estimate"] is None for row in motion.values()):
            manifest = json.loads((scene / "input-manifest.json").read_text())
            starts.append(manifest["window_start"])
    args.output.write_text(json.dumps(starts, indent=2) + "\n")
    print(f"Selected {len(starts)} windows")


if __name__ == "__main__":
    main()
