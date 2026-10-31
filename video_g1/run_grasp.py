"""Build the measured Book5 scene and physically replay the verified grasp plan."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """Run saved-input calibration, scene construction and free-object simulation."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/grasp"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def run(module, *items):
        subprocess.run([sys.executable, "-m", module, *map(str, items)], check=True)

    run(
        "video_g1.calibration",
        "--config",
        "configs/book5.json",
        "--camera-observations",
        "examples/full-video/camera-observations.json",
        "--output",
        args.output / "cameras",
    )
    run(
        "video_g1.scene",
        "--config",
        "configs/book5.json",
        "--cameras",
        args.output / "cameras",
        "--appearance",
        "examples/object-motion/scene",
        "--case-masks",
        "examples/full-video/case-masks",
        "--output",
        args.output / "scene",
    )
    run(
        "video_g1.rh56_native",
        "--scene",
        args.output / "scene",
        "--reference",
        "examples/rh56-evaluation/reference.npz",
        "--force-stop",
        "--output",
        args.output / "execution",
    )


if __name__ == "__main__":
    main()
