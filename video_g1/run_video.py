"""Build and evaluate the complete Book5 physical motion from published inputs."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    """Run calibrated scene, native upper-body planning, contact control and optional preview."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/video"))
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    timings = []
    started = time.monotonic()

    def run(module, *items):
        before = time.monotonic()
        subprocess.run([sys.executable, "-m", module, *map(str, items)], check=True)
        timings.append({"stage": module, "elapsed_s": time.monotonic() - before})
        (args.output / "pipeline.json").write_text(
            json.dumps(
                {
                    "stages": timings,
                    "elapsed_s": time.monotonic() - started,
                    "gpu_used": False,
                    "rendered": args.render,
                },
                indent=2,
            )
            + "\n"
        )

    run(
        "video_g1.calibration",
        "--config",
        "configs/book5.json",
        "--camera-observations",
        "data/book5/inputs/camera-observations.json",
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
        "data/book5/inputs/tracking-map",
        "--case-masks",
        "data/book5/inputs/masks",
        "--output",
        args.output / "scene",
    )
    run("video_g1.wrap_grasp", "--scene", args.output / "scene", "--output", args.output / "grasp")
    run(
        "video_g1.book5_sequence",
        "--scene",
        args.output / "scene",
        "--grasp",
        args.output / "grasp/reference.npz",
        "--output",
        args.output / "reference",
    )
    run("video_g1.sequence", "--reference", args.output / "reference", "--output", args.output / "rollout")
    run(
        "video_g1.sequence_replay",
        "--reference",
        args.output / "reference",
        "--rollout",
        args.output / "rollout",
        "--output",
        args.output / "replay-check.json",
    )
    if args.render:
        run(
            "video_g1.sequence_preview",
            "--reference",
            args.output / "reference",
            "--rollout",
            args.output / "rollout",
            "--cameras",
            args.output / "cameras",
            "--source",
            "data/book5/video.mp4",
            "--output",
            args.output / "preview",
        )


if __name__ == "__main__":
    main()
