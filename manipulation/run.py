"""Run the supplied demonstration-conditioned physical transfer experiment."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """Build targets and a grasp, execute physics, and optionally render/export observations."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/transfer"))
    parser.add_argument("--render", action="store_true", help="Requires graphics access")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def run(module, *items):
        subprocess.run([sys.executable, "-m", module, *map(str, items)], check=True)

    run("manipulation.targets", "--output", args.output / "targets")
    run("manipulation.fit", "--output", args.output / "grasp")
    run(
        "manipulation.rollout",
        "--fit",
        args.output / "grasp",
        "--task",
        args.output / "targets/task.json",
        "--output",
        args.output / "rollout",
    )
    if args.render:
        run("manipulation.preview", "--rollout", args.output / "rollout", "--output", args.output / "preview")
        run(
            "manipulation.export",
            "--rollout",
            args.output / "rollout",
            "--output",
            args.output / "observations",
        )


if __name__ == "__main__":
    main()
