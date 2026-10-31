"""Fetch pinned, simulation-only RH56 research code and hand assets."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ensure(path, spec, sparse=None):
    """Preserve existing revisions; fetch only a missing pinned checkout."""
    if (path / ".git").exists():
        revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if revision != spec["revision"]:
            raise SystemExit(f"Existing checkout has a different revision: {path}")
        return
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", spec["repository"], str(path)], check=True
    )
    if sparse:
        subprocess.run(["git", "-C", str(path), "sparse-checkout", "set", *sparse], check=True)
    subprocess.run(["git", "-C", str(path), "fetch", "--depth", "1", "origin", spec["revision"]], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", "--detach", spec["revision"]], check=True)


def main():
    """Download no policy weights, GPU stack, serial drivers or real-robot dependencies."""
    spec = json.loads((ROOT / "video_g1/RH56_UPSTREAM.json").read_text())
    path = ROOT / "third_party/rh56-controller-eval"
    ensure(path, spec["controller"])
    ensure(path / "h1_mujoco", spec["reference_hand"], ["inspire", "LICENSE"])
    print("Pinned RH56 planner ready. Use .venv-g1 for the headless evaluation.")


if __name__ == "__main__":
    main()
