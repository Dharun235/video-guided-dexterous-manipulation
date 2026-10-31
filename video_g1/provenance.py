"""Record runtime and input identities without including credentials or model data."""

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import scipy


def write_manifest(output, inputs):
    """Write software versions and SHA-256 checksums of the stage's immediate inputs."""
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    files = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs}
    sources = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "video_g1").glob("*.py"))
    }
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "git_head": revision.stdout.strip(),
        "input_sha256": files,
        "implementation_sha256": sources,
    }
    (output / "runtime.json").write_text(json.dumps(report, indent=2) + "\n")
