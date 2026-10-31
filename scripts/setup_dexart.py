"""Install the pinned retargeting solver without its unrelated demo datasets."""

import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATTERNS = ["/README.md", "/retarget_h2o.py", "/utils.py"]


def main():
    """Fetch solver sources and prepare the separate PyRoki/JAX CPU environment."""
    pins = json.loads((ROOT / "retargeting/UPSTREAM.json").read_text())
    target = ROOT / "third_party/humanoid-dexart-retarget"
    if not target.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--sparse",
                "--no-checkout",
                pins["repository"],
                str(target),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(target), "sparse-checkout", "set", "--no-cone", *SOURCE_PATTERNS], check=True
        )
        subprocess.run(
            ["git", "-C", str(target), "fetch", "--depth", "1", "origin", pins["revision"]], check=True
        )
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", pins["revision"]], check=True)
    actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    if actual != pins["revision"]:
        raise SystemExit(
            "Existing retargeter checkout has a different revision; preserve it before replacing"
        )
    subprocess.run(
        ["git", "-C", str(target), "sparse-checkout", "set", "--no-cone", *SOURCE_PATTERNS], check=True
    )
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("Install uv first")
    env = dict(os.environ, UV_CACHE_DIR=str(ROOT / ".cache/uv"))
    python = ROOT / ".venv-dexart/bin/python"
    if not python.exists():
        subprocess.run([uv, "venv", "--python", "3.12", str(ROOT / ".venv-dexart")], check=True, env=env)
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "-r",
            str(ROOT / "retargeting/requirements-mac.lock.txt"),
        ],
        check=True,
        env=env,
    )
    subprocess.run(
        [
            str(python),
            "-c",
            "from retargeting.upstream import load; load(); import jax; print(jax.devices())",
        ],
        cwd=ROOT,
        check=True,
    )
    print("Pinned upstream retargeter ready in .venv-dexart")


if __name__ == "__main__":
    main()
