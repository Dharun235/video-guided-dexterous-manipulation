"""Install the isolated CPU scene runtime and fetch checksum-pinned public models."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Prepare version-2 MoGe inference without installing CUDA-only version-3 extras."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--sam2", action="store_true", help="Also download optional SAM2.1 ONNX models")
    args = parser.parse_args()
    if not shutil.which("uv"):
        raise SystemExit("Install uv first: https://docs.astral.sh/uv/")
    manifest = json.loads((ROOT / "reconstruction/SCENE_MODELS.json").read_text())

    def run(*items):
        subprocess.run(list(map(str, items)), cwd=ROOT, check=True)

    env = ROOT / ".venv-scene"
    if not (env / "bin/python").exists():
        run("uv", "venv", "--python", "3.12", env)
    run(
        "uv",
        "pip",
        "install",
        "--python",
        env / "bin/python",
        "-r",
        ROOT / "reconstruction/requirements-scene.lock.txt",
        *(["--torch-backend", "cpu"] if sys.platform == "linux" else []),
    )
    source = ROOT / "third_party/moge-scene"
    pin = manifest["sources"]["moge_source"]
    if not source.exists():
        run("git", "clone", "--filter=blob:none", "--no-checkout", pin["repository"], source)
        run("git", "-C", source, "checkout", "--detach", pin["revision"])
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != pin["revision"]:
        raise SystemExit("Preserve the existing MoGe checkout before replacing a different revision.")
    run("uv", "pip", "install", "--python", env / "bin/python", "--no-deps", "-e", source)
    folder = ROOT / ".cache/scene-models"
    folder.mkdir(parents=True, exist_ok=True)
    for item in manifest["files"]:
        if item["name"].startswith("sam2") and not args.sam2:
            continue
        path = folder / item["name"]
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]:
            continue
        temporary = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(item["url"], temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != item["sha256"]:
            temporary.unlink()
            raise SystemExit(f"Checksum mismatch: {path.name}")
        temporary.replace(path)
    print("CPU scene runtime ready. Run .venv-scene/bin/python -m reconstruction.scene_check VIDEO")


if __name__ == "__main__":
    main()
