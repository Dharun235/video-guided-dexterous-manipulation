"""Download pinned Menagerie models and create the independent CPU runtime."""

import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "humanoid/UPSTREAM.json").read_text())
    source = ROOT / "third_party/menagerie"
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("Install uv first: https://docs.astral.sh/uv/")

    def run(*args):
        subprocess.run(list(map(str, args)), check=True, cwd=ROOT)

    base_patterns=["/unitree_g1/g1.xml", "/unitree_g1/LICENSE", "/unitree_g1/README.md",
                   "/unitree_g1/CHANGELOG.md", "/unitree_g1/g1.png"]

    if not source.exists():
        run("git", "clone", "--filter=blob:none", "--sparse", "--no-checkout", manifest["repository"], source)
        run("git", "-C", source, "sparse-checkout", "set", "--no-cone", *base_patterns)
        run("git", "-C", source, "fetch", "--depth", "1", "origin", manifest["revision"])
        run("git", "-C", source, "checkout", "--detach", manifest["revision"])
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != manifest["revision"]:
        raise SystemExit(
            "Existing Menagerie checkout is not the pinned revision; preserve it before replacing it."
        )
    if not (source/"unitree_g1/g1.xml").exists():
        run("git", "-C", source, "sparse-checkout", "add", "--no-cone", *base_patterns)
    robot=ET.parse(source/"unitree_g1/g1.xml")
    assets=["/unitree_g1/assets/"+a.get("file") for a in robot.findall("asset/*") if a.get("file")]
    run("git", "-C", source, "sparse-checkout", "set", "--no-cone", *base_patterns, *assets)
    environment = ROOT / ".venv-g1"
    if not (environment / "bin/python").exists():
        run(uv, "venv", "--python", manifest["python"], environment)
    run(
        uv,
        "pip",
        "install",
        "--python",
        environment / "bin/python",
        "-r",
        ROOT / "humanoid/requirements.lock.txt",
    )
    print("G1 ready. Run python3 -m scripts.setup_book_g1, then python3 -m humanoid --help")


if __name__ == "__main__":
    main()
