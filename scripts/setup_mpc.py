"""Fetch the pinned, dependency-light upstream MuJoCo predictive sampler."""

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / "manipulation/MPC_UPSTREAM.json").read_text())
    output = ROOT / "third_party/mujoco-mpc-python"
    output.mkdir(parents=True, exist_ok=True)
    for filename, remote in config["files"].items():
        target = output / filename
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == config["sha256"][filename]:
            continue
        url = f"https://raw.githubusercontent.com/google-deepmind/mujoco_mpc/{config['revision']}/{remote}"
        content = urllib.request.urlopen(url, timeout=60).read()
        if hashlib.sha256(content).hexdigest() != config["sha256"][filename]:
            raise ValueError("Upstream checksum mismatch")
        target.write_bytes(content)


if __name__ == "__main__":
    main()
