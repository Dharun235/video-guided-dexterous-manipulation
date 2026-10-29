"""Recompute only rejected tracking windows with jointly estimated depth."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

if __package__:
    from . import full_depth
    from .trackcraft_with_visibility import VisibilityRunner
else:
    import full_depth
    from trackcraft_with_visibility import VisibilityRunner


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--taa-repo", type=Path, required=True)
    parser.add_argument("--trackcraft-repo", type=Path, required=True)
    parser.add_argument("--da3-repo", type=Path, required=True)
    parser.add_argument("--book-cameras", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--starts-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    starts = json.loads(args.starts_json.read_text())
    subprocess.run([sys.executable, str(Path(__file__).with_name("make_windows.py")),
                    "--scene-dir", str(args.scene_dir), "--output", str(args.output),
                    "--starts", *map(str, starts)], check=True)
    common = ["--checkpoint", str(args.checkpoint), "--model-cache", str(args.model_cache),
              "--labels", "case", "--num-frames", "12", "--frame-stride", "1",
              "--height", "480", "--width", "832", "--no-track-video"]
    runner = VisibilityRunner(args.taa_repo, args.trackcraft_repo, common)
    try:
        for i, name in enumerate(json.loads((args.output / "windows.json").read_text())):
            scene = Path(name)
            print(f"Coherent repair {i+1}/{len(starts)}: {scene.name}", flush=True)
            (scene / "da3").rename(scene / "da3-batched-input")
            sys.argv = ["full_depth.py", "--scene-dir", str(scene), "--da3-root", str(args.da3_repo),
                        "--reference-da3", str(args.scene_dir / "da3"), "--book-cameras", str(args.book_cameras)]
            full_depth.main()
            runner.track(scene)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
