"""Request closer-overlap tracking only at observed trajectory breaks.

Runs rigid fits and pose checks between requests; no pose interpolation. A fixed
attempt limit prevents repeatedly spending GPU time at an unsupported interval.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

if __package__:
    from .trackcraft_with_visibility import VisibilityRunner
else:
    from trackcraft_with_visibility import VisibilityRunner


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--taa-repo", type=Path, required=True)
    parser.add_argument("--trackcraft-repo", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--initial-pose", type=Path, required=True)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-bridges", type=int, default=16)
    parser.add_argument("--coherent-depth", action="store_true")
    parser.add_argument("--da3-repo", type=Path)
    parser.add_argument("--book-cameras", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    paths = [Path(p) for p in json.loads(args.windows.read_text())]
    count = json.loads((args.scene_dir / "input-manifest.json").read_text())["source_frames_read"]
    env = dict(os.environ, PYTHONPATH=str(args.project))
    common = ["--checkpoint", str(args.checkpoint), "--model-cache", str(args.model_cache),
              "--labels", "case", "--start-idx", "0", "--num-frames", "12", "--frame-stride", "1",
              "--height", "480", "--width", "832", "--no-track-video"]
    runner = VisibilityRunner(args.taa_repo, args.trackcraft_repo, common)
    attempts, gaps = [], []
    try:
        for iteration in range(args.max_bridges + 1):
            paths.sort(key=lambda p: json.loads((p / "input-manifest.json").read_text())["window_start"])
            combined = args.output / "windows.json"
            combined.write_text(json.dumps([str(p.resolve()) for p in paths], indent=2))
            fit_folder = args.output / f"fit_{iteration:03d}"
            command = [sys.executable, "-m", "reconstruction.stitch_object_windows",
                       "--windows", str(combined), "--scene-data", str(args.scene_dir),
                       "--initial-pose", str(args.initial_pose), "--surface", str(args.surface),
                       "--output", str(fit_folder)]
            with (args.output / f"fit_{iteration:03d}.log").open("w") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = json.loads((fit_folder / "report.json").read_text())
            latest = dict(fit_folder=str(fit_folder.resolve()), accepted_frames=report["accepted_frames"],
                          total_frames=count, additional_windows=len(attempts), attempts=attempts)
            (args.output / "latest.json").write_text(json.dumps(latest, indent=2) + "\n")
            print(f"Iteration {iteration}: accepted {report['accepted_frames']}/{count}", flush=True)
            if not report["missing_intervals"] or iteration == args.max_bridges:
                break
            gap = report["missing_intervals"][0]["first_frame"]
            gaps.append(gap)
            used = {json.loads((p / "input-manifest.json").read_text())["window_start"] for p in paths}
            choices = [max(0, min(count - 12, gap - n)) for n in [4, 2, 1, 6]]
            choices = [start for start in choices if start not in used]
            if not choices or gaps[-4:] == [gap] * 4:
                print(f"Stopping at unsupported interval beginning frame {gap}; no invented connection", flush=True)
                break
            start = choices[0]
            prepared = args.output / f"bridge_{iteration:03d}"
            subprocess.run([sys.executable, str(Path(__file__).with_name("make_windows.py")),
                            "--scene-dir", str(args.scene_dir), "--output", str(prepared),
                            "--starts", str(start)], check=True)
            new = Path(json.loads((prepared / "windows.json").read_text())[0])
            if args.coherent_depth:
                if args.da3_repo is None or args.book_cameras is None:
                    raise ValueError("Coherent depth requires DA3 and book camera observations")
                if __package__:
                    from . import full_depth
                else:
                    import full_depth
                (new / "da3").rename(new / "da3-batched-input")
                previous_argv = sys.argv
                sys.argv = ["full_depth.py", "--scene-dir", str(new), "--da3-root", str(args.da3_repo),
                            "--reference-da3", str(args.scene_dir / "da3"), "--book-cameras", str(args.book_cameras),
                            "--batch-size", "12"]
                try:
                    full_depth.main()
                finally:
                    sys.argv = previous_argv
            print(f"Tracking extra window at frame {start} to test gap {gap}", flush=True)
            runner.track(new)
            paths.append(new)
            attempts.append(dict(window_start=start, gap_frame=gap, window=str(new.resolve())))
    finally:
        runner.close()


if __name__ == "__main__":
    main()
