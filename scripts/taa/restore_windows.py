"""Connect downloaded window outputs to local source frames and SAM3 masks."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--windows-dir", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--source-inputs", type=Path, required=True)
    parser.add_argument("--extra-window-root", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    scene, source = args.scene_dir.resolve(), args.source_inputs.resolve()
    for child in ["frames", "masks"]:
        path = scene / child
        if not path.exists():
            path.symlink_to(source / child, target_is_directory=True)
    windows = []
    candidates = list(args.windows_dir.glob("window_*"))
    for extra in args.extra_window_root:
        candidates.extend(extra.rglob("window_*"))
    candidates.sort(key=lambda p: json.loads((p / "input-manifest.json").read_text())["window_start"])
    for window in candidates:
        manifest = json.loads((window / "input-manifest.json").read_text())
        for child, suffix in [("frames", "jpg"), ("masks/case", "png"), ("masks/book", "png"), ("da3/depth", "npy")]:
            directory = window / child
            directory.mkdir(parents=True, exist_ok=True)
            for row in manifest["frames"]:
                path = directory / f"{row['frame_index']:06d}.{suffix}"
                if not path.is_symlink() and not path.exists():
                    path.symlink_to(scene / child / f"{row['source_frame']:06d}.{suffix}")
        windows.append(str(window.resolve()))
    # Preserve the received list as provenance; use local paths for execution.
    original = args.windows_dir / "windows.json"
    backup = args.windows_dir / "windows-remote.json"
    if original.exists() and not backup.exists():
        original.rename(backup)
    original.write_text(json.dumps(windows, indent=2) + "\n")
    print(f"Connected {len(windows)} windows to local inputs")


if __name__ == "__main__":
    main()
