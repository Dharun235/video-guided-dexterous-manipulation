"""Package tracking outputs and metadata without models or duplicated inputs."""
import argparse
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--windows-name", default="taa-windows-01")
    parser.add_argument("--scene-name", default="taa-dense-01")
    parser.add_argument("--extra-names", nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with tarfile.open(args.output, "w:gz", compresslevel=1) as archive:
        for name in [args.windows_name, *args.extra_names]:
            for path in sorted((args.root / name).rglob("*")):
                if not path.is_file() or path.is_symlink() or path.name == "pointmap_ref.npy":
                    continue
                archive.add(path, arcname=path.relative_to(args.root), recursive=False)
        for relative in ["input-manifest.json", "da3/cameras.npz", "da3/intrinsics.npz",
                         "da3/frame_indices.npy", "da3/config.json", "da3/quality.json"]:
            path = args.root / args.scene_name / relative
            archive.add(path, arcname=path.relative_to(args.root), recursive=False)
    print(f"Saved {args.output.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
