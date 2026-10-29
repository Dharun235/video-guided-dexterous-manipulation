"""Batch DA3 depth at every frame, calibrating the gauge to the reviewed book.

This extends TAA stage 00 for memory-safe full-video input. Phone intrinsics are
fixed to the initial DA3 calibration. Camera motion and metric scale use the
same estimated book dimensions as the existing scene, not measured calibration.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

_MODELS = {}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--da3-root", type=Path, required=True)
    parser.add_argument("--reference-da3", type=Path, required=True)
    parser.add_argument("--book-cameras", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--model", default="depth-anything/DA3NESTED-GIANT-LARGE")
    parser.add_argument("--revision", default="8615eefb62f2db4f8d6ebaa59160086981672829")
    args = parser.parse_args()
    sys.path.insert(0, str(args.da3_root / "src"))
    import torch
    from depth_anything_3.api import DepthAnything3
    output = args.scene_dir / "da3"
    output.mkdir(exist_ok=False)
    (output / "depth").mkdir()
    paths = sorted((args.scene_dir / "frames").glob("*.jpg"))
    manifest = json.loads((args.scene_dir / "input-manifest.json").read_text())
    source_ids = [r["source_frame"] for r in manifest["frames"]]
    k = np.load(args.reference_da3 / "intrinsics.npz")["intrinsics"][0].astype(np.float64)
    observations = {r["source_frame"]: r for r in json.loads(args.book_cameras.read_text())}
    x, y, z = .198 / 2, .129 / 2, .018
    corners = np.array([[-x, y, z], [x, y, z], [x, -y, z], [-x, -y, z]])
    rotations, translations, extrinsics = [], [], []
    for i in range(len(paths)):
        ok, rvec, tvec = cv2.solvePnP(corners, np.asarray(observations[source_ids[i]]["book_quad"]), k, None)
        if not ok:
            raise ValueError(f"Book camera failed at frame {i}")
        r = cv2.Rodrigues(rvec)[0]
        rotations.append(r.T)
        translations.append((-r.T @ tvec).ravel())
        ext = np.eye(4)
        ext[:3, :3] = r
        ext[:3, 3] = tvec.ravel()
        extrinsics.append(ext)
    model_key = (args.model, args.revision)
    if model_key not in _MODELS:
        _MODELS[model_key] = DepthAnything3.from_pretrained(args.model, revision=args.revision).to("cuda").eval()
    model = _MODELS[model_key]
    start = time.monotonic()
    quality = []
    for first in range(0, len(paths), args.batch_size):
        last = min(first + args.batch_size, len(paths))
        images = [Image.open(path).convert("RGB") for path in paths[first:last]]
        with torch.no_grad():
            prediction = model.inference(image=images, intrinsics=np.repeat(k[None], len(images), axis=0),
                                         process_res=504, process_res_method="upper_bound_resize", export_dir=None)
        for local, image in enumerate(images):
            i = first + local
            width, height = image.size
            depth = cv2.resize(np.asarray(prediction.depth[local], np.float32), (width, height))
            book_mask = cv2.imread(str(args.scene_dir / "masks/book" / f"{i:06d}.png"), 0)
            ys, xs = np.where(cv2.erode(book_mask, np.ones((7, 7), np.uint8)) > 0)
            rays = np.c_[(xs - k[0, 2]) / k[0, 0], (ys - k[1, 2]) / k[1, 1], np.ones(len(xs))]
            # Intersect camera rays with z=book_top in the book world frame.
            ext = extrinsics[i]
            expected = (z + (ext[:3, :3].T @ ext[:3, 3])[2]) / (rays @ ext[:3, :3][:, 2])
            observed = depth[ys, xs]
            valid = np.isfinite(expected) & (expected > 0) & np.isfinite(observed) & (observed > 0)
            if valid.sum() < 1000:
                raise RuntimeError(f"Not enough book support for depth scale at {i}")
            scale = float(np.median(expected[valid] / observed[valid]))
            calibrated = depth * scale
            error = float(np.median(np.abs(calibrated[ys[valid], xs[valid]] - expected[valid])))
            if not np.isfinite(scale) or not .01 < scale < 100:
                raise RuntimeError(f"Invalid depth scale at {i}")
            np.save(output / "depth" / f"{i:06d}.npy", calibrated.astype(np.float32))
            quality.append(dict(frame=i, batch_start=first, scale=scale, book_median_error_m=error,
                                supported_pixels=int(valid.sum()), usable=error < .015))
        print(f"Depth frames {first}..{last-1} / {len(paths)}; elapsed {time.monotonic()-start:.1f}s", flush=True)
    indices = np.arange(len(paths))
    quaternions = Rotation.from_matrix(np.asarray(rotations)).as_quat().astype(np.float32)
    intrinsics = np.repeat(k[None].astype(np.float32), len(paths), axis=0)
    np.savez_compressed(output / "cameras.npz", frame_indices=indices,
                        cam_quats_xyzw=quaternions, cam_trans=np.asarray(translations, np.float32), intrinsics=intrinsics)
    np.savez_compressed(output / "intrinsics.npz", frame_indices=indices, intrinsics=intrinsics)
    np.save(output / "frame_indices.npy", indices)
    (output / "quality.json").write_text(json.dumps(quality, indent=2) + "\n")
    config = dict(scene_id=args.scene_dir.name, source="DA3 + reviewed book-plane camera/scale calibration",
                  da3_model=args.model, process_res=504, ref_frame=0, metric_book_frame=True,
                  model_revision=args.revision,
                  book_dimensions_prior_m=[.198, .129, .018], camera_intrinsics=k.tolist(),
                  frames=len(paths), batches=int(np.ceil(len(paths) / args.batch_size)),
                  source_frame_indices=source_ids,
                  usable_frames=sum(r["usable"] for r in quality), elapsed_seconds=time.monotonic()-start,
                  limitations=["Camera and scale use estimated book dimensions",
                               "Batchwise depth and hidden geometry are model predictions",
                               "Book-plane residual does not validate moving-object depth"])
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
