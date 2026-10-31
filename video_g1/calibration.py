"""Fit a shared camera and per-frame book-plane poses from explicit dimensions.

Length/width/height are metres. Extrinsics map book-world points into the OpenCV
camera frame (x right, y down, z forward). The book rests at world z=0.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize_scalar


def plane_corners(size_m):
    """Return top-plane corners in image annotation order, in metres."""
    x, y, z = np.asarray(size_m, dtype=float)
    if min(x, y, z) <= 0:
        raise ValueError("All physical dimensions must be positive")
    return np.array([[-x / 2, y / 2, z], [x / 2, y / 2, z], [x / 2, -y / 2, z], [-x / 2, -y / 2, z]])


def solve_plane(quad, size_m, intrinsics):
    """Return world-to-camera rotation, translation and corner residuals in pixels."""
    corners = plane_corners(size_m)
    ok, rvec, tvec = cv2.solvePnP(corners, np.asarray(quad, dtype=float), intrinsics, None)
    if not ok or tvec[2, 0] <= 0:
        raise ValueError("No positive-depth plane registration")
    image = cv2.projectPoints(corners, rvec, tvec, intrinsics, None)[0].reshape(-1, 2)
    return cv2.Rodrigues(rvec)[0], tvec.ravel(), np.linalg.norm(image - quad, axis=1)


def fit_intrinsics(quads, size_m, image_size, focal=None):
    """Estimate one focal length from multiple plane views; keep centred principal point."""
    width, height = image_size

    def camera(f):
        return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]], dtype=float)

    def objective(log_f):
        k = camera(np.exp(log_f))
        errors = []
        for quad in quads:
            try:
                errors.extend(solve_plane(quad, size_m, k)[2])
            except ValueError:
                return 1e12
        return float(np.mean(np.square(errors)))

    if focal is None:
        fit = minimize_scalar(objective, bounds=(np.log(width * 0.25), np.log(width * 5)), method="bounded")
        focal = float(np.exp(fit.x))
    return camera(focal)


def run(config_path, camera_observations, output):
    """Write measured-dimension camera registrations without replacing old estimates."""
    config = json.loads(Path(config_path).read_text())
    rows = json.loads(Path(camera_observations).read_text())
    video = Path(config["source_video"])
    if hashlib.sha256(video.read_bytes()).hexdigest() != config["source_sha256"]:
        raise ValueError("Video checksum differs from calibration config")
    cap = cv2.VideoCapture(str(video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    selected = np.linspace(0, len(rows) - 1, min(len(rows), 32)).astype(int)
    quads = [np.asarray(rows[i]["book_quad"]) for i in selected]
    k = fit_intrinsics(quads, config["book_size_m"], (width, height), config["camera"]["focal_length_px"])
    rotations = []
    translations = []
    errors = []
    for row in rows:
        r, t, error = solve_plane(np.asarray(row["book_quad"]), config["book_size_m"], k)
        rotations.append(r)
        translations.append(t)
        errors.append(error)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "cameras.npz",
        source_frame=[r["source_frame"] for r in rows],
        seconds=[r["seconds"] for r in rows],
        rotation_world_to_camera=rotations,
        translation_world_to_camera_m=translations,
        intrinsics=k,
    )
    report = {
        "config": config,
        "source_frames": len(rows),
        "image_size_wh": [width, height],
        "intrinsics": k.tolist(),
        "mean_corner_residual_px": float(np.mean(errors)),
        "maximum_corner_residual_px": float(np.max(errors)),
        "scale_source": "User-supplied book dimensions",
        "camera_assumptions": config["camera"],
        "hand_scale": "Hand scale is configured separately; book-plane calibration does not determine it",
        "ground_truth": False,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                x: report[x]
                for x in [
                    "source_frames",
                    "intrinsics",
                    "mean_corner_residual_px",
                    "maximum_corner_residual_px",
                ]
            },
            indent=2,
        )
    )


def main():
    """Register the complete camera timeline using an explicit scene calibration file."""
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--camera-observations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.config, a.camera_observations, a.output)


if __name__ == "__main__":
    main()
