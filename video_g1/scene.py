"""Build a measured-size scene without overwriting earlier reconstruction evidence."""

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def build(config_path, cameras_path, appearance_path, output, case_masks):
    """Fit a resting-case silhouette and write explicit visual/collision approximations."""
    cfg = json.loads(config_path.read_text())
    cameras = np.load(cameras_path / "cameras.npz")
    k = cameras["intrinsics"]
    r = cameras["rotation_world_to_camera"][0]
    t = cameras["translation_world_to_camera_m"][0]
    size = np.array(cfg["case_size_m"])
    book = np.array(cfg["book_size_m"])
    output.mkdir(parents=True, exist_ok=False)
    assets = output / "assets"
    assets.mkdir()
    mask_path = case_masks
    masks = sorted(mask_path.glob("*.png"))
    mask = cv2.imread(str(masks[0]), cv2.IMREAD_GRAYSCALE)
    ys, xs = np.where(mask > 127)
    target = np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)
    corners = np.array([[x, y, z] for x in [-0.5, 0.5] for y in [-0.5, 0.5] for z in [-0.5, 0.5]]) * size

    def project(q):
        rotation = Rotation.from_euler("z", q[2]).as_matrix()
        points = corners @ rotation.T + [q[0], q[1], book[2] + size[2] / 2]
        uv = cv2.projectPoints(points, cv2.Rodrigues(r)[0], t, k, None)[0][:, 0]
        return np.r_[uv.min(axis=0), uv.max(axis=0)]

    fit = least_squares(
        lambda q: project(q) - target, [0, 0, 0], bounds=([-0.11, -0.11, -np.pi], [0.11, 0.11, np.pi])
    )
    q = fit.x
    root = ET.Element("mujoco", model="book5_measured_geometry")
    ET.SubElement(root, "compiler", angle="radian")
    asset = ET.SubElement(root, "asset")
    # Preserve photographed book texture, correcting the old mesh dimensions.
    shutil.copy2(appearance_path / "assets/book.png", assets / "book.png")
    old = appearance_path / "assets/book.obj"
    lines = []
    for line in old.read_text().splitlines():
        if line.startswith("v "):
            v = np.array(list(map(float, line.split()[1:4]))) * book / np.array([0.198, 0.129, 0.018])
            line = "v " + " ".join(map(str, v))
        lines.append(line)
    (assets / "book.obj").write_text("\n".join(lines) + "\n")
    ET.SubElement(asset, "texture", name="book", type="2d", file=str((assets / "book.png").resolve()))
    ET.SubElement(asset, "material", name="book", texture="book")
    ET.SubElement(asset, "mesh", name="book", file=str((assets / "book.obj").resolve()), inertia="shell")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "geom",
        name="table_collision",
        type="box",
        pos="0 0 -.02",
        size=".4 .19 .02",
        rgba=".65 .58 .46 1",
    )
    ET.SubElement(
        world,
        "geom",
        name="book_collision",
        type="box",
        pos=f"0 0 {book[2] / 2}",
        size=" ".join(map(str, book / 2)),
        rgba=".7 .65 .5 1",
    )
    ET.SubElement(
        world,
        "geom",
        name="book_appearance",
        type="mesh",
        mesh="book",
        material="book",
        contype="0",
        conaffinity="0",
    )
    body = ET.SubElement(
        world,
        "body",
        name="case",
        pos=f"{q[0]} {q[1]} {book[2] + size[2] / 2}",
        quat=" ".join(map(str, Rotation.from_euler("z", q[2]).as_quat()[[3, 0, 1, 2]])),
    )
    ET.SubElement(body, "freejoint", name="case_free")
    ET.SubElement(
        body,
        "geom",
        name="case_solid",
        type="box",
        size=" ".join(map(str, size / 2)),
        mass=str(cfg["physics_assumptions"]["case_mass_kg"]),
        friction=f"{cfg['physics_assumptions']['sliding_friction']} .005 .001",
        rgba=".035 .04 .035 1",
        condim="4",
    )
    ET.indent(root)
    (output / "scene.xml").write_text(ET.tostring(root, encoding="unicode"))
    metadata = {
        "source_sha256": cfg["source_sha256"],
        "camera": {
            "intrinsics": k.tolist(),
            "rotation": Rotation.from_matrix(r).as_rotvec().tolist(),
            "translation": t.tolist(),
            "position": (-r.T @ t).tolist(),
        },
        "config": cfg,
        "initial_case_position_m": [q[0], q[1], book[2] + size[2] / 2],
        "initial_case_yaw_rad": q[2],
        "case_bbox_error_px": float(np.linalg.norm(fit.fun)),
        "approximations": [
            "Rigid box case: no articulated lid",
            "Table footprint assumed 0.80 x 0.38 m",
            "Book fixed to table",
            "Mass and friction unmeasured",
            "Camera estimated from book plane; not ground truth",
        ],
    }
    (output / "reconstruction.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    """Create a new calibrated scene directory with measured dimensions."""
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--cameras", type=Path, required=True)
    p.add_argument("--appearance", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case-masks", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.config, a.cameras, a.appearance, a.output, a.case_masks), indent=2))


if __name__ == "__main__":
    main()
