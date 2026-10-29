"""Build a separate MuJoCo candidate from a TAA case mesh and reviewed book plane.

The first-frame silhouette refit is calibration, not an independent evaluation.
Hidden geometry, book dimensions, mass and friction remain assumptions.
"""
import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from reconstruction.book_space import BOOK_QUAD, BOOK_SIZE, corners


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--taa", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.taa / "sam3d_scaled/case"
    pose = json.loads((source / "pose.json").read_text())
    report = json.loads((source / "depth_scale.json").read_text())
    k = np.asarray(report["intrinsics"])
    mesh = trimesh.load(source / "mesh.glb", force="mesh")
    q = pose["rotation_quat_wxyz"]
    rotation = Rotation.from_quat(q[1:] + q[:1]).as_matrix()
    model_from_glb = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    to_rdf = np.diag([-1, -1, 1])
    transform = to_rdf @ rotation.T @ np.diag(pose["scale"]) @ model_from_glb
    vertices = mesh.vertices @ transform.T + to_rdf @ pose["translation"]
    centre = vertices.mean(axis=0)
    local = vertices - centre
    # Sample the surface deterministically for a low-cost silhouette fit.
    sample = local[np.linspace(0, len(local) - 1, 4000, dtype=int)]
    mask = cv2.imread(str(args.taa / "masks/case/000000.png"), 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    directions = np.c_[np.cos(angles), np.sin(angles)]
    target_support = (contour @ directions.T).max(axis=0)

    def camera_points(parameters, points):
        delta = Rotation.from_rotvec(parameters[:3]).as_matrix()
        translation = centre + np.r_[parameters[3:5], 0.]
        return points @ delta.T * np.exp(parameters[5]) + translation

    def project(points):
        homogeneous = points @ k.T
        return homogeneous[:, :2] / homogeneous[:, 2:]

    def residual(parameters):
        uv = project(camera_points(parameters, sample))
        return (uv @ directions.T).max(axis=0) - target_support

    fit = least_squares(residual, np.zeros(6), max_nfev=160,
                        bounds=([-.5, -.5, -.5, -.2, -.2, -1.5],
                                [.5, .5, .5, .2, .2, 1.5]))
    camera_vertices = camera_points(fit.x, local)
    # DA3 is not metric: calibrate against the same explicitly estimated book size.
    _, rvec, tvec = cv2.solvePnP(corners(BOOK_SIZE[2]), BOOK_QUAD.astype(float), k, None)
    book_mask = cv2.imread(str(args.taa / "masks/book/000000.png"), 0) > 0
    depth = np.load(args.taa / "da3/depth/000000.npy")
    reference_depth = np.median(depth[book_mask & np.isfinite(depth) & (depth > 0)])
    book_camera = corners(BOOK_SIZE[2]) @ cv2.Rodrigues(rvec)[0].T + tvec.ravel()
    metric_scale = float(book_camera[:, 2].mean() / reference_depth)
    world = (camera_vertices * metric_scale - tvec.ravel()) @ cv2.Rodrigues(rvec)[0]
    # Static initial object rests on the book. Record the correction explicitly.
    height_shift = float(BOOK_SIZE[2] - world[:, 2].min())
    world[:, 2] += height_shift
    anchor = world.mean(axis=0)
    shape = world - anchor
    indices = np.linspace(0, len(shape) - 1, 4000, dtype=int)
    camera_rotation_cv = cv2.Rodrigues(rvec)[0]

    def supported_world(parameters, points):
        delta = Rotation.from_rotvec(parameters[:3]).as_matrix()
        rotated = shape @ delta.T * np.exp(parameters[5])
        position = anchor + np.r_[parameters[3:5], 0.]
        position[2] = BOOK_SIZE[2] - rotated[:, 2].min()
        return points @ delta.T * np.exp(parameters[5]) + position

    def supported_residual(parameters):
        points = supported_world(parameters, shape[indices])
        uv = project(points @ camera_rotation_cv.T + tvec.ravel())
        return (uv @ directions.T).max(axis=0) - target_support

    supported_fit = least_squares(supported_residual, np.zeros(6), max_nfev=120,
                                  bounds=([-.4, -.4, -.4, -.05, -.05, -.7],
                                          [.4, .4, .4, .05, .05, .7]))
    world = supported_world(supported_fit.x, shape)
    centre_world = world.mean(axis=0)
    mesh.vertices = world - centre_world
    shutil.copytree(args.scene / "assets", args.output / "assets")
    visual_mesh = mesh.simplify_quadric_decimation(face_count=80000)
    visual_mesh.export(args.output / "assets/taa_case.stl")
    hull = mesh.convex_hull
    hull.export(args.output / "assets/taa_case_hull.stl")
    tree = ET.parse(args.scene / "scene.xml")
    root = tree.getroot()
    camera = root.find(".//camera[@name='source']")
    camera_rotation = cv2.Rodrigues(rvec)[0].T @ np.diag([1., -1., -1.])
    camera_quat = Rotation.from_matrix(camera_rotation).as_quat()[[3, 0, 1, 2]]
    camera.set("pos", " ".join(map(str, (-cv2.Rodrigues(rvec)[0].T @ tvec).ravel())))
    camera.set("quat", " ".join(map(str, camera_quat)))
    camera.set("fovy", str(np.degrees(2 * np.arctan(232 / k[1, 1]))))
    assets = root.find("asset")
    ET.SubElement(assets, "mesh", name="taa_case", file="taa_case.stl")
    ET.SubElement(assets, "mesh", name="taa_case_hull", file="taa_case_hull.stl")
    body = root.find(".//body[@name='case']")
    body.set("pos", " ".join(map(str, centre_world)))
    body.attrib.pop("quat", None)
    for geom in list(body.findall("geom")):
        body.remove(geom)
    ET.SubElement(body, "geom", name="case_solid", type="mesh", mesh="taa_case_hull",
                  mass=".055", rgba=".04 .04 .04 0", group="3")
    ET.SubElement(body, "geom", name="taa_case_appearance", type="mesh", mesh="taa_case",
                  mass="0", contype="0", conaffinity="0", rgba=".035 .04 .035 1")
    tree.write(args.output / "scene.xml")
    uv = project(world @ camera_rotation_cv.T + tvec.ravel())
    overlay = cv2.imread(str(args.taa / "frames/000000.jpg"))
    cv2.polylines(overlay, [cv2.convexHull(uv.astype(np.float32)).astype(int)],
                  True, (0, 0, 255), 2)
    cv2.imwrite(str(args.output / "calibration-overlay.jpg"), overlay)
    result = dict(source="TAA SAM3D + MoGe; DA3 scale; first-frame silhouette calibration",
                  mesh_vertices=len(mesh.vertices), mesh_faces=len(mesh.faces),
                  simulation_visual_faces=len(visual_mesh.faces),
                  watertight=bool(mesh.is_watertight),
                  initial_support_rmse_px=float(np.sqrt(np.mean(residual(np.zeros(6)) ** 2))),
                  fitted_support_rmse_px=float(np.sqrt(np.mean(residual(fit.x) ** 2))),
                  supported_fit_rmse_px=float(np.sqrt(np.mean(supported_residual(supported_fit.x) ** 2))),
                  supported_fit_parameters=supported_fit.x.tolist(),
                  fit_parameters=fit.x.tolist(), metric_scale=metric_scale,
                  book_support_height_correction_m=height_shift,
                  case_dimensions_world_m=mesh.extents.tolist(),
                  limitations=["First-frame fit is not independent validation",
                               "Hidden shape is generated; dimensions use estimated book size",
                               "Collision is a convex hull; mass and friction are assumed",
                               "No grasp or full-video object pose tracking is claimed"])
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
