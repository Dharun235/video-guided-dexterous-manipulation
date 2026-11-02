"""Robust RGB-D registration, temporal static-scene filtering and CPU TSDF fusion."""

import xml.etree.ElementTree as ET

import cv2
import numpy as np


def similarity(source, target):
    """Fit target = scale * rotation @ source + translation with proper rotation."""
    source = np.asarray(source, float)
    target = np.asarray(target, float)
    a = source - source.mean(0)
    b = target - target.mean(0)
    u, d, vt = np.linalg.svd(b.T @ a / len(a))
    sign = np.ones(3)
    sign[-1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag(sign) @ vt
    scale = float((d * sign).sum() / np.mean(np.sum(a * a, axis=1)))
    return scale, rotation, target.mean(0) - scale * rotation @ source.mean(0)


def register_similarity(source, target, threshold=0.025, seed=0):
    """Reject moving/depth-inconsistent matches with deterministic similarity RANSAC."""
    if len(source) < 20:
        return None
    rng = np.random.default_rng(seed)
    best = np.zeros(len(source), bool)
    for _ in range(160):
        ids = rng.choice(len(source), 3, replace=False)
        if np.linalg.matrix_rank(source[ids] - source[ids].mean(0), tol=1e-5) < 2:
            continue
        s, r, t = similarity(source[ids], target[ids])
        if not 0.5 < s < 2:
            continue
        errors = np.linalg.norm(s * source @ r.T + t - target, axis=1)
        good = errors < threshold
        if good.sum() > best.sum():
            best = good
    if best.sum() < 20:
        return None
    for _ in range(3):
        s, r, t = similarity(source[best], target[best])
        errors = np.linalg.norm(s * source @ r.T + t - target, axis=1)
        best = errors < threshold
        if best.sum() < 20:
            return None
    if not 0.5 < s < 2 or best.mean() < 0.35:
        return None
    transform = np.eye(4)
    transform[:3, :3] = r
    transform[:3, 3] = t
    return dict(
        scale=s,
        camera_to_world=transform,
        inliers=best,
        median_error_m=float(np.median(errors[best])),
        inlier_fraction=float(best.mean()),
    )


def camera_matrix(intrinsics, shape):
    """Convert normalized MoGe intrinsics to image-pixel intrinsics."""
    return np.asarray(intrinsics, float) * np.array([shape[1], shape[0], 1.0])[:, None]


def static_consensus(views, minimum_votes=3):
    """Reject surfaces contradicted by other unoccluded registered RGB-D observations.

    Occlusion is not free-space evidence. A point behind another surface is kept
    uncertain rather than removed. Colour checks use blurred imagery to reduce
    sensitivity to subpixel reprojection at printed text.
    """
    outputs = []
    for i, view in enumerate(views):
        p = view["points"] * view["scale"]
        valid = view["mask"].astype(bool) & ~view["excluded"] & np.isfinite(p).all(2)
        ys, xs = np.where(valid)
        local = p[ys, xs]
        pose = view["camera_to_world"]
        world = local @ pose[:3, :3].T + pose[:3, 3]
        own = cv2.GaussianBlur(view["image"], (5, 5), 0)[ys, xs].astype(float) / 255
        votes = np.zeros(len(local), int)
        contradictions = np.zeros(len(local), int)
        for j, other in enumerate(views):
            if i == j:
                continue
            t = other["camera_to_world"]
            q = (world - t[:3, 3]) @ t[:3, :3]
            k = camera_matrix(other["intrinsics"], other["depth"].shape)
            z = q[:, 2]
            uv = q @ k.T
            uv = uv[:, :2] / np.maximum(z[:, None], 1e-6)
            h, w = other["depth"].shape
            inside = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
            ids = np.flatnonzero(inside)
            pix = np.rint(uv[ids]).astype(int)
            u, v = pix[:, 0], pix[:, 1]
            good = other["mask"][v, u].astype(bool) & ~other["excluded"][v, u]
            ids = ids[good]
            u = u[good]
            v = v[good]
            depth = other["depth"][v, u] * other["scale"]
            tolerance = np.maximum(0.015, depth * 0.025)
            comparable = z[ids] <= depth + tolerance
            ids = ids[comparable]
            u = u[comparable]
            v = v[comparable]
            depth = depth[comparable]
            tolerance = tolerance[comparable]
            color = cv2.GaussianBlur(other["image"], (5, 5), 0)[v, u].astype(float) / 255
            bad = (z[ids] < depth - tolerance) | (np.mean(abs(color - own[ids]), axis=1) > 0.22)
            votes[ids] += 1
            contradictions[ids] += bad
        rejected = (votes >= minimum_votes) & (contradictions / np.maximum(votes, 1) > 0.6)
        dynamic = np.zeros(valid.shape, bool)
        dynamic[ys[rejected], xs[rejected]] = True
        outputs.append(
            dict(
                static=valid & ~dynamic,
                dynamic=dynamic,
                checked_points=int(np.sum(votes >= minimum_votes)),
                rejected_points=int(rejected.sum()),
            )
        )
    return outputs


def fuse_views(views, filters, voxel_size=0.005):
    """Integrate scaled, masked observations into one Open3D colour TSDF volume."""
    import open3d as o3d

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=voxel_size * 6,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for view, filt in zip(views, filters):
        depth = (view["depth"] * view["scale"]).astype(np.float32).copy()
        depth[~filt["static"]] = 0
        k = camera_matrix(view["intrinsics"], depth.shape)
        h, w = depth.shape
        intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, k[0, 0], k[1, 1], k[0, 2], k[1, 2])
        rgb = cv2.cvtColor(view["image"], cv2.COLOR_BGR2RGB)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsics, np.linalg.inv(view["camera_to_world"]))
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh


def source_texture(vertices, faces, fallback_colors, views, side, tile):
    """Bake only visible, static RGB observations; masked pixels retain fused fallback colour."""
    samples = vertices[faces]
    centres = samples.mean(axis=1)
    corners = np.concatenate([samples, centres[:, None]], axis=1)
    best = np.full(len(faces), -np.inf)
    selected = np.full(len(faces), -1, int)
    for index, view in enumerate(views):
        pose = view["camera_to_scene"]
        q = (corners - pose[:3, 3]) @ pose[:3, :3]
        z = q[:, :, 2]
        uv = q @ view["intrinsics"].T
        uv = uv[:, :, :2] / np.maximum(z[:, :, None], 1e-6)
        h, w = view["static"].shape
        inside = (
            (z > 0.05)
            & (uv[:, :, 0] >= 0)
            & (uv[:, :, 0] < w - 1)
            & (uv[:, :, 1] >= 0)
            & (uv[:, :, 1] < h - 1)
        )
        ids = np.flatnonzero(inside.all(axis=1))
        pix = np.rint(uv[ids]).astype(int)
        good = view["static"][pix[:, :, 1], pix[:, :, 0]].all(axis=1)
        depth = view["depth"][pix[:, 3, 1], pix[:, 3, 0]] * view["scale"]
        good &= abs(z[ids, 3] - depth) < np.maximum(0.02, depth * 0.03)
        a = uv[ids, 1] - uv[ids, 0]
        b = uv[ids, 2] - uv[ids, 0]
        area = abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / (1 + 0.03 * index)
        take = good & (area > best[ids])
        chosen = ids[take]
        best[chosen] = area[take]
        selected[chosen] = index
    blocks = np.empty((side, side, tile, tile, 3), np.uint8)
    blocks[:] = 0
    for index, face in enumerate(faces):
        blocks[index // side, index % side] = np.clip(fallback_colors[face].mean(0) * 255, 0, 255).astype(
            np.uint8
        )
    y, x = np.mgrid[:tile, :tile]
    s = x.ravel() / (tile - 1)
    t = y.ravel() / (tile - 1)
    observed = 0
    for index, view in enumerate(views):
        ids = np.flatnonzero(selected == index)
        for start in range(0, len(ids), 1000):
            group = ids[start : start + 1000]
            tri = samples[group]
            xyz = (
                tri[:, 0, None] * (1 - s - t)[None, :, None]
                + tri[:, 1, None] * s[None, :, None]
                + tri[:, 2, None] * t[None, :, None]
            )
            pose = view["camera_to_scene"]
            q = (xyz - pose[:3, 3]) @ pose[:3, :3]
            uv = q @ view["intrinsics"].T
            uv = uv[:, :, :2] / np.maximum(q[:, :, 2, None], 1e-6)
            mx = uv[:, :, 0].astype(np.float32)
            my = uv[:, :, 1].astype(np.float32)
            color = cv2.remap(cv2.cvtColor(view["image"], cv2.COLOR_BGR2RGB), mx, my, cv2.INTER_LINEAR)
            valid = cv2.remap(view["static"].astype(np.uint8), mx, my, cv2.INTER_NEAREST) > 0
            fallback = blocks[group // side, group % side].reshape(-1, tile * tile, 3)
            color[~valid] = fallback[~valid]
            blocks[group // side, group % side] = color.reshape(-1, tile, tile, 3)
            observed += int(valid.sum())
    atlas = blocks.transpose(0, 2, 1, 3, 4).reshape(side * tile, side * tile, 3)
    return atlas, float(observed / max(1, len(faces) * tile * tile))


def export_environment(mesh, output, source_camera=None, appearance=None):
    """Export fused appearance and finite planar collision patches as one MuJoCo scene.

    The visual mesh has no collisions: MuJoCo convexifies a mesh collider, which
    would fill free space between surfaces. Planar patches are explicit collision
    approximations, not recovered material/hidden geometry.
    """
    import open3d as o3d
    from scipy.spatial.transform import Rotation

    if len(mesh.triangles) > 50000:
        mesh = mesh.simplify_quadric_decimation(50000)
    mesh.compute_vertex_normals()
    output.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output / "environment.ply"), mesh)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    if not len(faces):
        raise ValueError("Fusion produced no surface triangles")
    side = int(np.ceil(np.sqrt(len(faces))))
    tile = 8 if appearance else 4
    atlas = np.zeros((side * tile, side * tile, 3), np.uint8)
    lines = ["v " + " ".join(f"{v:.9g}" for v in p) for p in vertices]
    uv = []
    for i, face in enumerate(faces):
        x = (i % side) * tile
        y = (i // side) * tile
        color = np.clip(colors[face].mean(0) * 255, 0, 255).astype(np.uint8)
        atlas[y : y + tile, x : x + tile] = color
        uv.extend([(x + 0.5, y + 0.5), (x + tile - 0.5, y + 0.5), (x + 0.5, y + tile - 0.5)])
    lines += ["vt " + str(u / len(atlas)) + " " + str(1 - v / len(atlas)) for u, v in uv]
    lines += [
        "f " + " ".join(f"{int(v) + 1}/{i * 3 + j + 1}" for j, v in enumerate(face))
        for i, face in enumerate(faces)
    ]
    (output / "environment.obj").write_text("\n".join(lines) + "\n")
    texture_fraction = 0.0
    if appearance:
        atlas, texture_fraction = source_texture(vertices, faces, colors, appearance, side, tile)
    cv2.imwrite(str(output / "environment.png"), cv2.cvtColor(atlas, cv2.COLOR_RGB2BGR))
    root = ET.Element("mujoco", model="fused_video_environment")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", name="environment", file="environment.obj", inertia="shell")
    ET.SubElement(asset, "texture", name="appearance", type="2d", file="environment.png")
    ET.SubElement(
        asset, "material", name="appearance", texture="appearance", texuniform="false", reflectance="0"
    )
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 2")
    if source_camera is not None:
        pose = source_camera["camera_to_scene"]
        q = Rotation.from_matrix(pose[:3, :3] @ np.diag([1.0, -1.0, -1.0])).as_quat()[[3, 0, 1, 2]]
        fovy = np.rad2deg(
            2 * np.arctan(source_camera["image_shape"][0] / (2 * source_camera["intrinsics"][1, 1]))
        )
        ET.SubElement(
            world,
            "camera",
            name="source",
            pos=" ".join(map(str, pose[:3, 3])),
            quat=" ".join(map(str, q)),
            fovy=str(fovy),
        )
    ET.SubElement(
        world,
        "geom",
        name="environment_visual",
        type="mesh",
        mesh="environment",
        material="appearance",
        contype="0",
        conaffinity="0",
        group="1",
    )
    o3d.utility.random.seed(0)
    cloud = mesh.sample_points_uniformly(min(18000, max(1000, len(vertices))))
    patches = []
    for i in range(4):
        if len(cloud.points) < 150:
            break
        plane, inliers = cloud.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=160)
        if len(inliers) < max(150, 0.08 * len(cloud.points)):
            break
        pts = np.asarray(cloud.points)[inliers]
        normal = np.array(plane[:3])
        normal /= np.linalg.norm(normal)
        centre = np.median(pts, 0)
        x = np.array([1.0, 0, 0])
        if abs(x @ normal) > 0.9:
            x = np.array([0.0, 1, 0])
        x -= normal * (x @ normal)
        x /= np.linalg.norm(x)
        rotation = np.stack([x, np.cross(normal, x), normal], axis=1)
        local = (pts - centre) @ rotation
        lo, hi = np.quantile(local, [0.03, 0.97], axis=0)
        half = (hi - lo) / 2
        if min(half[:2]) > 0.03:
            centre += rotation @ ((lo + hi) / 2)
            q = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
            ET.SubElement(
                world,
                "geom",
                name=f"collision_patch_{i}",
                type="box",
                pos=" ".join(map(str, centre)),
                size=f"{half[0]} {half[1]} .003",
                quat=" ".join(map(str, q)),
                rgba=".5 .5 .5 0",
                group="3",
            )
            patches.append(
                dict(centre_m=centre.tolist(), half_size_m=[*half[:2], 0.003], quaternion_wxyz=q.tolist())
            )
        cloud = cloud.select_by_index(inliers, invert=True)
    ET.ElementTree(root).write(output / "scene.xml")
    return dict(
        vertices=len(vertices),
        triangles=len(faces),
        collision_patches=patches,
        source_rgb_texture_fraction=texture_fraction,
        visual_mesh_collision=False,
        hidden_surfaces_filled=False,
    )
