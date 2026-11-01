"""Build conservative visible-surface scene proposals from learned metric point maps."""

import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.spatial import ConvexHull


def hand_region(hands, shape):
    """Rasterize and dilate landmark hulls to exclude the visible hand from objects."""
    h, w = shape
    mask = np.zeros(shape, np.uint8)
    for hand in hands:
        p = (np.array(hand["landmarks"])[:, :2] * [w, h]).astype(np.int32)
        cv2.fillConvexPoly(mask, cv2.convexHull(p[[0, 5, 9, 13, 17]]), 255)
        for chain in [
            [0, 1, 2, 3, 4],
            [0, 5, 6, 7, 8],
            [5, 9, 10, 11, 12],
            [9, 13, 14, 15, 16],
            [13, 17, 18, 19, 20],
        ]:
            cv2.polylines(mask, [p[chain]], False, 255, max(3, round(w * 0.008)))
    return mask > 0


def proposal_points(image, hands):
    """Place class-independent SAM prompts across the image and around fingertips."""
    h, w = image.shape[:2]
    points = [
        [float(x * w), float(y * h)] for y in np.linspace(0.15, 0.85, 5) for x in np.linspace(0.1, 0.9, 7)
    ]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(cv2.Canny(gray, 60, 140), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    seeds = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        x, y, bw, bh = cv2.boundingRect(contour)
        if 6 < bw < 0.45 * w and 6 < bh < 0.45 * h and cv2.contourArea(contour) > 25:
            centre = np.array([x + bw / 2, y + bh / 2])
            if not any(np.linalg.norm(centre - p) < 8 for p in seeds):
                seeds.append(centre)
        if len(seeds) >= 30:
            break
    points.extend([p.tolist() for p in seeds])
    for hand in hands:
        tips = np.array(hand["landmarks"])[[4, 8, 12, 16, 20], :2] * [w, h]
        centre = tips.mean(axis=0)
        points.extend([p.tolist() for p in tips])
        points.extend(
            [
                (centre + [dx, dy]).clip([0, 0], [w - 1, h - 1]).tolist()
                for dx, dy in [(0, 0), (-20, 0), (20, 0), (0, -20), (0, 20)]
            ]
        )
    return points


def rank_objects(candidates, hands, image, person_mask=None):
    """Rank non-hand masks near fingertips; report candidates rather than asserting contact."""
    h, w = image.shape[:2]
    handmask = hand_region(hands, (h, w))
    ranked = []
    tips = (
        np.concatenate([np.array(hd["landmarks"])[[4, 8, 12, 16, 20], :2] * [w, h] for hd in hands])
        if hands
        else None
    )
    for c in candidates:
        mask = c["mask"]
        area = int(mask.sum())
        overlap = float((mask & handmask).sum() / max(1, area))
        if person_mask is not None and (mask & person_mask).sum() / max(1, area) > 0.35:
            continue
        if overlap > 0.35:
            continue
        ys, xs = np.where(mask)
        if xs.max() - xs.min() > 0.85 * w or ys.max() - ys.min() > 0.85 * h:
            continue
        if sum([mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any()]) >= 2:
            continue
        if tips is not None:
            distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
            t = np.rint(tips).astype(int).clip([0, 0], [w - 1, h - 1])
            proximity = float(np.min(distance[t[:, 1], t[:, 0]]) / np.hypot(h, w))
        else:
            proximity = 1.0
        if tips is not None and proximity > 0.04:
            continue
        rank = c["score"] - 3 * proximity - 0.2 * overlap
        ranked.append(
            dict(
                **c,
                rank=float(rank),
                hand_overlap=overlap,
                tip_distance_fraction=proximity,
                bounds_px=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            )
        )
    return sorted(ranked, key=lambda c: -c["rank"])


def fit_support(points, valid, excluded, seed=0):
    """Fit a dominant visible support-plane hypothesis, retaining its residual and coverage."""
    keep = valid & ~excluded & np.isfinite(points).all(axis=2)
    keep[: int(len(keep) * 0.25)] = False
    cloud = points[keep]
    if len(cloud) < 100:
        raise ValueError("insufficient_scene_geometry")
    rng = np.random.default_rng(seed)
    cloud = cloud[rng.choice(len(cloud), min(4000, len(cloud)), replace=False)]
    threshold = max(0.005, float(np.median(cloud[:, 2])) * 0.01)
    best = np.zeros(len(cloud), bool)
    for _ in range(160):
        p = cloud[rng.choice(len(cloud), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        inliers = abs(np.einsum("ij,j->i", cloud - p[0], n)) < threshold
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 100:
        raise ValueError("support_plane_not_found")
    origin = np.median(cloud[best], axis=0)
    _, _, vt = np.linalg.svd(cloud[best] - origin, full_matrices=False)
    normal = vt[-1]
    if normal @ origin > 0:
        normal = -normal
    x = np.array([1.0, 0, 0])
    x -= normal * (x @ normal)
    x /= np.linalg.norm(x)
    rotation = np.stack([x, np.cross(normal, x), normal])
    residual = float(np.median(abs((cloud[best] - origin) @ normal)))
    return dict(
        origin=origin,
        rotation=rotation,
        residual_m=residual,
        inlier_fraction=float(best.mean()),
        threshold_m=threshold,
    )


def visible_extent(points, mask, plane):
    """Estimate visible extents in the support frame, without filling hidden surfaces."""
    cloud = points[mask & np.isfinite(points).all(axis=2)]
    if len(cloud) < 30:
        raise ValueError("insufficient_object_geometry")
    cloud = (cloud - plane["origin"]) @ plane["rotation"].T
    low, high = np.quantile(cloud, [0.02, 0.98], axis=0)
    return cloud, low, high


def export_scene(output, image, geometry, object_mask, hands):
    """Export point cloud, visible object hull and a loadable MuJoCo scene proposal."""
    output.mkdir(parents=True, exist_ok=True)
    points = geometry["points"]
    valid = geometry["mask"].astype(bool) & np.isfinite(points).all(axis=2)
    excluded = (
        object_mask
        | hand_region(hands, valid.shape)
        | geometry.get("person_mask", np.zeros(valid.shape, bool))
    )
    plane = fit_support(points, valid, excluded)
    cloud, low, high = visible_extent(points, object_mask & valid, plane)
    extents = high - low
    if np.min(extents) < 0.001:
        raise ValueError("degenerate_object_geometry")
    centre = (low + high) / 2
    hullcloud = cloud[:: max(1, len(cloud) // 1800)] - centre
    hull = ConvexHull(hullcloud)
    lines = ["v " + " ".join(map(str, p)) for p in hullcloud]
    lines += ["f " + " ".join(str(int(i) + 1) for i in face) for face in hull.simplices]
    (output / "object.obj").write_text("\n".join(lines) + "\n")
    world = (points - plane["origin"]) @ plane["rotation"].T
    keep = (
        valid & ~hand_region(hands, valid.shape) & ~geometry.get("person_mask", np.zeros(valid.shape, bool))
    )
    xyz = world[keep][::4]
    colors = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[keep][::4]
    with (output / "scene.ply").open("w") as f:
        f.write(
            f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        for p, c in zip(xyz, colors):
            f.write(" ".join(map(str, [*p, *c])) + "\n")
    support = world[valid & ~excluded]
    bounds = np.quantile(support, [0.02, 0.98], axis=0)
    half = np.maximum((bounds[1, :2] - bounds[0, :2]) / 2, 0.05)
    pos = (bounds[1, :2] + bounds[0, :2]) / 2
    root = ET.Element("mujoco", model="video_scene_proposal")
    ET.SubElement(root, "option", timestep=".002")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", name="object", file="object.obj")
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", pos="0 0 2")
    ET.SubElement(
        worldbody,
        "geom",
        name="support",
        type="box",
        size=f"{half[0]} {half[1]} .01",
        pos=f"{pos[0]} {pos[1]} -.01",
        rgba=".6 .55 .45 1",
    )
    # Place the estimated hull on its support at reset; retain the uncorrected pose in the report.
    reset = centre.copy()
    reset[2] -= low[2]
    body = ET.SubElement(worldbody, "body", name="object", pos=" ".join(map(str, reset)))
    ET.SubElement(body, "freejoint", name="object_free")
    ET.SubElement(
        body, "geom", type="mesh", mesh="object", mass=".055", friction=".7 .005 .001", rgba=".2 .25 .3 1"
    )
    ET.ElementTree(root).write(output / "scene.xml")
    np.savez_compressed(
        output / "geometry.npz",
        **geometry,
        support_origin_camera_m=plane["origin"],
        rotation_camera_to_scene=plane["rotation"],
        object_mask=object_mask,
    )
    return dict(
        scene_check=verify_scene(output / "scene.xml"),
        visible_extent_m=extents.tolist(),
        observed_bounds_m=[low.tolist(), high.tolist()],
        support_residual_m=plane["residual_m"],
        support_inlier_fraction=plane["inlier_fraction"],
        reset_support_correction_m=float(-low[2]),
        dimensions_source="MoGe-2 metric estimate of visible surfaces; hidden shape unresolved",
        geometry_kind="single-view visible point cloud and convex object hull",
        physical_assumptions=dict(mass_kg=0.055, sliding_friction=0.7, support_thickness_m=0.02),
        scene_frame="right handed; support z=0, +z toward camera, x projected camera-right",
    )


def verify_scene(path):
    """Compile a free-object scene and check finite passive settling under assumed physics."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(path.resolve()))
    data = mujoco.MjData(model)
    for _ in range(round(2 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    penetration = max([max(0.0, -float(c.dist)) for c in data.contact], default=0.0)
    return dict(
        stable_under_assumed_physics=bool(
            np.isfinite(data.qpos).all()
            and np.linalg.norm(data.qvel[:3]) < 0.05
            and np.linalg.norm(data.qvel[3:]) < 1
            and penetration < 0.003
            and data.qpos[2] > -0.03
        ),
        settle_thresholds=dict(linear_speed_m_s=0.05, angular_speed_rad_s=1.0, penetration_m=0.003),
        loads_in_mujoco=True,
        free_object=True,
        finite_state=bool(np.isfinite(data.qpos).all()),
        settle_linear_speed_m_s=float(np.linalg.norm(data.qvel[:3])),
        settle_angular_speed_rad_s=float(np.linalg.norm(data.qvel[3:])),
        maximum_final_penetration_m=penetration,
        assumed_physics=True,
    )
