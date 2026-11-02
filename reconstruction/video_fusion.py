"""Register a full video, exclude people/motion and fuse one static MuJoCo environment."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from .fusion_geometry import (
    camera_matrix,
    export_environment,
    fuse_views,
    register_similarity,
    static_consensus,
)
from .scene_check import dump
from .scene_geometry import fit_support, hand_region
from .video_checks import assess, inspect_hands, inspect_video


def excluded_people(image, hands, people):
    """Conservatively exclude complete hand hulls, forearms/person masks and a safety border."""
    mask = people(image) | hand_region(hands, image.shape[:2])
    h, w = image.shape[:2]
    for hand in hands:
        pts = (np.asarray(hand["landmarks"])[:, :2] * [w, h]).astype(np.int32)
        cv2.fillConvexPoly(mask.view(np.uint8), cv2.convexHull(pts), 1)
    for hand in hands:
        pts = np.asarray(hand["landmarks"])[:, :2] * [w, h]
        wrist = pts[0]
        direction = wrist - pts[9]
        length = np.linalg.norm(direction)
        if length > 2:
            end = wrist + direction / length * max(h, w) * 2
            width = max(12, round(np.linalg.norm(pts[5] - pts[17]) * 1.1))
            cv2.line(mask.view(np.uint8), tuple(wrist.astype(int)), tuple(end.astype(int)), 1, width)
    return cv2.dilate(mask.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0


def features(image, excluded, sift):
    """Detect repeatable static-scene features outside excluded image regions."""
    return sift.detectAndCompute(image, (~excluded).astype(np.uint8) * 255)


def correspondences(current, anchor):
    """Match current image features to accepted world-space anchor observations."""
    kp, des = current
    if des is None or anchor["descriptors"] is None:
        return None
    pairs = cv2.BFMatcher().knnMatch(des, anchor["descriptors"], k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.7 * n.distance]
    if len(good) < 20:
        return None
    image = np.array([kp[m.queryIdx].pt for m in good])
    ref = np.rint([anchor["keypoints"][m.trainIdx].pt for m in good]).astype(int)
    world = anchor["points"][ref[:, 1], ref[:, 0]] * anchor["scale"]
    pose = anchor["camera_to_world"]
    world = world @ pose[:3, :3].T + pose[:3, 3]
    valid = np.isfinite(world).all(1) & anchor["mask"][ref[:, 1], ref[:, 0]].astype(bool)
    return image[valid], world[valid]


def solve_camera(pixels, world, k):
    """Fit a pose to image evidence and expose inlier landmarks for flow fallback."""
    if len(world) < 20:
        return None
    ok, r, t, inliers = cv2.solvePnPRansac(
        world.astype(float),
        pixels.astype(float),
        k,
        None,
        reprojectionError=2.5,
        iterationsCount=160,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 20 or len(inliers) / len(world) < 0.35:
        return None
    ids = inliers.ravel()
    r, t = cv2.solvePnPRefineLM(world[ids], pixels[ids], k, None, r, t)
    camera_points = world[ids] @ cv2.Rodrigues(r)[0].T + t.ravel()
    if np.mean(camera_points[:, 2] > 0.05) < 0.95:
        return None
    projected = cv2.projectPoints(world[ids], r, t, k, None)[0].reshape(-1, 2)
    error = float(np.median(np.linalg.norm(projected - pixels[ids], axis=1)))
    if error > 2:
        return None
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(r)[0]
    transform[:3, 3] = t.ravel()
    return dict(
        camera_to_world=np.linalg.inv(transform),
        inliers=len(ids),
        median_reprojection_px=error,
        _world=world[ids],
        _pixels=pixels[ids],
    )


def track_camera(current, anchors, k):
    """Estimate camera pose from masked static descriptors; preserve rejected fits."""
    for anchor in anchors:
        pair = correspondences(current, anchor)
        if pair is None:
            continue
        result = solve_camera(*pair, k)
        if result is not None:
            return dict(**result, anchor_frame=anchor["frame"], method="sift_pnp")
    return None


def track_flow_camera(previous, image, excluded, k):
    """Recover a weak descriptor frame using bidirectionally checked static optical flow."""
    if previous is None or len(previous["pixels"]) < 20:
        return None
    pixels = previous["pixels"].astype(np.float32).reshape(-1, 1, 2)
    old = cv2.cvtColor(previous["image"], cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    nxt, status, error = cv2.calcOpticalFlowPyrLK(old, gray, pixels, None, winSize=(21, 21), maxLevel=3)
    if nxt is None:
        return None
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray, old, nxt, None, winSize=(21, 21), maxLevel=3)
    if back is None:
        return None
    pts = nxt.reshape(-1, 2)
    h, w = gray.shape
    uv = np.rint(pts).astype(int)
    valid = (status.ravel() > 0) & (reverse.ravel() > 0) & (error.ravel() < 20)
    valid &= np.linalg.norm(back.reshape(-1, 2) - pixels.reshape(-1, 2), axis=1) < 0.75
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    ids = np.flatnonzero(valid)
    ids = ids[~excluded[uv[ids, 1], uv[ids, 0]]]
    result = solve_camera(pts[ids].astype(float), previous["world"][ids], k)
    return dict(**result, method="optical_flow_pnp") if result is not None else None


def run(video, output, models, depth_stride=30, voxel_size=0.005, depth_cache=None):
    """Save every source frame's masks/pose evidence and integrate accepted depth keyframes."""
    from .scene_models import Hands, MetricGeometry, People

    cv2.setNumThreads(1)
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    (output / "masks").mkdir()
    (output / "keyframes").mkdir()
    checksum = hashlib.sha256(video.read_bytes()).hexdigest()
    if depth_cache is not None:
        manifest = json.loads((depth_cache / "report.json").read_text())
        if manifest.get("input_sha256") != checksum:
            raise ValueError("Depth-cache source video differs")
    computed_depth = 0
    cached_depth = 0
    started = time.monotonic()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or fps <= 0:
        cap.release()
        report = dict(
            stage="static_video_fusion",
            status="blocked",
            reason="invalid_video",
            message="Video cannot be decoded with valid timing",
        )
        dump(output / "report.json", report)
        return report
    detector = Hands(models)
    people = None
    sift = cv2.SIFT_create(nfeatures=2400)
    flow_state = None
    anchors = []
    views = []
    timeline = []
    frame = 0
    fov = None
    k = None
    try:
        facts, checks = inspect_video(video)
        if "error" not in facts:
            facts, checks, _ = inspect_hands(video, facts, checks, detector)
        gate = assess(facts, checks)
        dump(output / "input-check.json", dict(video=facts, **gate))
        if gate["status"] != "pass":
            report = dict(stage="static_video_fusion", status=gate["status"], reasons=gate["reasons"])
            dump(output / "report.json", report)
            return report
        people = People(models)
        metric = MetricGeometry(models)
        while True:
            ok, image = cap.read()
            if not ok:
                break
            image = cv2.resize(image, (512, round(image.shape[0] * 512 / image.shape[1])))
            hands = detector(image)
            excluded = excluded_people(image, hands, people)
            cv2.imwrite(str(output / "masks" / f"{frame:06d}.png"), excluded.astype(np.uint8) * 255)
            cv2.imwrite(str(output / "frames" / f"{frame:06d}.jpg"), image)
            current = features(image, excluded, sift)
            row = dict(
                frame=frame,
                time_s=frame / fps,
                decoded_timestamp_s=float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000,
                hands=len(hands),
                hand_observations=hands,
                excluded_fraction=float(excluded.mean()),
                camera_valid=False,
                depth_inferred=False,
            )
            selected = frame % depth_stride == 0 or frame == declared - 1
            ordered = [anchors[0], anchors[-1]] if len(anchors) > 1 else anchors
            if selected:
                cache_path = (
                    depth_cache / "keyframes" / f"{frame:06d}.npz" if depth_cache is not None else None
                )
                if cache_path is not None and cache_path.exists():
                    previous = np.load(cache_path)
                    if not np.array_equal(previous["image"], image):
                        raise ValueError("Depth-cache frame differs")
                    geometry = {
                        name: previous[name]
                        for name in ["points", "depth", "mask", "intrinsics", "normal"]
                        if name in previous
                    }
                    cached_depth += 1
                    row["depth_origin"] = "cached_prediction"
                else:
                    geometry = metric(image, fov_x=fov)
                    computed_depth += 1
                    row["depth_origin"] = "fresh_prediction"
                row["depth_inferred"] = True
                if k is None:
                    k = camera_matrix(geometry["intrinsics"], image.shape[:2])
                    fov = float(np.rad2deg(2 * np.arctan(0.5 / geometry["intrinsics"][0, 0])))
                    registered = dict(
                        scale=1.0, camera_to_world=np.eye(4), median_error_m=0.0, inlier_fraction=1.0
                    )
                else:
                    registered = None
                    for anchor in ordered:
                        pair = correspondences(current, anchor)
                        if pair is None:
                            continue
                        pixels, world = pair
                        pix = np.rint(pixels).astype(int)
                        raw = geometry["points"][pix[:, 1], pix[:, 0]]
                        valid = np.isfinite(raw).all(1) & geometry["mask"][pix[:, 1], pix[:, 0]].astype(bool)
                        registered = register_similarity(raw[valid], world[valid])
                        if registered is not None:
                            break
                if registered is not None:
                    view = dict(
                        frame=frame,
                        image=image,
                        excluded=excluded,
                        **geometry,
                        scale=registered["scale"],
                        camera_to_world=registered["camera_to_world"],
                    )
                    views.append(view)
                    anchor = dict(**view, keypoints=current[0], descriptors=current[1])
                    anchors.append(anchor)
                    pixels = np.array([kp.pt for kp in current[0]])
                    uv = np.rint(pixels).astype(int)
                    xyz = view["points"][uv[:, 1], uv[:, 0]] * view["scale"]
                    good = np.isfinite(xyz).all(1)
                    pose = view["camera_to_world"]
                    flow_state = dict(
                        image=image, pixels=pixels[good], world=xyz[good] @ pose[:3, :3].T + pose[:3, 3]
                    )
                    np.savez_compressed(output / "keyframes" / f"{frame:06d}.npz", **view)
                    row.update(
                        camera_valid=True,
                        camera_to_world=view["camera_to_world"].tolist(),
                        depth_scale=view["scale"],
                        alignment_error_m=registered["median_error_m"],
                        inlier_fraction=registered["inlier_fraction"],
                    )
                else:
                    row["depth_rejection_reason"] = "depth_registration_failed"
            if not row["camera_valid"]:
                pose = track_camera(current, ordered, k) if ordered else None
                if pose is None:
                    pose = track_flow_camera(flow_state, image, excluded, k)
                if pose is not None:
                    flow_state = dict(image=image, pixels=pose["_pixels"], world=pose["_world"])
                    row.update(
                        camera_valid=True,
                        **{
                            key: value.tolist() if isinstance(value, np.ndarray) else value
                            for key, value in pose.items()
                            if not key.startswith("_")
                        },
                    )
                else:
                    row["reason"] = "camera_tracking_failed"
            timeline.append(row)
            if frame % 60 == 0:
                print(f"Frame {frame}/{declared}: {len(views)} accepted depth views", flush=True)
                dump(
                    output / "progress.json",
                    dict(
                        processed_frames=frame + 1,
                        accepted_depth_views=len(views),
                        elapsed_s=time.monotonic() - started,
                    ),
                )
            frame += 1
    finally:
        cap.release()
        detector.close()
        if people is not None:
            people.close()
    dump(output / "timeline.json", timeline)
    if len(views) < 3:
        report = dict(
            stage="static_video_fusion",
            status="blocked",
            source_frames=frame,
            registered_frames=sum(r["camera_valid"] for r in timeline),
            depth_views=len(views),
            elapsed_s=time.monotonic() - started,
            input_sha256=checksum,
            reason="insufficient_registered_depth_views",
            message="Independent learned depth could not be aligned reliably; no environment was exported.",
        )
        dump(output / "report.json", report)
        print(json.dumps(report, indent=2))
        return report
    filters = static_consensus(views)
    for view, filt in zip(views, filters):
        cv2.imwrite(
            str(output / "masks" / f"dynamic-{view['frame']:06d}.png"), filt["dynamic"].astype(np.uint8) * 255
        )
    mesh = fuse_views(views, filters, voxel_size)
    first = views[0]
    plane = fit_support(first["points"], first["mask"].astype(bool), first["excluded"])
    scene_from_world = np.eye(4)
    scene_from_world[:3, :3] = plane["rotation"]
    scene_from_world[:3, 3] = -plane["rotation"] @ plane["origin"]
    mesh.transform(scene_from_world)
    appearance = [
        dict(
            image=v["image"],
            static=f["static"],
            depth=v["depth"],
            scale=v["scale"],
            intrinsics=camera_matrix(v["intrinsics"], v["depth"].shape),
            camera_to_scene=scene_from_world @ v["camera_to_world"],
        )
        for v, f in zip(views, filters)
    ]
    result = export_environment(
        mesh,
        output / "environment",
        appearance=appearance,
        source_camera=dict(
            camera_to_scene=scene_from_world @ first["camera_to_world"],
            intrinsics=k,
            image_shape=first["image"].shape[:2],
        ),
    )
    poses = np.full((len(timeline), 4, 4), np.nan)
    for row in timeline:
        if row["camera_valid"]:
            poses[row["frame"]] = scene_from_world @ np.array(row["camera_to_world"])
    np.savez_compressed(
        output / "cameras.npz",
        source_frame=np.arange(len(timeline)),
        source_time_s=np.arange(len(timeline)) / fps,
        camera_to_scene=poses,
        valid=np.isfinite(poses).all(axis=(1, 2)),
        intrinsics=k,
    )
    report = dict(
        implementation_sha256={
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ["video_fusion.py", "fusion_geometry.py", "scene_models.py", "video_checks.py"]
        },
        runtime={
            name: importlib.metadata.version(name)
            for name in ["open3d", "mujoco", "torch", "numpy", "opencv-contrib-python"]
        },
        platform=platform.platform(),
        device="cpu",
        camera_fov_x_deg=fov,
        integration_depth_limit_m=3.0,
        model_manifest_sha256=hashlib.sha256(
            (Path(__file__).parent / "SCENE_MODELS.json").read_bytes()
        ).hexdigest(),
        stage="static_video_fusion",
        status="review",
        input_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        source_frames=frame,
        declared_frames=declared,
        registered_frames=int(np.isfinite(poses).all(axis=(1, 2)).sum()),
        depth_stride=depth_stride,
        depth_views=len(views),
        computed_depth_views=computed_depth,
        cached_depth_views=cached_depth,
        voxel_size_m=voxel_size,
        elapsed_s=time.monotonic() - started,
        all_frames_masked=True,
        failed_pose_frames=[r["frame"] for r in timeline if not r["camera_valid"]],
        median_keyframe_alignment_m=float(
            np.median([r["alignment_error_m"] for r in timeline if "alignment_error_m" in r])
        ),
        temporal_checked_points=sum(f["checked_points"] for f in filters),
        temporal_rejected_points=sum(f["rejected_points"] for f in filters),
        excluded_pixel_observations=sum(int(v["excluded"].sum()) for v in views),
        scale="learned metric estimate; depth keyframes aligned with bounded similarity transforms",
        hand_landmark_coordinates="Image-normalized x/y and model-relative z; masking evidence, not calibrated 3D robot commands",
        moving_objects="Excluded inconsistent surfaces retained as per-keyframe masks; identity and object motion are not static-map geometry",
        camera_to_scene_convention="column-vector transform from OpenCV camera coordinates to z-up scene coordinates",
        reasons=[
            "Learned depth and scale need external validation",
            "Unseen and human-occluded surfaces are not filled",
            "Moving-object identity and geometry need a separate motion stage",
            "Collision patches may overlap or omit small surfaces and need contact review",
        ],
        **result,
    )
    dump(output / "report.json", report)
    print(json.dumps(report, indent=2))
    return report


def render(output, step=2):
    """Compare the source, excluded pixels and the fused static MuJoCo scene in source time."""
    import subprocess
    import xml.etree.ElementTree as ET

    import mujoco
    from scipy.spatial.transform import Rotation

    cameras = np.load(output / "cameras.npz")
    timeline = json.loads((output / "timeline.json").read_text())
    image = cv2.imread(str(output / "frames/000000.jpg"))
    h, w = image.shape[:2]
    k = cameras["intrinsics"]
    root = ET.parse(output / "environment/scene.xml")
    if root.find("worldbody/camera[@name='source']") is None:
        ET.SubElement(
            root.find("worldbody"),
            "camera",
            name="source",
            fovy=str(np.rad2deg(2 * np.arctan(h / (2 * k[1, 1])))),
        )

    model = mujoco.MjModel.from_xml_string(
        ET.tostring(root.getroot(), encoding="unicode"),
        assets={
            p.name: p.read_bytes() for p in (output / "environment").iterdir() if p.suffix in [".obj", ".png"]
        },
    )
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
    model.vis.global_.offheight = max(model.vis.global_.offheight, h)
    data = mujoco.MjData(model)
    preview = output / "preview"
    preview.mkdir(exist_ok=True)
    frames = preview / "frames"
    frames.mkdir(exist_ok=True)
    renderer = mujoco.Renderer(model, height=h, width=w)
    poses = cameras["camera_to_scene"]
    valid = cameras["valid"]
    fps = 1 / np.diff(cameras["source_time_s"]).mean()
    header = 40
    if (h + header) % 2:
        header += 1
    try:
        for counter, i in enumerate(range(0, len(timeline), step)):
            im = cv2.imread(str(output / "frames" / f"{i:06d}.jpg"))
            mask = cv2.imread(str(output / "masks" / f"{i:06d}.png"), 0) > 0
            removed = im.copy()
            removed[mask] = (removed[mask] * 0.25 + np.array([20, 30, 220]) * 0.75).astype(np.uint8)
            if valid[i]:
                pose = poses[i]
                rotation = pose[:3, :3] @ np.diag([1.0, -1.0, -1.0])
                model.cam_pos[0] = pose[:3, 3]
                model.cam_quat[0] = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera="source")
                scene = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            else:
                scene = np.zeros_like(im)
            canvas = np.zeros((h + header, w * 3, 3), np.uint8)
            canvas[header:, :w] = im
            canvas[header:, w : 2 * w] = removed
            canvas[header:, 2 * w :] = scene
            for panel, label in enumerate(
                ["Recorded video", "Excluded human pixels", "Fused static MuJoCo scene (review)"]
            ):
                cv2.putText(canvas, label, (panel * w + 8, 20), 0, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"frame {i} | {i / fps:.2f}s | depth on selected keyframes",
                (8, 35),
                0,
                0.35,
                (170, 200, 170),
                1,
            )
            cv2.imwrite(str(frames / f"{counter:06d}.jpg"), canvas)
            if i in [0, step * (len(timeline) // (2 * step)), step * ((len(timeline) - 1) // step)]:
                cv2.imwrite(str(preview / f"frame-{i:06d}.jpg"), canvas)
    finally:
        renderer.close()
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-framerate",
            str(fps / step),
            "-i",
            str(frames / "%06d.jpg"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(preview / "comparison.mp4"),
        ],
        check=True,
    )
    import shutil

    shutil.rmtree(frames)
    return preview / "comparison.mp4"


def main():
    """Fuse an entire accepted clip; --depth-stride 1 requests neural depth on every frame."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path(".cache/scene-models"))
    parser.add_argument("--depth-stride", type=int, default=30)
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--depth-cache", type=Path, help="Reuse matching raw predictions; re-estimate masks and registration"
    )
    args = parser.parse_args()
    if args.depth_stride < 1 or args.voxel_size <= 0:
        parser.error("Depth stride and voxel size must be positive")
    report = run(args.video, args.output, args.models, args.depth_stride, args.voxel_size, args.depth_cache)
    if args.render and "triangles" in report:
        render(args.output)
    raise SystemExit({"pass": 0, "review": 3, "blocked": 2}[report["status"]])


if __name__ == "__main__":
    main()
