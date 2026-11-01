"""Extract every source frame into an auditable right-hand track on Apple Silicon."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


def hand_pass(video, output, mano, samples=20, resume=False, cache=None, right_hand=False):
    """Estimate MANO hands on selected frames and save source-aligned observations."""
    if resume and (output / "manifest.json").exists():
        raise FileExistsError("Completed hand pass already exists")
    output.mkdir(parents=True, exist_ok=resume)
    os.environ["XDG_CACHE_HOME"] = str((cache or output / "cache").resolve())
    import fasthamer

    started = time.monotonic()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.arange(count) if samples == 0 else np.unique(np.linspace(0, count - 1, samples).astype(int))
    (output / "frames").mkdir(exist_ok=resume)
    (output / "overlays").mkdir(exist_ok=resume)
    print("Loading Apple Neural Engine hand reconstruction", flush=True)
    model = fasthamer.load(
        mode="image",
        max_hands=1,
        hand_model="mano",
        mano_path=str(mano),
        compute_units="CPU_AND_NE",
        force_handedness="right" if right_hand else None,
    )
    rows, thumbs = [], []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot decode frame {i}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t0 = time.monotonic()
        result = model(rgb)
        overlay = model.render(rgb, result, alpha=0.65)
        cv2.imwrite(str(output / "frames" / f"{i:06d}.jpg"), bgr)
        cv2.imwrite(str(output / "overlays" / f"{i:06d}.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        hands = []
        for h in result:
            hands.append(
                dict(
                    is_right=bool(h.is_right),
                    keypoints_2d=h.keypoints_2d.tolist(),
                    keypoints_3d=h.keypoints_3d.tolist(),
                    cam_t=h.cam_t.tolist(),
                    vertices=h.vertices.tolist(),
                    hand_pose=h.hand_pose.tolist(),
                    global_orient=h.global_orient.tolist(),
                    betas=h.betas.tolist(),
                )
            )
        rows.append(
            dict(
                source_frame=int(i),
                seconds=float(i / fps),
                hands=hands,
                inference_seconds=time.monotonic() - t0,
                focal=float(result.focal),
                cx=float(result.cx),
                cy=float(result.cy),
            )
        )
        thumb = cv2.resize(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), (416, 232))
        cv2.putText(
            thumb,
            f"{i / fps:.2f}s  hands={len(hands)}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
        )
        thumbs.append(thumb)
        print(f"Frame {i}: {len(hands)} hands, {rows[-1]['inference_seconds']:.2f}s", flush=True)
        if len(rows) % 10 == 0:
            (output / "hands.json").write_text(json.dumps(rows))
    cap.release()
    (output / "hands.json").write_text(json.dumps(rows))
    np.save(output / "hand_faces.npy", model.faces)
    while len(thumbs) % 4:
        thumbs.append(np.zeros_like(thumbs[0]))
    cv2.imwrite(
        str(output / "hand-contact-sheet.jpg"),
        np.vstack([np.hstack(thumbs[i : i + 4]) for i in range(0, len(thumbs), 4)]),
    )
    metadata = dict(
        source=str(video.resolve()),
        source_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        source_frames=count,
        fps=fps,
        sampled_frames=len(rows),
        detected_frames=sum(bool(r["hands"]) for r in rows),
        right_label_frames=sum(any(h["is_right"] for h in r["hands"]) for r in rows),
        elapsed_seconds=time.monotonic() - started,
        backend="FastHaMeR 0.8.0 / Core ML CPU_AND_NE",
        handedness_policy="right forced after visual review of single right hand"
        if right_hand
        else "raw detector labels",
        limitations=[
            "Framewise inferred hand geometry; hidden contacts are uncertain",
            "Camera translation is a model estimate, not calibrated metric ground truth",
            "No grasp or robot manipulation evaluated",
        ],
    )
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


def hand_video(output):
    """Encode the saved hand overlays using their source-frame timing."""
    rows = json.loads((output / "hands.json").read_text())
    lines = []
    for i, row in enumerate(rows):
        lines.append(f"file 'overlays/{row['source_frame']:06d}.jpg'")
        if i + 1 < len(rows):
            lines.append(f"duration {rows[i + 1]['seconds'] - row['seconds']:.9f}")
    lines.append(f"file 'overlays/{rows[-1]['source_frame']:06d}.jpg'")
    (output / "overlay-frames.txt").write_text("\n".join(lines) + "\n")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(output / "overlay-frames.txt"),
            "-vf",
            "fps=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output / "hand-overlay.mp4"),
        ],
        check=True,
    )


def export_track(output):
    """Export a dense timeline with NaN rows and validity flags for missing detections."""
    rows = json.loads((output / "hands.json").read_text())
    count = len(rows)
    joints = np.full((count, 21, 3), np.nan)
    pixels = np.full((count, 21, 2), np.nan)
    vertices = np.full((count, 778, 3), np.nan)
    translation = np.full((count, 3), np.nan)
    rotation = np.full((count, 3, 3), np.nan)
    valid = np.zeros(count, dtype=bool)
    for index, row in enumerate(rows):
        hands = [h for h in row["hands"] if h["is_right"]]
        if not hands:
            continue
        hand = hands[0]
        joints[index] = hand["keypoints_3d"]
        pixels[index] = hand["keypoints_2d"]
        vertices[index] = hand["vertices"]
        translation[index] = hand["cam_t"]
        rotation[index] = np.asarray(hand["global_orient"]).reshape(3, 3)
        valid[index] = True
    frames = np.array([r["source_frame"] for r in rows])
    seconds = np.array([r["seconds"] for r in rows])
    if not np.array_equal(frames, np.arange(count)):
        raise ValueError("Full-video extraction must include every frame, including detection gaps")
    np.savez_compressed(
        output / "hand_track.npz",
        source_frame=frames,
        seconds=seconds,
        valid=valid,
        joints_hand_centered_m=joints,
        joints_camera_m=joints + translation[:, None, :],
        joints_image_px=pixels,
        vertices_hand_centered_m=vertices,
        camera_translation_m=translation,
        global_orient=rotation,
    )
    runs = []
    for index in np.flatnonzero(~valid):
        if runs and index == runs[-1][-1] + 1:
            runs[-1].append(int(index))
        else:
            runs.append([int(index)])
    summary = dict(
        frames=count,
        detected_frames=int(valid.sum()),
        detection_coverage=float(valid.mean()),
        missing_intervals=[
            dict(
                first_frame=r[0], last_frame=r[-1], start_s=float(seconds[r[0]]), end_s=float(seconds[r[-1]])
            )
            for r in runs
        ],
        track_sha256=hashlib.sha256((output / "hand_track.npz").read_bytes()).hexdigest(),
        coordinates="OpenCV camera: x right, y down, z forward; estimated metres, not calibrated world positions.",
        landmarks="21 OpenPose-order hand landmarks; these are points, not 21 independent robot controls.",
        missing_data="NaN with valid=false. No interpolation, invented body pose or hidden-hand filling.",
        scope="Video hand estimates only; no object tracking, robot retargeting or physical grasp claim.",
    )
    (output / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    """Extract all source frames and export an auditable right-hand track."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mano", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache/hand-models"))
    args = parser.parse_args()
    hand_pass(args.video, args.output, args.mano, samples=0, cache=args.cache, right_hand=True)
    export_track(args.output)
    hand_video(args.output)


if __name__ == "__main__":
    main()
