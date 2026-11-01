"""Video integrity, quality and temporal checks independent of neural model packages."""

import cv2
import numpy as np


def quality(image):
    """Measure blur and exposure at fixed image width for comparable thresholds."""
    gray = cv2.cvtColor(cv2.resize(image, (640, 360)), cv2.COLOR_BGR2GRAY)
    return dict(
        blur_variance=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        clipped_fraction=float(((gray < 8) | (gray > 247)).mean()),
        mean_luma=float(gray.mean() / 255),
    )


def inspect_video(path, sample_hz=2, max_duration_s=180):
    """Decode the complete clip; sample inference frames while checking every transition."""
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
        cap.release()
        return dict(error="invalid_video", message="The file could not be decoded with usable timing."), []
    if declared / fps > max_duration_s:
        cap.release()
        return dict(
            error="duration_limit", message=f"Clip exceeds the {max_duration_s:g} second analysis budget."
        ), []
    samples = []
    cuts = []
    previous = None
    index = 0
    times = []
    still = 0
    stride = max(1, round(fps / sample_hz))
    while True:
        ok, image = cap.read()
        if not ok:
            break
        small = cv2.resize(image, (160, 90))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        if previous is not None:
            distance = cv2.compareHist(previous[0], hist, cv2.HISTCMP_BHATTACHARYYA)
            jump = float(np.mean(cv2.absdiff(previous[1], small)) / 255)
            still += int(jump < 0.001)
            if distance > 0.65 and jump > 0.18:
                cuts.append(
                    dict(frame=index, time_s=index / fps, histogram_distance=distance, pixel_jump=jump)
                )
        previous = (hist, small)
        times.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000)
        if index % stride == 0:
            scale = min(1, 640 / image.shape[1])
            sample = cv2.resize(image, None, fx=scale, fy=scale)
            samples.append(dict(frame=index, time_s=index / fps, image=sample, quality=quality(sample)))
        index += 1
        if index / fps > max_duration_s:
            break
    cap.release()
    if not index:
        return dict(error="invalid_video", message="No frames could be decoded."), []
    return dict(
        fps=fps,
        decoded_frames=index,
        declared_frames=declared,
        duration_s=index / fps,
        sampled_frames=len(samples),
        sample_hz=sample_hz,
        cuts=cuts,
        incomplete_decode=bool(declared > 0 and index < declared - 1),
        nonmonotonic_timestamps=bool(np.any(np.diff(times) < 0)),
        visual_freeze_fraction=float(still / max(1, index - 1)),
        frozen_fraction=float(sum(np.diff(times) == 0) / max(1, index - 1)),
    ), samples


def assess(video, rows):
    """Return explainable gates; brief missing intervals are distinct from no hand evidence."""
    reasons = []

    def add(code, severity, message):
        reasons.append(dict(code=code, severity=severity, message=message))

    if "error" in video:
        add(video["error"], "blocked", video["message"])
    else:
        if video["incomplete_decode"]:
            add("incomplete_decode", "blocked", "The file ended before its declared frame count.")
        if video["nonmonotonic_timestamps"]:
            add("invalid_timestamps", "blocked", "Video timestamps run backwards.")
        if video.get("visual_freeze_fraction", 0) > 0.98:
            add("static_video", "review", "The clip is almost entirely static; motion evidence needs review.")
        if video["cuts"]:
            add("camera_cut", "review", "Abrupt transitions need separate continuous scenes.")
        if video["duration_s"] < 0.5:
            add("video_too_short", "blocked", "Provide at least half a second of visible movement.")
        if rows:
            if np.mean([r["quality"]["blur_variance"] < 12 for r in rows]) > 0.7:
                add("severe_blur", "blocked", "Most inspected frames lack usable image detail.")
            if np.mean([r["quality"]["clipped_fraction"] > 0.8 for r in rows]) > 0.7:
                add("severe_exposure", "blocked", "Most inspected frames are severely under- or overexposed.")
            visible = [bool(r.get("hands")) for r in rows]
            if not any(visible):
                add(
                    "hand_not_detected",
                    "blocked",
                    "No reliable hand observations were found in the inspected frames. Use a clearer view of the demonstrating hand.",
                )
            elif sum(visible) < 2:
                add("insufficient_hand_evidence", "review", "Only one inspected frame contains a hand.")
            if any(len(r.get("hands", [])) > 1 for r in rows):
                add(
                    "multiple_hands",
                    "review",
                    "The initial manipulation scope supports one demonstrating hand.",
                )
    status = (
        "blocked" if any(r["severity"] == "blocked" for r in reasons) else "review" if reasons else "pass"
    )
    return dict(status=status, reasons=reasons)
