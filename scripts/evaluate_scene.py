"""Download pinned evaluation clips and check automatic Stage 1 outcomes on CPU."""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from reconstruction.scene_check import dump, run


def fetch(clips, cache):
    """Fetch public clips only when requested and verify their recorded checksums."""
    cache.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        path = cache / clip["name"]
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(clip["url"], temporary)
            temporary.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != clip["sha256"]:
            raise ValueError(f"Validation clip checksum differs: {clip['name']}")


def controls(cache, names=None):
    """Derive labelled adverse cases from the downloaded gesture clip, preserving source identity."""
    capture = cv2.VideoCapture(str(cache / "dex-gestures.mp4"))
    frames = []
    for index in range(120):
        ok, image = capture.read()
        if not ok:
            break
        if index % 2 == 0:
            frames.append(cv2.resize(image, (640, 360)))
    capture.release()
    if len(frames) < 30:
        raise ValueError("Gesture control source is too short or undecodable")
    for name in ["blur", "blackout", "occlusion", "cut", "two-hands", "corrupt"]:
        if names is not None and name + ".avi" not in names:
            continue
        path = cache / (name + ".avi")
        if name == "corrupt":
            path.write_bytes(b"invalid container")
        else:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15, (640, 360))
            if not writer.isOpened():
                raise RuntimeError("MJPG encoder unavailable")
            for i, frame in enumerate(frames):
                image = frame.copy()
                if name == "blur":
                    image = cv2.GaussianBlur(image, (0, 0), 24)
                if name == "blackout" or name == "occlusion" and 20 <= i < 30:
                    image = np.zeros_like(image)
                if name == "cut" and i >= 30:
                    image = cv2.bitwise_not(image)
                if name == "two-hands":
                    image = np.hstack(
                        [cv2.resize(image, (320, 360)), cv2.resize(cv2.flip(image, 1), (320, 360))]
                    )
                writer.write(image)
            writer.release()
        yield dict(
            name=path.name,
            category="control",
            mode="gesture",
            transformation=name,
            parent="dex-gestures.mp4",
            expected_blocked=name in ["blur", "blackout", "corrupt"],
        )


def summary(clip, report):
    """Score independently labelled gate expectations without treating review as scene success."""
    seen = report.get("hand_evidence", {}).get("frames_with_hands", 0)
    environment = report.get("error_kind")
    if clip["category"] == "negative_decode":
        matched = report["status"] == "blocked" and not environment
    elif clip["category"].startswith("negative"):
        matched = report["status"] == "blocked" and not environment
    elif clip["category"] == "control":
        reasons = {r["code"] for r in report["reasons"]}
        required = {
            "blur": "severe_blur",
            "blackout": "severe_exposure",
            "cut": "camera_cut",
            "two-hands": "multiple_hands",
            "corrupt": "invalid_video",
        }
        name = clip["transformation"]
        matched = (
            (report["status"] == "blocked") == clip["expected_blocked"]
            and not environment
            and (required.get(name) in reasons if name in required else seen >= 2)
        )

    else:
        matched = seen >= 2 and report["status"] != "blocked" and not environment
    return dict(
        name=clip["name"],
        category=clip["category"],
        mode=clip["mode"],
        input_sha256=report.get("input_sha256"),
        status=report["status"],
        reasons=[r["code"] for r in report["reasons"]],
        hand_frames=seen,
        inspected_frames=report.get("hand_evidence", {}).get("inspected_frames", 0),
        elapsed_s=report["elapsed_s"],
        gate_expectation_matched=bool(matched),
        environment_or_inference_error=environment,
        scene_proposals=sum("scene" in v for v in report.get("scene_views", [])),
    )


def compare_segmenters(models):
    """Compare CPU proposals on one frame; overlap uses an estimated SAM3 reference mask."""
    from reconstruction.scene_geometry import proposal_points
    from reconstruction.scene_models import Segmenter

    capture = cv2.VideoCapture("data/book5/video.mp4")
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise ValueError("Cannot decode Book5 benchmark frame")
    image = cv2.resize(image, (512, 288))
    target = cv2.imread("data/book5/inputs/masks/000000.png", cv2.IMREAD_GRAYSCALE)
    target = cv2.resize(target, (512, 288), interpolation=cv2.INTER_NEAREST) > 0
    results = []
    for variant in ["efficient", "sam2"]:
        model = Segmenter(models, variant)
        started = time.monotonic()
        candidates = model.proposals(image, proposal_points(image, []))
        elapsed = time.monotonic() - started
        overlaps = [
            float((c["mask"] & target).sum() / max(1, (c["mask"] | target).sum())) for c in candidates
        ]
        results.append(
            dict(
                backend=variant,
                elapsed_s=elapsed,
                candidates=len(candidates),
                oracle_best_candidate_iou=max(overlaps, default=0),
                providers=model.encoder.get_providers(),
            )
        )
    return dict(
        results=results,
        reference="estimated SAM3 mask, not independent ground truth",
        interpretation="Oracle candidate overlap on one image; not automatic selection accuracy or a general speed benchmark",
    )


def fusion_summary(clip, report, output):
    """Separate input refusal, reconstruction failure and exported maps needing review."""
    if report.get("reason") == "insufficient_registered_depth_views":
        outcome = "reconstruction_blocked"
    elif "triangles" in report:
        outcome = "scene_exported_for_review"
    else:
        outcome = "input_" + report["status"]
    return dict(
        name=clip["name"],
        category=clip["category"],
        status=report["status"],
        outcome=outcome,
        source_frames=report.get("source_frames"),
        registered_frames=report.get("registered_frames"),
        depth_views=report.get("depth_views"),
        elapsed_s=report.get("elapsed_s"),
        median_alignment_m=report.get("median_keyframe_alignment_m"),
        reasons=report.get("reasons", report.get("reason")),
        environment_exported=(output / "environment/scene.xml").is_file(),
        reconstruction_accuracy_verified=False,
    )


def evaluate_fusion(args, clips):
    """Run the same full-video fusion command on selected downloaded clips."""
    from reconstruction.video_fusion import run as fuse

    results = []
    for clip in clips:
        output = args.output / Path(clip["name"]).stem
        report = fuse(args.cache / clip["name"], output, args.models, depth_stride=args.depth_stride)
        results.append(fusion_summary(clip, report, output))
        dump(
            args.output / "summary.json",
            dict(
                clips=results,
                stage="full_video_static_fusion",
                reconstruction_accuracy_verified=False,
                raw_videos_redistributed=False,
            ),
        )
        print(clip["name"], results[-1]["outcome"], flush=True)


def main():
    """Save per-clip reports locally and a compact evaluation summary for review."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--fusion",
        action="store_true",
        help="Evaluate whole-video static fusion instead of input/scene proposals",
    )
    parser.add_argument(
        "--clips", nargs="+", help="Select downloaded clip filenames from the source manifest"
    )
    parser.add_argument("--depth-stride", type=int, default=30)
    parser.add_argument(
        "--compare-segmenters",
        action="store_true",
        help="Also test the optional SAM2 ONNX pair against EfficientSAM",
    )
    parser.add_argument("--cache", type=Path, default=Path(".cache/scene-validation"))
    parser.add_argument("--models", type=Path, default=Path(".cache/scene-models"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.depth_stride < 1:
        parser.error("Depth stride must be positive")
    if args.fusion and args.compare_segmenters:
        parser.error("Segmentation comparison is separate from fusion evaluation")
    manifest = json.loads(Path("reconstruction/SCENE_VALIDATION.json").read_text())
    sources = manifest["clips"]
    control_names = {name + ".avi" for name in manifest["controls"]}
    known = {c["name"] for c in sources} | control_names
    requested = set(args.clips) if args.clips else ({c["name"] for c in sources} if args.fusion else known)
    unknown = requested - known
    if unknown:
        parser.error("Unknown clips: " + ", ".join(sorted(unknown)))
    selected_sources = [c for c in sources if c["name"] in requested]
    selected_controls = requested & control_names
    downloads = list(selected_sources)
    if selected_controls and not any(c["name"] == "dex-gestures.mp4" for c in downloads):
        downloads.append(next(c for c in sources if c["name"] == "dex-gestures.mp4"))
    if args.download:
        fetch(downloads, args.cache)
    clips = (
        selected_sources + list(controls(args.cache, selected_controls))
        if selected_controls
        else selected_sources
    )
    for clip in clips:
        path = args.cache / clip["name"]
        if not path.exists():
            raise FileNotFoundError(f"{path}; rerun with --download")
        if clip.get("sha256") and hashlib.sha256(path.read_bytes()).hexdigest() != clip["sha256"]:
            raise ValueError(f"Validation clip checksum differs: {clip['name']}")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.fusion:
        evaluate_fusion(args, clips)
        return
    results = []
    for clip in clips:
        path = args.cache / clip["name"]
        if not path.exists():
            raise FileNotFoundError(f"{path}; rerun with --download")
        report = run(path, args.output / path.stem, args.models, mode=clip["mode"], views=2)
        results.append(summary(clip, report))
        print(clip["name"], report["status"], flush=True)
        dump(
            args.output / "summary.json",
            dict(
                clips=results,
                all_gate_expectations_matched=all(r["gate_expectation_matched"] for r in results),
                scene_accuracy_validated=False,
                raw_videos_redistributed=False,
            ),
        )
    if args.compare_segmenters:
        document = json.loads((args.output / "summary.json").read_text())
        document["segmentation_comparison"] = compare_segmenters(args.models)
        dump(args.output / "summary.json", document)
    if not all(r["gate_expectation_matched"] for r in results):
        raise SystemExit("Some validation expectations failed; inspect summary.json")


if __name__ == "__main__":
    main()
