"""Inspect a video and automatically build a checked scene proposal on CPU."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .scene_geometry import export_scene, proposal_points, rank_objects
from .video_checks import assess, inspect_video


def dump(path, value):
    """Write portable JSON, rejecting nonfinite values rather than hiding invalid estimates."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def overlay(row, candidates):
    """Show hand estimates and ranked object proposals on the source frame."""
    image = row["image"].copy()
    h, w = image.shape[:2]
    palette = [[40, 190, 70], [230, 130, 30], [180, 40, 180]]
    for i, c in enumerate(candidates[:3]):
        mask = c["mask"]
        image[mask] = (image[mask] * 0.5 + np.array(palette[i]) * 0.5).astype(np.uint8)
        x, y, _, _ = c["bounds_px"]
        cv2.putText(image, f"candidate {i + 1}", (x, max(15, y)), 0, 0.45, palette[i], 1)
    for hand in row["hands"]:
        for p in hand["landmarks"]:
            cv2.circle(image, (round(p[0] * w), round(p[1] * h)), 2, (0, 255, 255), -1)
    cv2.putText(
        image, f"frame {row['frame']} / {row['time_s']:.2f}s", (8, h - 10), 0, 0.5, (255, 255, 255), 1
    )
    return image


def run(video, output, models, segmenter="efficient", mode="auto", views=3):
    """Run video gates and bounded neural reconstruction; always save a reasoned stage report."""
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    report = dict(
        runtime={
            name: version(name)
            for name in ["mediapipe", "onnxruntime", "torch", "mujoco", "numpy", "opencv-contrib-python"]
        },
        implementation_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*Path(__file__).parent.glob("scene*.py"), Path(__file__).parent / "video_checks.py"]
        },
        schema_version=1,
        stage=1,
        input_name=video.name,
        status="blocked",
        reasons=[],
        device="cpu",
        platform=platform.platform(),
        mode=mode,
        thresholds=dict(
            severe_blur_variance=12,
            severe_blur_fraction=0.7,
            severe_clipping_fraction=0.8,
            cut_histogram_distance=0.65,
            cut_pixel_jump=0.18,
        ),
        metric_scale="estimated, not measured",
        automatic=True,
        manual_annotations=0,
    )

    def finish():
        report["elapsed_s"] = time.monotonic() - started
        dump(output / "report.json", report)
        return report

    people = None
    facts, rows = inspect_video(video)
    report["video"] = facts
    if "error" in facts:
        report.update(assess(facts, []))
        return finish()
    report["input_sha256"] = hashlib.sha256(video.read_bytes()).hexdigest()
    try:
        if not (models / "hand_landmarker.task").is_file():
            raise FileNotFoundError("Hand model missing; run python3 -m scripts.setup_scene")
        from .scene_models import Hands, MetricGeometry, People, Segmenter

        detector = Hands(models)
        try:
            for row in rows:
                row["hands"] = detector(row["image"])
            if not any(r["hands"] for r in rows):
                facts, rows = inspect_video(video, sample_hz=10)
                for row in rows:
                    row["hands"] = detector(row["image"])
                report["video"] = facts
                report["dense_hand_verification"] = True
        finally:
            detector.close()
        report.update(assess(facts, rows))
        report["hand_evidence"] = dict(
            inspected_frames=len(rows),
            frames_with_hands=sum(bool(r["hands"]) for r in rows),
            missing_sample_times_s=[r["time_s"] for r in rows if not r["hands"]],
            inference="independent-frame CPU hand detection; handedness score is not presence confidence",
        )
        dump(output / "observations.json", [{k: v for k, v in r.items() if k != "image"} for r in rows])
        selected = np.linspace(0, len(rows) - 1, min(6, len(rows))).astype(int)
        contact = (
            np.vstack(
                [
                    np.hstack([cv2.resize(overlay(rows[i], []), (320, 180)) for i in selected[j : j + 3]])
                    for j in range(0, len(selected), 3)
                ]
            )
            if len(selected) == 6
            else overlay(rows[0], [])
        )
        cv2.imwrite(str(output / "preview.jpg"), contact)
        if report["status"] == "blocked" or any(
            r["code"] in ["camera_cut", "multiple_hands"] for r in report["reasons"]
        ):
            return finish()
        if mode == "gesture":
            report["scene_required"] = False
            return finish()
        visible = [i for i, r in enumerate(rows) if r["hands"]]

        def closure(index):
            p = np.array(rows[index]["hands"][0]["landmarks"])[:, :2]
            return float(
                np.mean(
                    np.linalg.norm(p[[8, 12, 16, 20]] - p[0], axis=1)
                    / np.maximum(np.linalg.norm(p[[5, 9, 13, 17]] - p[0], axis=1), 1e-6)
                )
            )

        chosen = sorted(visible, key=closure)[: min(views * 2, len(visible))]

        people = People(models)
        sam = Segmenter(models, segmenter)
        metric = MetricGeometry(models)
        estimates = []
        object_frames = []
        stage_reasons = []
        built_views = 0
        for index in chosen:
            if built_views >= views:
                break
            row = rows[index]
            image = row["image"]
            t = time.monotonic()
            person_mask = people(image)
            candidates = rank_objects(
                sam.proposals(image, proposal_points(image, row["hands"])), row["hands"], image, person_mask
            )
            cv2.imwrite(str(output / f"frame-{row['frame']:06d}.jpg"), overlay(row, candidates))
            summary = dict(
                frame=row["frame"],
                time_s=row["time_s"],
                candidates=[{k: v for k, v in c.items() if k != "mask"} for c in candidates],
                segmentation_s=time.monotonic() - t,
            )
            if not candidates:
                summary["reason"] = "object_not_identified"
                object_frames.append(summary)
                continue
            # Keep identity uncertainty explicit; this ranking does not establish a demonstrated contact.
            if len(candidates) > 1 and candidates[0]["rank"] - candidates[1]["rank"] < 0.08:
                summary["ambiguous"] = True
            small = cv2.resize(image, (512, round(image.shape[0] * 512 / image.shape[1])))
            t = time.monotonic()
            geometry = metric(small)
            summary["geometry_s"] = time.monotonic() - t
            geometry["person_mask"] = cv2.resize(
                person_mask.astype(np.uint8),
                (small.shape[1], small.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            mask = cv2.resize(
                candidates[0]["mask"].astype(np.uint8),
                (small.shape[1], small.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            try:
                result = export_scene(
                    output / f"view-{row['frame']:06d}", small, geometry, mask, row["hands"]
                )
                summary.update(result)
                if result["scene_check"]["stable_under_assumed_physics"]:
                    built_views += 1
                    estimates.append(result["visible_extent_m"])
                else:
                    summary["reason"] = "unstable_scene_proposal"
                summary["scene"] = f"view-{row['frame']:06d}/scene.xml"
            except (ValueError, RuntimeError) as error:
                summary["reason"] = str(error)
            object_frames.append(summary)
        report["scene_views"] = object_frames
        report["models"] = dict(
            hands="MediaPipe hand_landmarker v1",
            segmentation=segmenter,
            geometry="MoGe-2 vits-normal",
            revision_manifest="reconstruction/SCENE_MODELS.json",
        )
        if not estimates:
            stage_reasons.append(
                dict(
                    code="scene_not_reconstructed",
                    severity="review",
                    message="No supported object geometry could be reconstructed. Inspect the candidate overlays.",
                )
            )
        else:
            a = np.array(estimates)
            variation = np.std(a, axis=0) / np.maximum(np.mean(a, axis=0), 0.001)
            report["visible_extent_variation_fraction"] = variation.tolist()
            report["dimension_consistency_is_not_accuracy"] = True
            if len(estimates) < 2 or np.max(variation) > 0.25:
                stage_reasons.append(
                    dict(
                        code="uncertain_scale_or_identity",
                        severity="review",
                        message="Visible dimensions vary or have insufficient repeated evidence; object identity, occlusion and scale need review.",
                    )
                )
            if any(v.get("reason") == "unstable_scene_proposal" for v in object_frames):
                stage_reasons.append(
                    dict(
                        code="unstable_scene_proposal",
                        severity="review",
                        message="At least one scene failed passive settling under the assumed physics.",
                    )
                )
            if any(v.get("ambiguous") for v in object_frames):
                stage_reasons.append(
                    dict(
                        code="ambiguous_object",
                        severity="review",
                        message="Several segmented objects have similar interaction scores.",
                    )
                )
            if any(v.get("support_inlier_fraction", 0) < 0.3 for v in object_frames if "scene" in v):
                stage_reasons.append(
                    dict(
                        code="uncertain_support",
                        severity="review",
                        message="The support-plane hypothesis explains too little visible geometry.",
                    )
                )
            stage_reasons.append(
                dict(
                    code="hidden_geometry_unverified",
                    severity="review",
                    message="Single-view hulls contain visible surfaces only; hidden object shape and cross-view identity are not verified.",
                )
            )
        report["reasons"] += stage_reasons
        if stage_reasons and report["status"] == "pass":
            report["status"] = "review"
        report["task_mode"] = mode if mode != "auto" else "manipulation_candidate"
        return finish()
    except (ImportError, FileNotFoundError, OSError) as error:
        report.update(
            status="blocked",
            error_kind="environment",
            reasons=[
                dict(
                    code="inference_environment_error",
                    severity="blocked",
                    message=f"{type(error).__name__}: {error}",
                )
            ],
        )
        return finish()

    except (RuntimeError, ValueError) as error:
        report.update(
            status="blocked",
            error_kind="inference",
            reasons=[
                dict(code="inference_failed", severity="blocked", message=f"{type(error).__name__}: {error}")
            ],
        )
        return finish()
    finally:
        if people is not None:
            people.close()


def main():
    """Accept one video and save reports/previews without requiring manual point prompts."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models", type=Path, default=Path(".cache/scene-models"))
    parser.add_argument("--segmenter", choices=["efficient", "sam2"], default="efficient")
    parser.add_argument("--mode", choices=["auto", "manipulation", "gesture"], default="auto")
    parser.add_argument("--views", type=int, default=3, choices=range(1, 6))
    args = parser.parse_args()
    output = args.output or Path("runs") / ("scene-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    report = run(args.video, output, args.models, args.segmenter, args.mode, args.views)
    print(
        json.dumps(
            dict(status=report["status"], report=str(output / "report.json"), reasons=report["reasons"]),
            indent=2,
        )
    )
    raise SystemExit(2 if report["status"] == "blocked" else 0)


if __name__ == "__main__":
    main()
