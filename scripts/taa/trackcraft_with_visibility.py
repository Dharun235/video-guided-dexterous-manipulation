"""Run unmodified TAA stage 08 and retain TrackCraft3R visibility probabilities.

The original stage exports XYZ flow without visibility. Hidden predicted points
must not silently become observed rigid-pose constraints.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


class VisibilityRunner:
    """Reuse one model for both scheduled and adaptively requested windows."""

    def __init__(self, taa_repo, trackcraft_repo, arguments):
        sys.path.insert(0, str(trackcraft_repo.resolve()))
        import evaluation.wan_scene_flow_predictor as predictor_module
        self.predictor_module = predictor_module
        self.original = predictor_module.WanSceneFlowPredictor
        self.instance = None
        self.visibility = None
        self.arguments = arguments
        self.trackcraft_repo = trackcraft_repo
        runner = self

        class WithVisibility(self.original):
            def predict(self, images_pil, *positional, **keywords):
                result = super().predict(images_pil, *positional, **keywords)
                probabilities = self._last_vis_dense
                if probabilities is None or len(probabilities) != len(images_pil):
                    raise RuntimeError("Visibility prediction must cover every selected frame")
                runner.visibility = np.asarray(probabilities, np.float32)
                return result

        def cached_predictor(*positional, **keywords):
            if runner.instance is None:
                runner.instance = WithVisibility(*positional, **keywords)
            return runner.instance

        predictor_module.WanSceneFlowPredictor = cached_predictor
        self.source = taa_repo / "scripts/08_trackcraft_flow.py"
        spec = importlib.util.spec_from_file_location("taa_trackcraft", self.source)
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.module
        spec.loader.exec_module(self.module)

    def track(self, scene):
        self.visibility = None
        sys.argv = [str(self.source), "--trackcraft-repo", str(self.trackcraft_repo),
                    "--scene-dir", str(scene), *self.arguments]
        self.module.main()
        if self.visibility is None:
            raise RuntimeError("No completed inference")
        export_visibility(scene, self.visibility)

    def close(self):
        self.predictor_module.WanSceneFlowPredictor = self.original


def main():
    parser = argparse.ArgumentParser(__doc__, add_help=False)
    parser.add_argument("--taa-repo", type=Path, required=True)
    parser.add_argument("--trackcraft-repo", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--windows-json", type=Path)
    args, remaining = parser.parse_known_args()
    runner = VisibilityRunner(args.taa_repo, args.trackcraft_repo, remaining)
    try:
        scenes = [args.scene_dir] if args.windows_json is None else [Path(p) for p in json.loads(args.windows_json.read_text())]
        for index, scene in enumerate(scenes):
            print(f"Tracking window {index+1}/{len(scenes)}: {scene}", flush=True)
            runner.track(scene)
    finally:
        runner.close()


def export_visibility(scene, visibility):
    folder = scene / "trackcraft"
    config = json.loads((folder / "config.json").read_text())
    for label in config["labels_processed"]:
        pixels = np.load(folder / label / "pixel_ij.npy")
        destination = folder / label / "visibility"
        destination.mkdir(exist_ok=False)
        for i, frame in enumerate(config["tracked_frames"]):
            np.save(destination / f"{frame:06d}.npy", visibility[i, pixels[:, 0], pixels[:, 1]])
    config["visibility_exported"] = True
    config["visibility_note"] = "Predicted probability, not measured visibility or calibrated confidence"
    (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
