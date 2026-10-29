"""Run unmodified TAA SAM3 segmentation with CPU offload for long clips."""
import importlib.util
import os
import sys
from pathlib import Path


def main():
    root = Path(os.environ.get("TAA_ROOT", "/workspace/taa-work/TAA"))
    source = root / "scripts/02_sam3_segment.py"
    # Avoid treating the checkout's parent folder as a SAM3 namespace package.
    sys.path[0] = str(source.parent)
    spec = importlib.util.spec_from_file_location("taa_segment", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = module.build_sam3_predictor

    def build(*args, **kwargs):
        predictor = factory(*args, **kwargs)
        original = predictor.handle_request

        def handle(request):
            if request.get("type") == "start_session":
                request = dict(request, offload_video_to_cpu=True,
                               offload_state_to_cpu=True)
            return original(request=request)

        predictor.handle_request = handle
        return predictor

    module.build_sam3_predictor = build
    module.main()


if __name__ == "__main__":
    main()
