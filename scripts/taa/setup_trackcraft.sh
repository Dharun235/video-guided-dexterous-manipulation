#!/usr/bin/env bash
set -euo pipefail
root=/workspace/taa-work
export HF_HOME=/workspace/hf-cache
export UV_CACHE_DIR="$root/uv-cache"
export WANDB_MODE=disabled
# Prepare these pinned checkouts before invoking the script. No training runs.
test "$(git -C "$root/TrackCraft3r" rev-parse HEAD)" = 21e8fcaf4b6375b3044cead210d5808e1d81760b
test "$(git -C "$root/TAA" rev-parse HEAD)" = ba06907ee29daaf6563e4e87d104e2d26d4327d1
test "$(git -C "$root/DA3" rev-parse HEAD)" = 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
uv venv --python 3.11 "$root/env-trackcraft"
py="$root/env-trackcraft/bin/python"
uv pip install --python "$py" torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python "$py" 'setuptools<81' wheel 'numpy<2' 'huggingface_hub<1' pip
uv pip install --python "$py" --no-build-isolation -e "$root/TrackCraft3r" \
  transformers==4.46.3 peft==0.13.2 'huggingface_hub<1' 'numpy<2' \
  opencv-python-headless==4.11.0.86
uv pip install --python "$py" hatchling editables addict jaxtyping omegaconf \
  moviepy==1.0.3 plyfile pycolmap evo e3nn pillow_heif open3d==0.18.0 fastapi uvicorn
uv pip install --python "$py" --no-deps --no-build-isolation -e "$root/DA3"
"$py" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.1-T2V-1.3B', revision='37ec512624d61f7aa208f7ea8140a131f93afc9a',
                  allow_patterns=['diffusion_pytorch_model*.safetensors', 'models_t5_umt5-xxl-enc-bf16.pth',
                                  'Wan2.1_VAE.pth', 'google/*', 'config.json', 'configuration.json'],
                  local_dir='/workspace/taa-work/TrackCraft3r/checkpoints/wan_models/Wan-AI/Wan2.1-T2V-1.3B')
snapshot_download('trackcraft3r/checkpoint', revision='e397c1050bd540062cbd6642ba06a344ca80f116',
                  local_dir='/workspace/taa-work/TrackCraft3r/checkpoints/trackcraft3r')
PY
# Keep the installation log and freeze the environment after inference passes.
