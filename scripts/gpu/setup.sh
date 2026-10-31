#!/usr/bin/env bash
set -euo pipefail
PHYSICS_ROOT="${1:-/workspace/physics}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$PHYSICS_ROOT"
if [[ ! -d "$PHYSICS_ROOT/spider/.git" ]]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/facebookresearch/spider.git "$PHYSICS_ROOT/spider"
fi
git -C "$PHYSICS_ROOT/spider" sparse-checkout set spider examples docs/guide
git -C "$PHYSICS_ROOT/spider" fetch --depth 1 origin 44717007de41cbef7565dff7ff9f4453557a2d3d
git -C "$PHYSICS_ROOT/spider" checkout --detach 44717007de41cbef7565dff7ff9f4453557a2d3d
if [[ ! -d "$PHYSICS_ROOT/env" ]]; then
  uv venv --system-site-packages "$PHYSICS_ROOT/env"
fi
uv pip install --python "$PHYSICS_ROOT/env/bin/python" -r "$SCRIPT_DIR/requirements-linux.txt"
"$PHYSICS_ROOT/env/bin/python" -c 'import torch; assert torch.cuda.is_available(), "A CUDA-enabled PyTorch image is required"; print(torch.__version__)'
