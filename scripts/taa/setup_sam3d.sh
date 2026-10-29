#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/workspace/hf-cache
export UV_CACHE_DIR=/workspace/taa-work/uv-cache
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=4
root=/workspace/taa-work
uv venv --python 3.11 "$root/env-sam3d"
py="$root/env-sam3d/bin/python"
uv pip install --python "$py" torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python "$py" pip 'setuptools<81' wheel packaging ninja 'numpy<2' 'huggingface_hub<1' hf_transfer
uv pip install --python "$py" --no-deps -e "$root/sam3d"
uv pip install --python "$py" hydra-core==1.3.2 hydra-submitit-launcher==1.2.0 loguru==0.7.2 einops easydict tqdm pillow scipy opencv-python-headless trimesh 'open3d==0.18.0' 'timm==0.9.16' 'lightning==2.3.3' 'spconv-cu121==2.3.8' 'xformers==0.0.28.post3' 'point-cloud-utils==0.29.5' 'xatlas==0.0.9' fvcore iopath roma rootutils pyrender h5py pandas matplotlib seaborn ftfy omegaconf 'plyfile<1.1' safetensors 'transformers<5' 'diffusers<0.33' 'imageio[ffmpeg]' pyyaml typeguard
uv pip install --python "$py" kaolin==0.17.0 --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html
uv pip install --python "$py" gradio==5.49.0 addict optree astor einops-exts fasteners jsonlines objsize scikit-image colorama simplejson pyvista pymeshfix igraph rtree
uv pip install --python "$py" --no-build-isolation 'pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47'
uv pip install --python "$py" --no-build-isolation 'MoGe @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b' 'gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7'
# Prefer the published CUDA wheel to a long FlashAttention source build.
"$py" - <<'PY'
import json,urllib.request,subprocess
url='https://api.github.com/repos/Dao-AILab/flash-attention/releases/tags/v2.8.3'
data=json.load(urllib.request.urlopen(url))
assets=[x for x in data['assets'] if 'cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl' in x['name']]
if not assets:raise SystemExit('No matching prebuilt FlashAttention wheel; inspect before compiling')
subprocess.run(['uv','pip','install','--python',__import__('sys').executable,assets[0]['browser_download_url']],check=True)
PY
cd "$root/sam3d"
"$py" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('facebook/sam-3d-objects',local_dir='/workspace/taa-work/sam3d/checkpoints/hf')
PY
"$py" "$root/sam3d/patching/hydra"
# The downloaded pipeline lives under checkpoints/hf/checkpoints. Use
# --checkpoint-tag hf/checkpoints with TAA, and CONDA_PREFIX=/usr/local/cuda
# because upstream notebook/inference.py reads that variable for CUDA_HOME.
