# Preparing video observations

The included recording and reviewed inputs run directly with `make video`.
This guide describes how those inputs are produced. Use the supplied inputs
for the exact Book5 example; neural reconstruction may vary between runs.
A new recording requires new annotations, metric dimensions, hand registration
and reviewed motion phases in `video_g1.book5_sequence`.

## Four-stage video-entry design

This is the planned extension of the calibrated example. Each stage writes a
report and inspectable output before the next stage runs. The first supported
manipulation scope is a continuous tabletop clip with one hand and a rigid
object. Gesture-only clips require hands, but not an object or grasp.

| Stage | Required output | Gate |
|---|---|---|
| 1. Scene and input verification | Camera/scale estimates, support geometry, object masks/mesh, scene preview | Decodable clip, reliable hand evidence, task-required entities, consistent geometry and usable scale |
| 2. Motion and interactions | Timestamped hand/object trajectories, inferred contacts/events and overlay | Identity/pose continuity, explicit gaps, sufficient evidence around interactions |
| 3. Robot retargeting | Native arm/waist/finger plan, feasible grasp candidates and preview | Reach, limits, collisions, fit/contact feasibility and bounded retiming |
| 4. Physical execution and verification | Actual commands/states/contacts, task report and comparison video | Stable contact, object-motion/placement checks and independent motor replay |

Use `pass` to continue, `review` to pause with uncertain intervals and evidence,
and `blocked` when required evidence or feasibility is absent. Store reason
codes, frame intervals, measurements, thresholds and suggested remedies. A model
or environment failure is distinct from an unsuitable video. No numeric confidence
should be presented as a calibrated probability without validation.

Checks run at stage boundaries and over temporal windows within each stage;
critical contact checks run continuously during physical execution. If initial
sampling finds no hand, verify additional frames before returning
`hand_not_detected`: “No reliable hand observations were found; use a clearer
view of the demonstrating hand.” A brief occlusion is a flagged interval, not
proof that the entire task is impossible. Other reasons include invalid video,
camera cut, missing task object, uncertain scale, tracking loss, unreachable
motion and unstable grasp. Ambiguous or unsupported tasks must be reported.

### Automatic dimensions

Evaluate [MoGe-2](https://github.com/microsoft/MoGe) metric point/depth maps and
[UniDepthV2](https://github.com/lpiccinelli-eth/UniDepth) metric depth as candidate
scale estimators. Segment objects, fuse reliable views and estimate visible
extents in the scene coordinate frame. Test temporal scale stability, support
alignment and silhouette consistency; hidden dimensions require uncertainty.
Object-category size priors can assist, but a class name does not determine an
instance's dimensions. Predicted scale, measured scale and inferred hidden
geometry must remain separately labelled.

Compare candidates against measured Book5 dimensions and a separate recording
before selecting a backend. Record dimensional error, temporal variation,
device, memory and time. Check model licenses and pin revisions. The current
MoGe repository defaults to MoGe-3 and documents macOS restrictions; do not
silently upgrade the selected MoGe-2 candidate or assume MPS compatibility.
These candidates are proposed, not installed dependencies of `make video`.

Track implementation and acceptance criteria in the four GitHub stage issues.
A stage PR must demonstrate its output and failures before downstream automation
relies on it. Keep the saved-input Book5 path as the regression example.

## Hand observations on macOS

Requirements: Apple Silicon macOS, uv, FFmpeg and a licensed `MANO_RIGHT.pkl`
obtained from the [MANO project](https://mano.is.tue.mpg.de/). Keep the model local.

```bash
uv venv --python 3.12 .venv-book-reconstruction
uv pip install --python .venv-book-reconstruction/bin/python \
  -r reconstruction/requirements-book-mac.lock.txt
.venv-book-reconstruction/bin/python -m reconstruction.video_hands \
  --video data/book5/video.mp4 --mano /path/to/MANO_RIGHT.pkl \
  --output runs/full-hands
```

FastHaMeR downloads/converts its models on first use. Core ML compilation needs
normal macOS cache access. Inspect the full overlay and missing intervals before
accepting the hand estimates. A detection is not a verified pose or contact.

| File | Contents |
|---|---|
| `manifest.json` | Video checksum, frame count, timing and runtime |
| `hands.json` | Estimated per-frame hand parameters and mesh vertices |
| `hand_track.npz` | All source frames, timestamps, validity mask and arrays |
| `coverage.json` | Detection coverage and missing intervals |
| `hand-overlay.mp4` | Estimated mesh over the recording |

No unseen whole-body pose is inferred from this close-up recording.

## Book map and moving cameras

```bash
.venv-book-reconstruction/bin/python -m reconstruction.book_space build \
  runs/book-map --video data/book5/video.mp4
.venv-book-reconstruction/bin/python -m reconstruction.video_cameras \
  --output runs/book-cameras --scene runs/book-map --hands runs/full-hands
```

`book_space` is annotated for Book5 and defines the reconstruction map scale
(198 × 129 × 18 mm book). `video_cameras` tracks book-plane image features.
The physical scene instead uses measured 220 × 220 × 20 mm dimensions through
`configs/book5.json` and `video_g1.calibration`. Preserve the distinction between
map coordinates and task coordinates.

Review `data/book5/inputs/camera-observations.json` for the corner-observation
format. Fit the physical cameras and scene with the calibration/scene commands
in `video_g1.run_video`; they also run automatically through `make video`.

## Scene observations on Linux/NVIDIA

Tested compute: RTX 6000 Ada, 48 GB VRAM, Linux, CUDA 12.1 and Python 3.11.
Use separate environments from the CPU simulation. Approved Hugging Face access
is required for SAM3 and SAM 3D Objects; authenticate privately on the host.
Model weights and credentials stay outside the repository.

```bash
.venv-book-reconstruction/bin/python -m scripts.taa.prepare_video \
  --video data/book5/video.mp4 --output runs/taa-full
.venv-book-reconstruction/bin/python -m scripts.taa.prepare_video \
  --video data/book5/video.mp4 --stride 60 --output runs/taa-check
```

The first command writes all frames and records FPS-based timestamps. The second selects 14 frames,
including the last frame. Both save the source checksum and frame mapping.
Transfer each folder as an archive with Mac metadata excluded:
`COPYFILE_DISABLE=1 tar --no-xattrs -cf frames.tar -C runs taa-full taa-check`.

### Reconstruction environment

The tested host is Linux with an RTX 6000 Ada, 48 GB VRAM. Model access requires
approved Hugging Face access and a private login on the host. Keep credentials
out of Git. Checkouts used:

| Repository | Revision |
|---|---|
| brains-bots-n-behavior/track-articulate-act | `ba06907ee29daaf6563e4e87d104e2d26d4327d1` |
| ByteDance-Seed/Depth-Anything-3 | `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` |
| facebookresearch/sam3 | `f1647e8d0c1d280991f4d1b7d974c2704537fe66` |
| facebookresearch/sam-3d-objects | `f91db411c50efee93d8db7aeb323885650f6f722` |

Place these checkouts at `/workspace/taa-work/{TAA,DA3,sam3,sam3d}`.
Use each project's installation instructions for DA3 and SAM3 in `env-depth`.
Our [SAM 3D setup script](../scripts/taa/setup_sam3d.sh) creates `env-sam3d`;
it assumes the checkouts, uv and CUDA toolkit are already present. The two
`scripts/taa/*-linux-observed.txt` files record the installed package versions;
editable paths refer to the checkouts on that host.

Copy `scripts/taa/segment_video.py` into `/workspace/taa-work/`, then run:

```bash
cd /workspace/taa-work
export HF_HOME=/workspace/hf-cache
env-depth/bin/python TAA/scripts/00_da3_depth_cameras.py \
  --scene-dir taa-check --da3-root DA3 \
  --model-name depth-anything/DA3NESTED-GIANT-LARGE
env-depth/bin/python segment_video.py \
  --scene-dir taa-check --prompts-json taa-check/prompts.json --version sam3
env-depth/bin/python segment_video.py \
  --scene-dir taa-full --prompts-json taa-full/prompts.json --version sam3
CONDA_PREFIX=/usr/local/cuda env-sam3d/bin/python TAA/scripts/03_sam3d_reconstruct.py \
  --scene-dir taa-check --sam3d-repo /workspace/taa-work/sam3d \
  --checkpoint-tag hf/checkpoints --labels case --candidate-idx 0 \
  --pointmap-source moge --save-input --no-cudnn
env-sam3d/bin/python TAA/scripts/04_sacle_mesh.py \
  --scene-dir taa-check --labels case --candidate-idx 0
```

The wrapper enables CPU offload for full-video segmentation without editing TAA.
The selected SAM 3D release does not accept TAA's extra DA3 point-map arguments,
so reconstruction uses MoGe and stage 04 aligns against DA3 afterward. Upstream
reads `CONDA_PREFIX` as its CUDA location; the environment itself uses uv.
Keep new attempts in new scene folders. TAA's filename `04_sacle_mesh.py` is
intentional here. Download outputs before terminating temporary GPU storage.

### Tracking anchor

Download reconstruction outputs and register the reconstructed case against
the book map:

```bash
.venv-book-reconstruction/bin/python -m reconstruction.taa_scene \
  --taa runs/taa-check --scene runs/book-map --output runs/taa-candidate
.venv-g1/bin/python -m scripts.taa.export_anchor \
  --scene runs/taa-candidate/scene.xml --output runs/object-anchor
```

This exports a mesh/pose candidate and sampled case surface. The included
`data/book5/inputs/tracking-map` and `tracking-anchor` preserve the selected
reconstruction. Physical execution uses measured box collision geometry;
the reconstructed map supplies texture, coordinates and tracking evidence.

## Full-video object tracking

Copy source frames/masks, the initial DA3 calibration, book-camera observations,
anchor files, `scripts/taa/` and `reconstruction/` to the Linux host.
The following commands run from `/workspace/taa-work`; the project modules
are under `project/`. The Mac folder `runs/taa-inputs` and the host folder
`taa-full` contain the same prepared source frames and masks.

Use Linux, uv and an NVIDIA GPU. The tested host has 48 GB VRAM. From
`/workspace/taa-work`, prepare the external checkouts:

```bash
git clone https://github.com/brains-bots-n-behavior/track-articulate-act TAA
git -C TAA checkout ba06907ee29daaf6563e4e87d104e2d26d4327d1
git clone https://github.com/cvlab-kaist/TrackCraft3r TrackCraft3r
git -C TrackCraft3r checkout 21e8fcaf4b6375b3044cead210d5808e1d81760b
git clone https://github.com/ByteDance-Seed/Depth-Anything-3 DA3
git -C DA3 checkout 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
bash setup_trackcraft.sh
```

The setup script downloads public tracking/base-model weights and tokenizer
files. Model files remain outside Git. It pins the Torch-compatible Transformers
and PEFT versions; installing their newest releases breaks this runtime.

## Estimate depth at every frame

```bash
env-trackcraft/bin/python full_depth.py \
  --scene-dir taa-full --da3-root DA3 --reference-da3 taa-check/da3 \
  --book-cameras book-camera-poses.json --batch-size 12
```

The model processes twelve frames at a time. Camera intrinsics stay fixed to
the initial calibration. Book matching estimates each camera pose, and the
visible book plane calibrates each depth map's scale. `da3/quality.json` records
the scale and book-plane residual for every frame. A median plane residual above
15 mm excludes that frame's tracking windows. This check does not validate depth
on the moving case.

## Track overlapping windows

```bash
env-trackcraft/bin/python make_windows.py \
  --scene-dir taa-full --output taa-windows --length 12 --step 8
env-trackcraft/bin/python trackcraft_with_visibility.py \
  --taa-repo TAA --trackcraft-repo TrackCraft3r --scene-dir taa-full \
  --windows-json taa-windows/windows.json \
  --checkpoint TrackCraft3r/checkpoints/trackcraft3r/model.safetensors \
  --model-cache TrackCraft3r/checkpoints/wan_models \
  --labels case --num-frames 12 --frame-stride 1 \
  --height 480 --width 832 --no-track-video
```

The 748-frame recording needs 93 windows. Input files use links rather than
copies. The wrapper reuses one loaded model and saves predicted visibility
alongside TAA's 3D flow. It leaves the upstream source unchanged.

## Fit and join rigid motion

```bash
PYTHONPATH=project env-trackcraft/bin/python bridge_object_motion.py \
  --taa-repo TAA --trackcraft-repo TrackCraft3r --scene-dir taa-full \
  --windows taa-windows/windows.json --project project \
  --initial-pose initial_pose.json --surface case_surface.npy \
  --checkpoint TrackCraft3r/checkpoints/trackcraft3r/model.safetensors \
  --model-cache TrackCraft3r/checkpoints/wan_models \
  --output object-bridges --max-bridges 16 --coherent-depth \
  --da3-repo DA3 --book-cameras book-camera-poses.json
```

Point qualification requires predicted visibility of at least 0.7 in both frames
and projection inside the visible case mask. RANSAC/Kabsch fits rotation and
translation with at least twenty points and 60% inliers. The fit threshold is
the larger of 2.5 mm and 4% of the visible reference-cloud extent.

Overlapping pose estimates must agree within 15 mm and 0.35 rad. The projected
mesh must cover at least 80% of the visible case mask, allowing a four-pixel
boundary margin. These checks detect some failures; they do not measure true
pose accuracy.

When joining fails, the adaptive pass requests a closer reference window just
before the break. It also estimates depth jointly within that window: otherwise
depth changes between independent batches can look like nonrigid object motion.
The original global-depth inputs are retained beside the corrected window.
The pass does not interpolate a pose or attach the object to the hand.
Repeated failure at the same break stops further requests there. Each fit
has a new folder, and `latest.json` identifies the final candidate.

The run also recomputes the nine original windows with rejected rigid fits using
joint depth estimates, rather than rerunning every window:

```bash
PYTHONPATH=project env-trackcraft/bin/python find_bad_windows.py \
  --windows taa-windows/windows.json --output repair-starts.json
env-trackcraft/bin/python coherent_repairs.py \
  --taa-repo TAA --trackcraft-repo TrackCraft3r --da3-repo DA3 \
  --book-cameras book-camera-poses.json --scene-dir taa-full \
  --starts-json repair-starts.json \
  --checkpoint TrackCraft3r/checkpoints/trackcraft3r/model.safetensors \
  --model-cache TrackCraft3r/checkpoints/wan_models --output coherent-repairs
```

### Download and join

Archive window outputs and full depth data before terminating the host:

```bash
env-trackcraft/bin/python pack_tracking.py --root /workspace/taa-work \
  --windows-name taa-windows --scene-name taa-full \
  --extra-names object-bridges coherent-repairs --output tracking-results.tar.gz
tar -C taa-full -czf depth-full.tar.gz da3
```

Download and extract the archives on the Mac, preserving the directory layout.
Restore image links and join the rigid estimates:

```bash
.venv-g1/bin/python -m scripts.taa.restore_windows \
  --windows-dir runs/taa-windows --scene-dir runs/taa-full \
  --source-inputs runs/taa-inputs \
  --extra-window-root runs/object-bridges runs/coherent-repairs
.venv-g1/bin/python -m scripts.taa.hand_occlusion \
  --hands runs/full-hands --taa runs/taa-inputs --output runs/hand-occlusion
.venv-g1/bin/python -m reconstruction.stitch_object_windows \
  --windows runs/taa-windows/windows.json --scene-data runs/taa-full \
  --initial-pose runs/object-anchor/initial_pose.json \
  --surface runs/object-anchor/case_surface.npy --occlusion-masks runs/hand-occlusion \
  --hand-prior runs/full-hands/hand_track.npz --hand-anchor-frame 320 \
  --backward-pass --output runs/object-motion
.venv-g1/bin/python -m scripts.taa.check_object_track \
  --track runs/object-motion --scene runs/taa-candidate/scene.xml \
  --surface runs/object-anchor/case_surface.npy
```

Pose joining uses visibility ≥0.7, RANSAC/Kabsch rigid fits and visible-mask
agreement. Adaptive windows recompute depth jointly near tracking breaks.
An uncertain component can use an image/depth/hand-orientation initialization;
its separate component ID and `requires_review` flag remain explicit. Missing
poses stay NaN. The selected input has 740/748 pose candidates, 310 review flags
and eight gaps. Rigid residuals measure internal consistency, not true 3D accuracy.

## Register and fit hand gestures

`video_g1.refine_hand` registers hand geometry to image landmarks
with inferred support/contact priors; inspect its report and selected frames.
Generate the contact-aware registration with full-video masks:

```bash
.venv-g1/bin/python -m video_g1.refine_hand \
  --hands runs/full-hands --cameras runs/video-01/cameras \
  --objects data/book5/inputs/object-track.npz \
  --case-masks runs/taa-inputs/masks/case --config configs/book5.json \
  --output runs/hand-registration
```

`video_g1.hand_fit` fits native arm/hand targets using the registered landmarks:

```bash
.venv-g1/bin/python -m video_g1.hand_fit \
  --hands runs/full-hands --cameras runs/video-01/cameras \
  --scene runs/video-01/scene --calibration runs/video-01/scene \
  --hand-length-m 0.23 \
  --refined-hands runs/hand-registration \
  --output runs/hand-fit-01
```

The output `hand-prior.npz` separates estimated landmarks and finger commands.
Review reach, palm orientation, initial/final gestures and missing detections
before selecting a fit. The supplied initial-gesture fit and RH56-derived palm
prior are calibrated inputs; their source fields are documented in
[data/book5/README.md](../data/book5/README.md). Phase boundaries and grasp
pre-shapes are explicit in `video_g1.book5_sequence` and `video_g1.wrap_grasp`.
Use the physical execution and contact checks in [methods](methods.md) to
validate those proposals. Keep raw observations separate from execution targets.

Pinned external revisions are in `scripts/taa/UPSTREAM.json`; observed NVIDIA
environment packages are in `scripts/taa/*-linux-observed.txt`. Hand-inference
packages are pinned in `reconstruction/requirements-book-mac.lock.txt`.
