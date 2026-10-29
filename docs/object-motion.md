# Estimate the moving case pose

The selected path uses TAA's TrackCraft3R stage on overlapping video windows.
DA3 supplies per-frame depth. Visible 3D point correspondences supply rigid
motion; overlapping estimates connect that motion to the initial case pose.
The output contains position, rotation, timestamps, validity and fit diagnostics.

Current result: 740/748 candidate poses. Of these, 430 connect to the initial
calibration and 310 belong to a flagged, uncertain relocalized component.
Eight frames remain invalid. [Video, data and scene](../examples/object-motion/README.md)

## Inputs

- Complete source frames and SAM3 masks from [scene reconstruction](taa.md).
- The initial DA3 intrinsics and reviewed book-camera observations.
- The case mesh candidate and its initial pose.

The pipeline uses an estimated 198 × 129 × 18 mm book to establish camera and
depth scale. This is a scale prior, not a measured calibration. Hand silhouettes exclude occluded edges during refinement. An explicitly
flagged relocalization also uses a hand-orientation prior; it is an uncertain
initialization, not an observed rigid attachment.

## Export the initial anchor

For the included recording, `examples/object-motion/` contains the exact scene,
anchor and sampled surface used by the published candidate. Rebuilding the
estimated mesh can change the result; use those supplied inputs to inspect it.

On the Mac:

```bash
.venv-g1/bin/python -m scripts.taa.export_anchor \
  --scene runs/taa-candidate/scene.xml --output runs/object-anchor
```

Transfer `initial_pose.json`, `case_surface.npy`, the source inputs, camera
observations, `scripts/taa/` and `reconstruction/` to the GPU host. Keep the latter
under `/workspace/taa-work/project/reconstruction/`. Checkouts and model
revisions are recorded in [UPSTREAM.json](../scripts/taa/UPSTREAM.json).
The examples call the Mac source-input folder `runs/taa-inputs` and its GPU copy
`taa-full`. Downloaded camera/depth metadata uses `runs/taa-full` on the Mac;
the restore command links it to the source images and masks.

## Set up tracking

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

## Save and review

Use `pack_tracking.py` to save the window outputs, metadata and adaptive results
without copying models, linked input images or derived reference point maps.
Archive `taa-full/da3/` separately to retain the full depth data.

```bash
env-trackcraft/bin/python pack_tracking.py --root /workspace/taa-work \
  --windows-name taa-windows --scene-name taa-full \
  --extra-names object-bridges coherent-repairs --output tracking-results.tar.gz
tar -C taa-full -czf depth-full.tar.gz da3
```

After downloading and extracting the tracking archive on the Mac:

```bash
.venv-g1/bin/python -m scripts.taa.restore_windows \
  --windows-dir runs/taa-windows --scene-dir runs/taa-full \
  --source-inputs runs/taa-inputs \
  --extra-window-root runs/object-bridges runs/coherent-repairs
.venv-g1/bin/python -m scripts.taa.hand_occlusion \
  --hands runs/full-hands --taa runs/taa-inputs --output runs/hand-occlusion
.venv-g1/bin/python -m scripts.taa.case_depth_summary \
  --taa runs/taa-full --archive runs/depth-full.tar.gz
.venv-g1/bin/python -m reconstruction.stitch_object_windows \
  --windows runs/taa-windows/windows.json --scene-data runs/taa-full \
  --initial-pose runs/object-anchor/initial_pose.json \
  --surface runs/object-anchor/case_surface.npy --occlusion-masks runs/hand-occlusion \
  --hand-prior runs/full-hands/hand_track.npz --hand-anchor-frame 320 \
  --backward-pass --output runs/object-motion
.venv-g1/bin/python -m scripts.taa.preview_object_motion \
  --taa runs/taa-full --taa-check runs/taa-check \
  --scene runs/taa-candidate/scene.xml \
  --cameras runs/book5-cameras/camera_poses.json \
  --track runs/object-motion --output runs/object-review
```

Inspect `object-motion.mp4`, `report.json` and `windows.json`. Rejected poses are
NaN in `object_track.npz` and appear explicitly in the video. The MuJoCo preview
imposes estimated poses; motor-driven manipulation is a subsequent stage.

The local image check projects the mesh against visible case edges, using the
estimated HaMeR hand silhouette to ignore occluded edges. A bounded pose
correction may improve alignment. Corrections, edge residuals and refined-frame
flags are saved. The rigid residual describes the proposal before this image
correction; it is not the final pose's 3D accuracy.

An uncertain component can be initialized from visible case pixels, predicted
depth and a hand-orientation prior calibrated at the reviewed grasp frame 320.
It is marked with `requires_review` and a separate component ID. This assumes
an approximate hand–case orientation relationship for initialization; regrasping
can violate it. It is not an observed grasp constraint or a simulation attachment.
Omit `--hand-prior` to retain only the continuously connected trajectory.
The backward pass uses observed frame correspondences; it does not interpolate.

Run the numerical and geometric check separately:

```bash
.venv-g1/bin/python -m scripts.taa.check_object_track \
  --track runs/object-motion --scene runs/taa-candidate/scene.xml \
  --surface runs/object-anchor/case_surface.npy
```

## Limits

The case's hidden shape, scale and initial pose are estimated. Window composition
can accumulate drift. A small rigid residual means that predicted points agree
with a rigid transform; it is not millimetre-level ground-truth accuracy.
The next retargeting stage should use only reviewed poses and preserve gaps.
The current imposed-pose check has maximum penetration 1.25 mm and median final
table clearance 13.8 mm. Neither this check nor the preview establishes a
physically executed placement. Contact correction is still required.
