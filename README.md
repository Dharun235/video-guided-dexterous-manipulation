# Video-Guided Dexterous Manipulation

Convert a recorded tabletop hand demonstration into motion targets for a Unitree
G1 arm with a five-finger Inspire right hand. The pipeline estimates human hand
motion, reconstructs the scene and moving case, then fits robot joints.
The current outputs are kinematic estimates. A motor-driven grasp has not yet
been demonstrated.

## Current status

| Stage | Method | Current output |
|---|---|---|
| Human hand | FastHaMeR, Core ML and MANO | 683/748 frames detected |
| Scene and camera | Book-plane calibration, SAM3, DA3 and SAM 3D Objects through TAA | Textured MuJoCo scene with collision geometry and estimated scale |
| Moving case | TAA/TrackCraft3R, rigid fitting and visible-image refinement | 740/748 pose candidates; 310 need orientation review; eight remain missing |
| Robot joints | humanoid-dexart-retarget, PyRoki and JAX | 227 fitted frames with a static case; moving-case targets are not integrated |
| Physical execution | MuJoCo actuator and contact control | Pending |

Pose coverage measures available estimates, not accuracy. The object preview
imposes case poses; the robot preview imposes joint poses. Neither demonstrates
a physical grasp. Training and evaluation methods have not been selected.

## Task checklist

- [x] Include the recorded demonstration and source checksums.
- [x] Extract right-hand landmarks, mesh and wrist orientation across the video.
- [x] Estimate camera motion and scene scale from the book plane.
- [x] Segment the book and case across all 748 frames.
- [x] Build a textured MuJoCo scene with case collision geometry.
- [x] Estimate moving-case poses and preserve missing/uncertain frames.
- [x] Fit the G1 arm and Inspire right hand using the upstream retargeting solver.
- [x] Publish source comparisons, pose data and numerical checks.
- [ ] Review the 310 uncertain case poses and eight missing poses.
- [ ] Correct final case placement, currently estimated 13.8 mm above the table.
- [ ] Integrate reviewed moving-case poses into robot retargeting.
- [ ] Reduce joint discontinuities and verify reach, limits and collisions.
- [ ] Execute with motor controls and a free object; verify grasp, slip and release.
- [ ] Export verified simulated camera images, joint states, actions and contacts.
- [ ] Select training and evaluation methods after physical execution is verified.

## Videos and results

The recording contains 748 frames. Audio is removed; video frames are unchanged.
Each comparison shows a different stage, rather than a complete robot execution.

| Recorded demonstration | Estimated human hand |
|---|---|
| [![Recorded demonstration](examples/source-recording/preview.gif)](examples/source-recording/book5.mp4) | ![Estimated hand mesh](examples/hand-extraction/preview.gif) |
| [Watch the source video](examples/source-recording/book5.mp4) | [Detection coverage and missing frames](examples/hand-extraction/coverage.json) |

| Scene and segmentation | Estimated case motion in MuJoCo |
|---|---|
| [![Scene reconstruction](examples/taa/scene-comparison.jpg)](docs/taa.md) | [![Case motion comparison](examples/object-motion/review-grid.jpg)](examples/object-motion/object-motion.mp4) |
| [Watch full-video masks](examples/taa/full-video-masks.mp4) | [Watch source beside MuJoCo](examples/object-motion/object-motion.mp4) |

[Watch the G1/Inspire joint-fit comparison](examples/dexart-retarget/kinematic-fit.mp4).
This robot preview has a static case and imposed joint poses. The case-motion
preview separately imposes object poses. These stages are not yet integrated
into a motor-driven manipulation.

## Inspect the saved result

After cloning the repository, run from its root:

```bash
make setup
make review-object
```

This checks the included pose arrays and scene geometry without downloading GPU
models. [Pose data, scene and validation](examples/object-motion/README.md) are
included. Full processing commands follow below.

## Requirements and setup

Apple Silicon macOS, Python 3.12, Git, [uv](https://docs.astral.sh/uv/), FFmpeg
and graphics access. Hand extraction also requires your licensed MANO model.
Robot assets and inference models download during setup or first use.
The TAA object-tracking stage also uses a Linux NVIDIA GPU; the tested host has
48 GB VRAM. Its outputs can be reviewed on the Mac.

```bash
git clone https://github.com/Dharun235/video-guided-dexterous-manipulation.git
cd video-guided-dexterous-manipulation
make setup
python3 -m scripts.setup_dexart
uv venv --python 3.12 .venv-book-reconstruction
uv pip install --python .venv-book-reconstruction/bin/python \
  -r reconstruction/requirements-book-mac.lock.txt
make test
```

Run the following commands from the project root. Choose new output folders
for later attempts. Full outputs remain local; selected previews are included.

## 1. Extract hand motion

```bash
.venv-book-reconstruction/bin/python -m reconstruction.video_hands \
  --video examples/source-recording/book5.mp4 --mano /path/to/MANO_RIGHT.pkl \
  --output runs/full-hands
```

FastHaMeR/Core ML estimates hand landmarks, mesh and wrist orientation for each
frame. Inspect `hand-overlay.mp4` and `coverage.json`. Missing detections remain
invalid rather than being filled with invented poses.

Current result: 683/748 frames detected (91.3%). This measures detection
coverage, not pose accuracy. [Missing intervals](examples/hand-extraction/coverage.json)

## 2. Build the scene and estimate camera poses

```bash
.venv-book-reconstruction/bin/python -m reconstruction.book_space build \
  runs/book5-space --video examples/source-recording/book5.mp4
.venv-book-reconstruction/bin/python -m reconstruction.video_cameras \
  --hands runs/full-hands --scene runs/book5-space --output runs/book5-cameras
```

The reviewed book plane supplies scale and camera geometry. SIFT matching and
RANSAC estimate camera movement across frames. This step produces approximate static geometry and camera poses. The separate
GPU stage below estimates the moving case trajectory.

The GPU reconstruction branch uses [Track, Articulate, Act](https://github.com/brains-bots-n-behavior/track-articulate-act),
SAM3, Depth Anything 3 and SAM 3D Objects. SAM3 tracks the book and case through
all 748 frames. DA3 now supplies depth for every frame. A separate case mesh
candidate has gravity and convex collision geometry in MuJoCo. TrackCraft3R
estimates moving-case poses in overlapping windows, with visibility and fit checks.
[Commands, previews and measurements](docs/taa.md)
· [Object-motion pipeline](docs/object-motion.md)

[Recorded case motion beside the MuJoCo candidate](examples/object-motion/object-motion.mp4)

Final candidate: 740/748 frames, with 310 estimates flagged for orientation review
and eight missing poses. Coverage is not pose accuracy. Placement still needs
contact correction before these targets can guide verified robot execution.

## 3. Retarget with humanoid-dexart-retarget

```bash
.venv-dexart/bin/python -m retargeting.video \
  --hands runs/full-hands --scene runs/book5-space \
  --cameras runs/book5-cameras/camera_poses.json --output runs/full-retarget \
  --stride 3 --window 24
```

The [upstream solver](https://github.com/gtziafas/humanoid-dexart-retarget)
uses a Laplacian interaction objective, PyRoki and JAX. Our adapter supplies the
observed right hand, static book anchors and the matching G1/Inspire geometry.
The pelvis and unobserved joints stay supported/fixed. The moving case's
trajectory is not yet supplied.

The first trial solved 227 observed frames at about 20 Hz. Mean wrist/fingertip
residuals are 31.7/24.9 mm, compared with 11.7/10.6 mm for the earlier SciPy
baseline on the same frames. Residuals are against estimated targets. The new
trial has abrupt target changes and is not ready for physical execution.
[Adapter, measurements and limitations](docs/retargeting.md)

## 4. Review the motion

```bash
.venv-g1/bin/python -m scripts.preview_hand_retarget \
  --hands runs/full-hands --retarget runs/full-retarget --output runs/fit-preview
```

[Source beside the fitted robot](examples/dexart-retarget/kinematic-fit.mp4)

The preview imposes joint poses and leaves the object static. Missing detections
show a warning and the previous robot pose. Check orientation, finger shape,
reach, discontinuities and whether the full demonstrated motion is represented.

## 5. Verify physical execution — pending

Resolve detection gaps, refine hand–object interaction and execute through
MuJoCo motor controls. Check collision, sustained grasp, slip, release and
placement with the object free. The current preview does not establish these.
The pelvis is supported; walking and balance are outside scope.

## Project files

| Folder | Purpose |
|---|---|
| `reconstruction/` | Hand extraction, scene/camera fitting and object-pose estimates |
| `humanoid/` | G1/Inspire model composition, baseline fitting and geometry checks |
| `retargeting/` | Pinned upstream solver adapter and model conversion |
| `scripts/` | Robot setup and motion preview |
| `examples/` | Source recording and current results |
| `runs/` | Local outputs; excluded from Git |

[Output formats and assumptions](docs/running.md) ·
[Upstream retargeter and adapter](docs/retargeting.md)

## Problems and lessons

| Observation | Consequence |
|---|---|
| 65/748 frames lack a hand detection | Keep invalid rows; do not invent unseen finger motion |
| Independent depth batches broke rigid tracking at frame 288 | Re-estimate depth jointly within overlapping tracking windows |
| 310 object poses require relocalization review | Preserve component IDs and review flags in downstream targets |
| Upstream wrist/tip residuals are 31.7/24.9 mm | The current retargeting trial needs refinement before control |
| Five object poses penetrate by more than 1 mm; final clearance is 13.8 mm | Image agreement alone does not establish a physically valid placement |

## Acknowledgements

| Role | Work used |
|---|---|
| Simulator and robot assets | [MuJoCo](https://github.com/google-deepmind/mujoco), [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), [Unitree Robotics](https://github.com/unitreerobotics/xr_teleoperate) |
| Human hand estimation | [HaMeR](https://github.com/geopavlakos/hamer), [FastHaMeR](https://github.com/VimalMollyn/fasterhamer), [FastHands](https://github.com/VimalMollyn/Mediapipe-Hands-PyTorch-CoreML), [MANO](https://mano.is.tue.mpg.de/), [Core ML Tools](https://github.com/apple/coremltools) |
| Scene reconstruction | [Track, Articulate, Act](https://github.com/brains-bots-n-behavior/track-articulate-act), [SAM3](https://github.com/facebookresearch/sam3), [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3), [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects), [MoGe](https://github.com/microsoft/MoGe) |
| Object motion | [TrackCraft3R](https://github.com/cvlab-kaist/TrackCraft3r) and its Wan2.1 base model |
| Robot retargeting | [humanoid-dexart-retarget](https://github.com/gtziafas/humanoid-dexart-retarget), [PyRoki](https://github.com/chungmin99/pyroki), [jaxls](https://github.com/brentyi/jaxls) |

This repository adds the recording-specific calibration, input adapters,
visibility and rigid-fit checks, flagged pose relocalization, MuJoCo scene
integration and review tools. Upstream models and solvers remain credited to
their authors. [Third-party notices](THIRD_PARTY_NOTICES.md) record applicable
terms; licensed MANO files and model weights are not included.
