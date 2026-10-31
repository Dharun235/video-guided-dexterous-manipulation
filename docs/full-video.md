# Full-video transfer

This experimental path keeps every source frame, including free-hand gestures,
rotation and release. It generates robot control proposals and evaluates them in
MuJoCo. It does not train a policy. Full physical imitation is not solved.

## Inputs and scale

Book5 contains 748 frames at approximately 60 Hz. The right hand is observed in
683 frames. Missing observations remain NaN; interpolated motor commands are
labelled separately. The iPhone 14 was handheld in the left hand.

The user supplied a square 22 × 22 × 2 cm book and a 6 × 6 × 2 cm case. The initial comparison used unmeasured MANO hand scale. The user subsequently
reported approximately 23 cm from wrist crease to middle fingertip. That
measurement is an additional uncertain constraint, not a guarantee of depth accuracy. Case mass (55 g), friction
(0.7), table extent and camera distortion remain assumptions. The book is fixed;
the case is a free rigid box. The robot pelvis is supported.

## Steps

Run from the project root after [setup](running.md). The small saved hand/camera
inputs and initial case mask are included under `examples/full-video/`; steps
3–7 run without neural inference or licensed model files. For new observations,
follow [hand extraction](running.md) and [TAA segmentation](taa.md). Use a fresh
output directory for every run.

1. Extract all right-hand poses with FastHaMeR/MANO. Review the mesh overlay.
2. Track the book corners with SIFT/RANSAC using `reconstruction.video_cameras`.
   Its legacy scene supplies image annotations; the next step replaces metric
   camera calibration using the measured dimensions.
3. Fit one focal length and per-frame book-relative cameras:

```bash
.venv-g1/bin/python -m video_g1.calibration \
  --config configs/book5.json \
  --camera-observations examples/full-video/camera-observations.json \
  --output runs/full-video-calibration
```

4. Build measured-size collision geometry and fit the initial case silhouette:

```bash
.venv-g1/bin/python -m video_g1.scene \
  --config configs/book5.json --cameras runs/full-video-calibration \
  --appearance examples/object-motion/scene \
  --case-masks examples/full-video/case-masks --output runs/full-video-scene
```

The book texture comes from the recording. The box case and assumed table are
explicit approximations. This builder does not generate a new neural scene mesh.

5. Fit all observed hand poses jointly to the native G1/Inspire model:

```bash
.venv-g1/bin/python -m video_g1.reference \
  --hands examples/full-video/hand-observations --cameras runs/full-video-calibration \
  --scene runs/full-video-scene --calibration runs/full-video-scene \
  --seed examples/full-video/initialization \
  --output runs/full-video-reference --iterations 80
```

The solver uses bounded SciPy least squares, analytic MuJoCo landmark derivatives,
linear Inspire joint coupling, palm orientation and temporal regularization.
A sequential initialization reduces arm-configuration switches. The 13 controls
are seven arm/wrist and six hand channels. Every source frame remains in the
reference. Execution is slowed where necessary to respect configured command
speed limits; these are design limits, not measured hardware guarantees.

`--seed PATH` optionally loads an earlier `retarget.npz` initialization.
`--anchor-frame 240` tests a constant task-space translation using the reviewed
thumb/index midpoint at the grasp. This changes the robot proposal, preserves raw
camera-registered observations separately and is not a metric hand calibration.
The tested anchor did not solve the grasp and worsened wrist fitting.

6. Execute motor targets with a free case:

```bash
.venv-g1/bin/python -m video_g1.execute \
  --reference runs/full-video-reference --output runs/full-video-execution
```

Only reset initializes joint positions. During execution, the controller applies
position targets and robot-only bias compensation. No object pose playback,
attachments or object forces are used. This baseline has no tactile correction;
the separate successful translation adaptation uses contact feedback.

7. Inspect the entire timeline:

```bash
.venv-g1/bin/python -m video_g1.compare \
  --reference runs/full-video-reference --rollout runs/full-video-execution \
  --cameras runs/full-video-calibration --source examples/source-recording/book5.mp4 \
  --output runs/full-video-comparison
```

The preview samples every fourth frame across the complete timeline; numeric
outputs retain all 748 frames. Playback follows source time and labels execution
time. Phone-view panels hide the torso visually to expose the hand; collisions
remain active and the fourth panel shows the complete robot. This estimated
moving camera is not an onboard robot camera.

## Outputs

| File | Contents |
|---|---|
| `cameras.npz` | Source indices/times, shared intrinsics and world-to-camera extrinsics |
| `reference.npz` | All indices/times, 13 commands, observed flags, command provenance and fitted landmarks |
| `joint-map.json` | Ordered command and 19 joint-state names |
| `rollout.npz` | Physical joint states, velocities, commands, full positions and five normal/shear force pairs |
| `runtime.json` | Input/source hashes, software versions and host architecture |
| `comparison.mp4` | Source, fitted hand, executed hand and full robot |

Lengths are metres, joint angles radians, times seconds and forces newtons.
Contact forces sum external finger contacts; they are simulated measurements.
The exported commands are controller targets, not recorded human actions.

## Measured results

The measured-book camera fit has a mean corner residual of 1.28 pixels. A low
image residual does not establish accurate hand depth or camera intrinsics.

| Fit | Wrist error | Fingertip error | Largest command step | Execution time |
|---|---:|---:|---:|---:|
| Independent initialization, 20 global evaluations | 10.5 mm | 14.2 mm | 4.42 rad | 101.7 s |
| Sequential initialization, converged in 72 evaluations | 7.3 mm | 11.2 mm | 0.34 rad | 21.9 s |
| Grasp translation anchor, stopped at 80 evaluations | 39.0 mm | 15.9 mm | 0.87 rad | 22.7 s |

These are kinematic errors against estimated human landmarks. The unanchored
physical replay has 42.0 mm wrist and 83.3 mm fingertip errors. It does not grasp
the case. The anchor also fails. Palm orientation and collisions remain difficult;
Inspire cannot reproduce all human finger splay independently.

The GPU bundle converts existing object candidates through their old image
projection into the measured geometry/camera convention. This is not new tracking.
Source gaps, components and review flags remain separate from interpolated control
references. Any support correction is recorded. GPU refinement is experimental;
see its saved comparison before accepting a result.

## Compute and checks

The above calibration, fitting and physical replay ran on Apple Silicon arm64
macOS using Python 3.12 and `humanoid/requirements.lock.txt` (MuJoCo 3.8.1).
The selected full fit took 79.9 s and replay approximately 5 s. These timings exclude
hand extraction, downloads and rendering. Exact chip information was not recorded.

GPU tests use one RTX 4090, 24 GB VRAM, Python 3.12, PyTorch 2.8.0+cu128,
MuJoCo 3.7.0, MuJoCo Warp 3.7.0.1 and Warp 1.12.1 in a separate Linux environment.
SPIDER stays pinned and unmodified; its license remains applicable.

`make test` includes synthetic metric-camera recovery, finite-difference palm and
coupled-joint derivatives, command provenance and speed-limit checks. CI also
checks formatting/lint for this package. Tests establish numerical contracts;
they do not certify perception accuracy or physical grasp success.

## Remaining work

- Resolve hand/object depth consistently without relying on a translation anchor.
- Correct uncertain object orientation through the flip and occlusion.
- Include collision and contact constraints in full-trajectory retargeting.
- Demonstrate maintained opposition, lift, flip and release in physical execution.
- Validate timing and object/hand errors over the entire video before dataset use.

## Full-timeline GPU comparison

The RTX 4090 test used 64 candidate worlds, one sampling iteration per update,
a 0.32 s horizon and a 3 rad/s command slew limit. It ran for 22.70 s of simulated
time against the 22.708 s anchored reference, ending 8.5 ms before its last timestamp.
Compilation and optimization took 357 s; reference replay took 20.2 s.

| Same converted object reference | Mean position error | Maximum position error |
|---|---:|---:|
| Unoptimized motor-reference replay | 101.8 mm | 257.2 mm |
| GPU optimization | 101.8 mm | 257.2 mm |

Neither run grasps or lifts the case. Maximum upward movement is 0.75 mm, not a
successful lift. One candidate-reward warning reported a non-finite candidate;
the executed plant remained finite. Increasing compute is not justified until
hand/object alignment and contact constraints are corrected.

All 1,135 saved position-state vectors matched exactly in an independent CPU
MuJoCo 3.7.0 replay of the 11,350 saved commands. This checks physical command
execution, not reference accuracy or task success.
[Reports, trajectory and replay check](../examples/full-video/gpu/)

The full-video SciPy solver is this project's implementation. The separate
[humanoid-dexart-retarget/PyRoki adapter](retargeting.md) remains available; it did
not generate these new full-timeline results. SPIDER supplies the unmodified
sampling optimizer and Warp backend for the GPU experiment.


## Measured-hand and contact-prior checks

`video_g1.reference --hand-length-m .23` estimates one hand scale from the median
wrist-to-middle-tip length in extended poses. In this recording 52 frames meet
the straightness criterion; the model length is 171.8 mm and the scale is 1.3385.
The 23 cm check worsens the fit: wrist/tip errors are 86.1/42.0 mm, and physical
errors are 122.7/100.1 mm. It does not solve the task. These results expose an
inconsistency between hand reconstruction, metric assumptions and embodiment.

An optional diagnostic refines each rigid hand registration with image evidence,
soft support-surface constraints and image-inferred contacts. It samples bone
interiors too; a fingertip pinch at frame 240 was an unsupported assumption.
The source shows the hand resting over the case there.

```bash
.venv-g1/bin/python -m video_g1.refine_hand \
  --hands examples/full-video/hand-observations \
  --cameras runs/full-video-calibration \
  --objects examples/full-video/gpu/object-candidates.npz \
  --case-masks runs/taa-full-01/masks/case \
  --config configs/book5.json --output runs/full-video-hand-refinement
```

This optional step needs the full TAA mask sequence. It produces 675 refined
registrations; frames with missing object candidates are unavailable for
refinement. Raw hand observations remain separate. Mean image displacement is
7.94 px; all refinements require review. The included saved proposal is `examples/full-video/refinement/`. Use
`--hand-length-m .23 --refined-hands examples/full-video/refinement` when fitting
an explicit refinement proposal. Image proximity does not establish contact,
and the table footprint and rigid-box geometry remain assumptions.

### GPU commands

First generate a separate anchored reference with the reference command above,
adding `--anchor-frame 240` and choosing a new output directory. This is a failed
experimental reference; it must not replace the unanchored diagnostic silently.
Package it locally:

```bash
.venv-g1/bin/python -m video_g1.gpu_bundle \
  --reference runs/full-video-anchor --cameras runs/full-video-calibration \
  --config configs/book5.json --objects examples/object-motion \
  --old-cameras examples/full-video/camera-observations.json \
  --old-calibration examples/physics/calibration/reconstruction.json \
  --output runs/full-video-gpu-input
```

Transfer that directory and `scripts/gpu/` to a Linux CUDA host. In the separate
GPU environment, use new output directories:

```bash
bash scripts/gpu/setup.sh /workspace/physics
/workspace/physics/env/bin/python scripts/gpu/optimize.py \
  --spider /workspace/physics/spider --input /workspace/full-video-gpu-input \
  --output /workspace/full-video-replay --seconds 22.70 --replay-only --slew 3
/workspace/physics/env/bin/python scripts/gpu/optimize.py \
  --spider /workspace/physics/spider --input /workspace/full-video-gpu-input \
  --output /workspace/full-video-optimized --seconds 22.70 \
  --samples 64 --iterations 1 --horizon .32 --slew 3
/workspace/physics/env/bin/python scripts/gpu/check_commands.py \
  --input /workspace/full-video-gpu-input \
  --trajectory /workspace/full-video-optimized --output /workspace/command-replay.json
```

The commands record a bounded failure comparison, not a successful-grasp recipe.


The refined full-trajectory test preserves all 748 samples and uses 675 valid
refinements. Its physical wrist/tip errors are 28.8/35.4 mm, versus 42.0/83.3 mm
for the original unmeasured baseline. These errors use different pose estimates;
they are not an independent accuracy benchmark. Execution takes 25.9 s. The case
is not lifted; large external contact forces remain. The latest comparison and
arrays use this refinement, while `baseline/` preserves the earlier fit reports.


A measured-box refit directly against the full mask sequence retained only
204/748 candidates (frames 6–242), with median visible-edge error 0.79 px. It
failed through the manipulation and showed depth drift even before occlusion.
It is rejected as a full object trajectory; it does not replace the existing
candidate archive. This reinforces why low pixel residual and coverage alone
cannot certify a physical pose.

```bash
.venv-g1/bin/python -m video_g1.objects \
  --hands examples/full-video/hand-observations \
  --cameras runs/full-video-calibration \
  --prior examples/full-video/gpu/object-candidates.npz \
  --case-masks runs/taa-full-01/masks/case --scene runs/full-video-scene \
  --config configs/book5.json --output runs/full-video-object-mask-check
```
