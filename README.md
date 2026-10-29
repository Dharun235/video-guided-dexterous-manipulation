# Video-Guided Dexterous Manipulation

Convert a recorded human manipulation into a Unitree G1 arm and five-finger
Inspire right-hand reference, then verify execution through contact physics
in MuJoCo. The current work focuses on motion extraction and retargeting.
Policy training is outside the active scope.

## Source video

![Recorded human demonstration](examples/source-recording/preview.gif)

[Full video](examples/source-recording/book5.mp4) · One 748-frame recording;
audio removed, video frames unchanged.

## Workflow and status

| Stage | Output | Status |
|---|---|---|
| 1. Extract the complete hand motion | Per-frame landmarks, hand mesh estimates, wrist orientation, overlays and missing-frame mask | 748 frames processed; 683 detections; review required |
| 2. Retarget to G1/Inspire | Seven arm and six hand targets with fit residuals | 683 frames fitted; interaction-aware refinement pending |
| 3. Verify physical execution | Actuator-driven motion, free-object contact, grasp and placement measurements | Earlier adapted reference available; new full-video trajectory not physically verified |

The previous reference shortened the motion and used planned grasp/carry/release
segments. The new path retains the source timeline and marks missing estimates
explicitly. It does not claim that all source gestures already transfer correctly.

## Requirements

- Apple Silicon macOS for the tested FastHaMeR/Core ML hand-extraction path.
- Python 3.12, Git, [uv](https://docs.astral.sh/uv/) and FFmpeg.
- Licensed MANO assets for hand reconstruction; these are not included.
- Graphics access for MuJoCo rendering. The included reference replay needs no MANO files.

## Installation

```bash
git clone https://github.com/Dharun235/video-guided-dexterous-manipulation.git
cd video-guided-dexterous-manipulation
make setup
make test
```

For the hand-extraction environment:

```bash
uv venv --python 3.12 .venv-book-reconstruction
uv pip install --python .venv-book-reconstruction/bin/python \
  -r reconstruction/requirements-book-mac.lock.txt
```

## Step 1: extract hand motion

```bash
.venv-book-reconstruction/bin/python -m reconstruction.video_hands \
  --video examples/source-recording/book5.mp4 \
  --mano /path/to/MANO_RIGHT.pkl --output runs/full-hands
```

Open `runs/full-hands/hand-overlay.mp4` and inspect `coverage.json`.
`hand_track.npz` stores all frame timestamps, 21 hand landmarks, mesh vertices,
wrist orientation and a validity mask. Missing detections remain NaN.
Coordinates are camera estimates; scene calibration is required before robot IK.

![Full-video hand estimates](examples/hand-extraction/preview.gif)

The current run detects the right hand in 683/748 frames (91.3%). Missing
estimates remain explicit; this is detection coverage, not pose accuracy.
[Frame coverage](examples/hand-extraction/coverage.json)

## Step 2: retarget to G1/Inspire

The direct IK baseline uses the complete hand track, a reviewed scene and camera
poses. It fits seven right-arm controls and six Inspire controls without scripted
grasp phases. See [setup and usage](docs/running.md) for the command and prerequisites.

The requested [humanoid-dexart-retarget](https://github.com/gtziafas/humanoid-dexart-retarget)
provides an interaction-aware G1/Inspire solver for structured hand/object data.
Our single-right-hand adapter is pending. FRoM-W1/H-ACT and Mink are alternatives,
not installed stages of this project. See [retargeting integration](docs/retargeting.md).

[Watch the source beside the fitted robot](examples/hand-extraction/kinematic-fit.mp4).
This view imposes joint poses and leaves the object static; it is not physical execution.

The current baseline fits 683 detected frames. Mean residuals against the
estimated hand are 11.7 mm at the wrist and 10.6 mm at the fingertips. These
measure fit to inferred targets, not ground-truth accuracy or grasp success.

## Existing physics reference

![Earlier contact-controlled robot reference](examples/contact-demo/preview.gif)

```bash
.venv-g1/bin/python -m scripts.replay_example --output runs/example-replay
```

This applies saved motor commands and checks robot motion, object motion and
fingertip forces. It is an earlier adapted task reference, not the new full-video
retargeting result. The pelvis is fixed; scene geometry and touch are simulated.

## Checks before physical execution

- Review hand identity, orientation, missed frames and source-motion coverage.
- Check joint limits, reach, wrist/fingertip residuals and discontinuities.
- Execute with motor targets and physics steps; imposed joint poses only test kinematics.
- Measure sustained grasp, object slip, penetration, release and placement.
- Keep the object free; do not animate its pose to claim a successful grasp.

## Acknowledgements

[MuJoCo](https://github.com/google-deepmind/mujoco),
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) and
[Unitree Robotics](https://github.com/unitreerobotics/xr_teleoperate) provide the
simulator and robot assets. Hand estimation uses
[HaMeR](https://github.com/geopavlakos/hamer),
[FastHaMeR](https://github.com/VimalMollyn/fasterhamer),
[FastHands](https://github.com/VimalMollyn/Mediapipe-Hands-PyTorch-CoreML),
[MANO](https://mano.is.tue.mpg.de/) and
[Core ML Tools](https://github.com/apple/coremltools).
See [third-party notices](THIRD_PARTY_NOTICES.md) for asset and model terms.
