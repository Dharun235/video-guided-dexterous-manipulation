# Setup and usage

Run commands from the project root. Use a new output folder for every attempt.
The tested platform is Apple Silicon macOS. Install Git, uv and FFmpeg; the
setup scripts create Python 3.12 environments and fetch pinned robot assets.

## Robot setup

```bash
make setup
make test
```

This installs the simulation environment and fetches pinned G1/Inspire assets.
The current preview uses kinematics; physical execution is not implemented in
this workflow yet. The main README contains the ordered commands for each step.

## Hand extraction

```bash
uv venv --python 3.12 .venv-book-reconstruction
uv pip install --python .venv-book-reconstruction/bin/python \
  -r reconstruction/requirements-book-mac.lock.txt
.venv-book-reconstruction/bin/python -m reconstruction.video_hands \
  --video examples/source-recording/book5.mp4 --mano /path/to/MANO_RIGHT.pkl \
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

No unseen body pose is inferred. WHAM and 4D-Humans are body-reconstruction tools
and are not part of this close-up hand workflow.

## Scene and camera prerequisites

The Book5 scene builder fits a reviewed book plane and explicit case geometry:

```bash
.venv-book-reconstruction/bin/python -m reconstruction.book_space build \
  runs/book5-space --video examples/source-recording/book5.mp4
.venv-book-reconstruction/bin/python -m reconstruction.video_cameras \
  --output runs/book5-cameras --scene runs/book5-space --hands runs/full-hands
```

These utilities contain Book5-specific dimensions and annotations. Review their
outputs. The camera stage uses image features on the book plane; it does not
estimate object motion or construct a planned manipulation trajectory. A different video needs new
scene annotations and calibration. The existing reviewed camera estimates remain
an approximation and are not independent ground truth.

## Direct hand retargeting baseline

```bash
.venv-g1/bin/python -m humanoid.retarget_hand \
  --hands runs/full-hands --scene runs/book5-space \
  --cameras runs/book5-cameras/camera_poses.json --output runs/full-retarget
```

The output contains a composed `scene.xml`, named controls in `joint-map.json`,
`retarget.npz` and fit metrics in `report.json`. All input timestamps remain;
missing or unusable detections produce invalid rows with NaN actions. Source hand
geometry is registered against predicted image landmarks and reviewed cameras.
Fit residuals are against those estimates, not measured human ground truth.

This is kinematic fitting. It does not certify collision avoidance or a grasp.
Do not send the result straight to the physical replay interface: invalid frames,
trajectory smoothing, execution timing and contact refinement must be addressed
and reviewed first. The [interaction-aware adapter](retargeting.md) is pending.

## Inspect the kinematic fit

```bash
.venv-g1/bin/python -m scripts.preview_hand_retarget \
  --hands runs/full-hands --retarget runs/full-retarget --output runs/fit-preview
```

Open `runs/fit-preview/kinematic-fit.mp4`. The source and fitted robot are shown
at matching timestamps. The case remains static and robot poses are imposed;
this view is explicitly not a physical execution. Invalid hand estimates show a
warning while the previous robot pose remains displayed.
