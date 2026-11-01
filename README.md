# Video-Guided Dexterous Manipulation

[![Pipeline tests](https://github.com/Dharun235/video-guided-dexterous-manipulation/actions/workflows/tests.yml/badge.svg)](https://github.com/Dharun235/video-guided-dexterous-manipulation/actions/workflows/tests.yml)

Convert a recorded tabletop demonstration into a supported Unitree G1 motion
with an Inspire five-finger right hand in MuJoCo. The example follows the full
recording: hand gestures, grasp, carry, turn, placement, release and final gestures.
Position motors and contact feedback move the robot and the free object.

[![Recording and robot comparison](examples/book5/demo.gif)](examples/book5/demo.mp4)

Clockwise from top left: recording, simulated phone view, contact view and robot.
The comparison follows source time: the 12.5-second recording is retimed to
70.5 seconds of physical execution.
[Watch video](examples/book5/demo.mp4) · [Source recording](data/book5/video.mp4)

## Run

Requirements: Python 3.12, Git and [uv](https://docs.astral.sh/uv/).
FFmpeg is needed for video/GIF output. The saved-input pipeline runs on CPU;
Apple Silicon macOS and Ubuntu CI are tested.

```bash
git clone https://github.com/Dharun235/video-guided-dexterous-manipulation.git
cd video-guided-dexterous-manipulation
make setup
make test
make video OUTPUT=runs/video-01
```

Setup installs pinned dependencies and downloads the G1/Inspire models. The run
uses the supplied hand, camera and object observations. Each output folder must
be new. To include the comparison video and GIF:

```bash
.venv-g1/bin/python -m video_g1.run_video --output runs/video-02 --render
```

Results are saved under the selected folder: `scene/`, `grasp/`, `reference/`,
`rollout/`, `replay-check.json` and, when requested, `preview/`.

## Pipeline

| Stage | Method |
|---|---|
| Observe | FastHaMeR/MANO hand estimates; TAA segmentation/reconstruction and TrackCraft3R object-pose candidates |
| Build | Book-corner camera calibration, measured box collisions and photographed book texture |
| Retarget | Human-derived gestures and turn anchors, RH56-derived wrap prior, SciPy/MuJoCo arm–waist inverse kinematics |
| Execute | Native coupled Inspire joints, position motors, shared-body grasp confirmation and per-finger pressure feedback |
| Verify | Pose/contact/placement checks and independent replay of the complete motor stream |

The recording determines gestures, turn anchors, placement evidence and scene
appearance. Reviewed phases, reachable workspace, grasp geometry and slower timing
adapt it to the robot. The pelvis is supported; the case moves through simulated
contact forces. Book5 is a calibrated example, so another recording requires
reviewing its scene, hand registration and motion phases.

| Nominal result | Measurement |
|---|---|
| Source coverage | All 748 timeline frames retained |
| Case turn and lift | 121.4° and 103.4 mm |
| Five-finger contact during scored turn | All five above 0.05 N in every sample |
| Final placement | 4.9 mm from adapted target; released and stationary |
| Motor replay | Identical saved position states |

[Methods and measurements](docs/methods.md) ·
[Preparing video inputs](docs/preparation.md) ·
[Machine-readable result](examples/book5/report.json)

## Data and layout

The recording was captured on an iPhone 14 held in the left hand while the right
hand manipulated the case. Book: 22 × 22 × 2 cm; case: 6 × 6 × 2 cm.
[Collection note and input files](data/book5/README.md).

```text
data/book5/      Recording and reviewed inputs
examples/book5/  Comparison, planned commands, physical rollout and results
video_g1/       Planning, contact control, execution and rendering
humanoid/       G1/Inspire model assembly and joint interface
reconstruction/ Hand, camera and object observation tools
scripts/        Model setup and GPU input preparation
docs/           Methods and input preparation
```

## Development

Run commands from the project root. Tests live beside their modules. Keep
observations separate from planned commands and physical state; retain missing
values and review flags. Use new run folders and compare numerical rollouts when
changing motion or control. CI checks formatting, unit tests, full CPU execution
and motor replay.

```bash
uvx --from ruff==0.17.3 ruff check video_g1 humanoid reconstruction scripts
uvx --from ruff==0.17.3 ruff format --check video_g1 humanoid reconstruction scripts
make test
```

## Acknowledgements

Built with [MuJoCo](https://github.com/google-deepmind/mujoco),
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie),
[Unitree Inspire assets](https://github.com/unitreerobotics/xr_teleoperate),
[RH56 controller](https://github.com/correlllab/rh56_controller),
[FastHaMeR](https://github.com/VimalMollyn/fasterhamer),
[TAA](https://github.com/brains-bots-n-behavior/track-articulate-act)
and [TrackCraft3R](https://github.com/cvlab-kaist/TrackCraft3r).
Model sources, revisions and asset terms are listed in
[third-party notices](THIRD_PARTY_NOTICES.md). Project code uses the [MIT license](LICENSE).
