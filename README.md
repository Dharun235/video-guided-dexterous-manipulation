# Video-Guided Dexterous Manipulation

[![Pipeline tests](https://github.com/Dharun235/video-guided-dexterous-manipulation/actions/workflows/tests.yml/badge.svg)](https://github.com/Dharun235/video-guided-dexterous-manipulation/actions/workflows/tests.yml)

Drive a supported Unitree G1 with a native Inspire five-finger right hand from a
recorded tabletop demonstration. The Book5 example includes the initial hand
turn, grasp, transport, in-hand turn, placement, release and final wrist/finger gestures.
Execution uses MuJoCo motor controls and contact feedback. The fingers approach
loosely open and close around the case; all five contacts are checked during the
turn.

[![Full-video comparison](examples/video-to-g1/preview.gif)](examples/video-to-g1/comparison.mp4)

Clockwise from top left: recording, simulated phone view, contact view and G1.
Playback follows source time; the 12.5-second recording is retimed to 70.5 seconds
of simulation. The phone view hides the torso visually; collisions remain active.
[Full-quality comparison](examples/video-to-g1/comparison.mp4) ·
[Source recording](examples/source-recording/book5.mp4) ·
[Methods, results and design choices](docs/full-sequence.md)

| Nominal check | Result |
|---|---|
| Source timeline | All 748 frames retained |
| Initial palm orientation | Mean error 0.79° against extracted estimates |
| Physical case turn | 121.4°; all five fingers retain contact through the scored turn |
| Final placement | 4.9 mm from the adapted target; released and stationary |

## Recording

Recorded on an iPhone 14 held in the left hand while the right hand manipulated
the case. The full sample contains 748 frames over 12.5 seconds. The book measures
22 × 22 × 2 cm and the case 6 × 6 × 2 cm. The recording supplies hand gestures,
turn anchors, placement evidence and scene appearance.
[Recorded data and collection note](examples/source-recording/README.md)

## Run

Apple Silicon macOS, Python 3.12, Git, uv and FFmpeg. The saved-input pipeline
runs on CPU.

```bash
git clone https://github.com/Dharun235/video-guided-dexterous-manipulation.git
cd video-guided-dexterous-manipulation
make setup
make test
make video OUTPUT=runs/video-01
```

[Fresh-checkout verification](docs/verification.json) covers setup, all 50 tests,
physical execution, exact motor replay and video/GIF generation. CI also runs
the complete CPU sequence.

The command builds the measured scene, fits arm/hand/waist commands, executes
physics, scores the result and independently checks motor-command replay. Use a
new output folder for each attempt. To also render the comparison:

```bash
.venv-g1/bin/python -m video_g1.run_video --output runs/video-02 --render
```

## Approach and design

| Stage | Implementation |
|---|---|
| Human observations | FastHaMeR/MANO hand estimates; raw gaps retained |
| Scene and object inputs | TAA-derived reconstruction inputs and TrackCraft3R pose candidates; measured box collision geometry and book texture |
| Motion planning | Human-derived gestures, RH56-derived wrap prior, SciPy/MuJoCo arm–waist IK, reviewed phases and retiming |
| Physical execution | Position motors, robot-only bias compensation, shared-body grasp confirmation and finger-pressure feedback |
| Verification | Full-timeline pose/contact/placement metrics, numerical replay, sensitivity checks and CI |

The grasp/carry fingers use a calibrated geometric prior and simulated contact
feedback. Initial/final gestures and turn anchors come from the recording.
[Data-derived inputs and prescribed adaptations](docs/full-sequence.md#approach-and-data-use)
are listed separately in the method guide.

The scene uses measured rigid-box collision geometry and a fixed book. The pelvis
is supported, and native waist joints extend the arm's reach during the turn.
Reviewed phases and speed limits adapt the recording to the robot. Case mass
55 g and sliding friction 0.7 are simulation settings.

## Problems resolved

| Problem observed | Correction and result |
|---|---|
| Fingers curled before contact | Open-to-wrap prior reduced approach flexion from 1.53 to 0.26 rad; all five fingers load the case during the scored turn |
| Book contact stopped a finger closing | Shared-body confirmation and continuous pressure feedback keep the fingers engaged with the held object |
| Release brushed the case about 27 mm | Settle, release and retract sequence finishes within 4.9 mm of the placement target |

## Guides and credits

[Complete current pipeline](docs/full-sequence.md) ·
[New hand observations](docs/running.md) · [Scene inputs](docs/taa.md) ·
[Object tracking](docs/object-motion.md) · [Grasp evaluation](examples/rh56-evaluation/README.md)

Built on [RH56 controller](https://github.com/correlllab/rh56_controller),
[TAA](https://github.com/brains-bots-n-behavior/track-articulate-act),
[TrackCraft3R](https://github.com/cvlab-kaist/TrackCraft3r),
[FastHaMeR](https://github.com/VimalMollyn/fasterhamer),
[humanoid-dexart-retarget](https://github.com/gtziafas/humanoid-dexart-retarget)
and [MuJoCo](https://github.com/google-deepmind/mujoco), with Unitree G1/Inspire assets.
[Full credits](docs/acknowledgements.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)
