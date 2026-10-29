# Video-Guided Dexterous Manipulation

**Goal:** develop a learned policy that controls a Unitree G1 with a five-finger
Inspire right hand, matching the grasp, lift, carry and placement in a recorded
human demonstration.

The project currently includes video-based hand estimation, an approximate
MuJoCo scene, contact-controlled robot motion and verified demonstration data.
SmolVLA is a candidate policy; the final training code, method, compute and
evaluation protocol are still to be selected. Training on the new collection
has not started.

## Recorded demonstration

![Recorded hand picking up the case and placing it on the table](examples/source-recording/preview.gif)

[Full recorded video](examples/source-recording/book5.mp4) · Audio removed; video frames unchanged.

## Current simulation reference

![Robot reference motion and simulated fingertip forces](examples/contact-demo/preview.gif)

The reference uses retargeted motion and contact-based grip control. The panels
show two simulation views and fingertip forces. This is the behavior the trained
policy should learn to match; training on the new dataset is pending.

## What we use

| Component | Role |
|---|---|
| Recorded task video | Human motion and task reference |
| FastHaMeR / Core ML | Estimate hand poses from the video |
| Unitree G1 + Inspire right hand | Robot arm and five-finger hand |
| Inverse kinematics | Adapt the estimated motion to the robot's joints and reach |
| MuJoCo | Simulate the scene, collisions, friction and free-object interaction |
| Simulated fingertip forces | Adjust grip during reference-data collection |
| LeRobot / SmolVLA adapter | Existing policy interface; candidate for the next stage |

The scene is an approximate reconstruction, and the robot's pelvis is supported
and fixed. Human motion is adapted to the robot's hand and joint limits.

## Training data

The prepared collection contains **96 verified episodes / 24,732 samples**:
60 training, 18 validation and 18 test episodes. Each sample records two camera
views, joint angles, fingertip forces and executed motor commands.

Variations change the starting state, case position, appearance, lighting,
cameras, mass and friction. Related visual versions stay in the same split.
The collection uses contact-controlled reference actions as training targets;
the model receives observations, without source timestamps or object ground truth.

The recorded video and compact simulation example are included in GitHub.
The full collection and model weights are not bundled. See the
[dataset specification](training/contact-dataset.json) and
[data format](docs/data-and-evaluation.md) for package details.

## Tasks and progress

- [x] Record and publish the human demonstration.
- [x] Estimate hand motion and reconstruct the task scene.
- [x] Retarget the motion to the G1/Inspire arm and hand.
- [x] Produce a physical reference with contact-based grip control.
- [x] Generate and verify the training data and grouped splits.
- [x] Publish a runnable simulation example.
- [ ] Select the policy, training method and compute.
- [ ] Define the evaluation protocol and motion-similarity measures.
- [ ] Train the policy and run it with live simulation observations.
- [ ] Assess motion similarity, grasp stability and placement on unseen conditions.

## Requirements

| Task | Requirements |
|---|---|
| Run the included example | Python 3.12, Git, [uv](https://docs.astral.sh/uv/), FFmpeg and graphics access |
| Estimate hand motion | Apple Silicon macOS, separate reconstruction environment and licensed MANO archive |
| Generate a new collection | Reviewed scene/motion artifacts and collector dependencies described in the setup guide |
| Train and evaluate a policy | Full prepared data; method, software configuration and compute to be selected |

Apple Silicon macOS is the tested local platform. Setup downloads the pinned
robot assets. The included example needs neither a rented GPU nor MANO files.

## Installation

```bash
git clone https://github.com/Dharun235/video-guided-dexterous-manipulation.git
cd video-guided-dexterous-manipulation
python3 -m scripts.setup_g1
python3 -m scripts.setup_book_g1
make test
```

## Run the example

```bash
.venv-g1/bin/python -m scripts.replay_example --output runs/example-replay
```

Open `runs/example-replay/execution.mp4`. The command applies saved motor targets
through physics and verifies robot motion, object motion and fingertip forces.
Use a new output folder each time. The full training collection and model
weights are separate local artifacts.

## Workflow

1. **Estimate human motion.** Process the recording and review hand detections,
   scene dimensions and object annotations. See [setup and usage](docs/running.md).
2. **Build the robot reference.** Adapt the motion to the G1 arm and Inspire hand,
   then check the grasp and placement through MuJoCo contact physics.
3. **Prepare demonstration data.** Collect physical and visual variations, retain
   failed attempts for analysis, and verify grouped splits and recorded signals.
   See [data preparation](docs/contact-data-preparation.md).
4. **Train a policy — pending.** Select the learning method and compute before
   starting a run. The [existing SmolVLA interface](training/README.md) and
   [candidate commands](training/CONTACT_TRAINING.md) are implementation options.
5. **Evaluate learned control — pending.** Define how to compare motion with the
   human demonstration and test grasp, release and placement under new conditions.
   The included reference replay checks the simulator and recorded commands;
   it does not demonstrate a trained policy's success.

The setup guide lists commands and required artifacts for the implemented stages.
Full collection generation requires reviewed intermediate artifacts that are not
included in the public example.

[Example files](examples/contact-demo/README.md) ·
[Data format](docs/data-and-evaluation.md) ·
[Licenses](THIRD_PARTY_NOTICES.md)

## Problems and lessons

- **Depth and scale are uncertain in one video.** Review scene dimensions and
  object poses; treat the reconstruction as an approximation.
- **Hand orientation and robot placement affected the motion.** Check the right
  hand, camera alignment and robot position at the table's longer edge first.
- **Matching poses did not guarantee a stable grasp.** Verify lift, hold and
  release through contact physics, and use fingertip feedback to adjust grip.
- **Many physical variations failed the task checks.** Keep failed attempts for
  analysis and filter demonstration data with unchanged acceptance criteria.
- **Similar-looking episodes can leak across splits.** Keep visual variants of
  the same physical movement together.
- **Improving the original scene did not ensure wider success.** Check learned
  control on unseen conditions as well as against the source demonstration.

## Acknowledgements

- [MuJoCo](https://github.com/google-deepmind/mujoco) and
  [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) for the simulator and G1 model.
- [Unitree Robotics](https://github.com/unitreerobotics/xr_teleoperate) for the Inspire hand assets and
  [G1 wrist attachment reference](https://github.com/unitreerobotics/unitree_ros).
- [HaMeR](https://github.com/geopavlakos/hamer),
  [FastHaMeR](https://github.com/VimalMollyn/fasterhamer),
  [FastHands](https://github.com/VimalMollyn/Mediapipe-Hands-PyTorch-CoreML),
  [MANO](https://mano.is.tue.mpg.de/) and
  [Core ML Tools](https://github.com/apple/coremltools) for hand estimation and its Mac runtime.
- Hugging Face [LeRobot](https://github.com/huggingface/lerobot) and
  [SmolVLA](https://huggingface.co/lerobot/smolvla_base) for the policy and training framework.

Upstream code, models and assets retain their respective terms; see
[third-party notices](THIRD_PARTY_NOTICES.md) for licenses and pinned sources.
