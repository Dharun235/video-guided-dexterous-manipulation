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

## Dataset preview

![Training, validation and test episodes with both recorded camera views](examples/data-preview/splits.gif)

[Watch the video grid](examples/data-preview/splits.mp4) ·
[View a still image](examples/data-preview/splits.jpg)

Columns show training, validation and test data. Each column contains two
physical movements; each movement shows the front and inspection cameras.
The clips play at the recorded pace, with shorter clips holding their final
frame. These are contact-controlled reference demonstrations, not learned-policy
results.

| Split | Physical movements | Episodes | Samples |
|---|---:|---:|---:|
| Training | 20 | 60 | 15,720 |
| Validation | 6 | 18 | 4,464 |
| Test | 6 | 18 | 4,548 |

The collection contains 96 verified episodes from 32 physical movements, derived
from one human recording. Each movement has three visual variants. Starting
state, case position, appearance, lighting, cameras, mass and friction vary;
visual variants of the same movement stay in one split.

Each sample contains two camera images, 19 joint positions, 10 fingertip force
values and 13 executed motor targets. Source timestamps, object ground truth
and audit labels are excluded from policy observations.

The recorded human video and previews are included in GitHub; the full dataset
and model weights are not bundled. See [preview details](examples/data-preview/README.md),
[dataset specification](training/contact-dataset.json) and
[data format](docs/data-and-evaluation.md).

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

- Physical collection rejected 136 of 168 candidate attempts. The 32 accepted
  movements produced 96 episodes with visual variants. Keep failed attempts and
  apply the same task checks before accepting demonstration data.
- Accepted reference movements had a median placement error of 6.1 mm and a
  maximum of 26.1 mm. Check placement numerically, even when the motion looks right.
- The minimum raised thumb/opposing-finger hold was 4.15 s; the maximum sampled
  fingertip normal force was 39.95 N. Check sustained contact and force peaks
  alongside lift height; a completed lift alone does not describe grasp quality.
- The 96 episodes contain 32 physical movements, not 96 independent human demos.
  Keep each movement's visual variants together: 20 training, 6 validation and
  6 test groups.
- An earlier policy passed grasp/lift/release checks in all 7 test cases, but
  only 1 met the 3 cm placement tolerance. Placement accuracy remained a separate
  failure after grasping worked.
- An experimental arm correction reduced original-scene placement error from
  5.57 cm to 1.04 cm, while success on the earlier test suite fell from 1/7 to
  0/7. An improvement on the source scene did not establish generalization.

The policy results above use the earlier 36-episode collection. Training on the
current 96-episode collection has not started. See [measured results](docs/results.md).

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
