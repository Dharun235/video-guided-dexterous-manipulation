# Video-Guided Dexterous Manipulation

**Goal:** train SmolVLA to control a Unitree G1 hand and reproduce the grasp,
lift, carry and placement shown in a recorded human demonstration.

The current simulation provides a physical reference of the task and verified
training data. The next step is to train the model to generate the robot's
commands from camera images, joint state and simulated fingertip forces.
Evaluation will cover motion similarity, grasp stability and placement accuracy.

## Recorded demonstration

![Recorded hand picking up the case and placing it on the table](examples/source-recording/preview.gif)

[Full recorded video](examples/source-recording/book5.mp4) · Audio removed; video frames unchanged.

## Current simulation reference

![Robot reference motion and simulated fingertip forces](examples/contact-demo/preview.gif)

The reference uses retargeted motion and contact-based grip control. The panels
show two simulation views and fingertip forces. This is the behavior the trained
policy should learn to reproduce; training on the new dataset is pending.

## What we use

| Component | Role |
|---|---|
| Recorded task video | Human motion and task reference |
| FastHaMeR / Core ML | Estimate hand poses from the video |
| Unitree G1 + Inspire right hand | Robot arm and five-finger hand |
| Inverse kinematics | Adapt the estimated motion to the robot's joints and reach |
| MuJoCo | Simulate the scene, collisions, friction and free-object interaction |
| Simulated fingertip forces | Adjust grip during reference-data collection |
| LeRobot / SmolVLA | Fine-tuning and learned robot control |

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

### Where the data is

Paths below are relative to the project folder. The full collection is local;
**a public download is not currently published**. GitHub includes the recorded
video and [replay example](examples/contact-demo/README.md).

| Item | Location |
|---|---|
| Full dataset | `runs/book5-contact-dataset-02/` |
| GPU-transfer archive (~698 MB) | `runs/book5-contact-dataset-02.tar.gz` |
| Split rows | `runs/book5-contact-dataset-02/data/{train,validation,test}.jsonl` |
| Images, states, forces and actions | Dataset's `train/`, `validation/`, `test/` episode folders |
| Feature definitions and normalization | Dataset's `feature-contract.json`, `statistics.json`, `statistics-tactile.json` |

[Dataset metadata and checksums](training/contact-dataset.json) identify the
verified package. After transferring/extracting it into `runs/`, verify it with:

```bash
.venv-g1/bin/python -m scripts.verify_contact_data runs/book5-contact-dataset-02
```

## Tasks and progress

- [x] Record and publish the human demonstration.
- [x] Estimate hand motion and reconstruct the task scene.
- [x] Retarget the motion to the G1/Inspire arm and hand.
- [x] Produce a physical reference with contact-based grip control.
- [x] Generate and verify the training data and grouped splits.
- [x] Publish a reproducible simulation example.
- [ ] Fine-tune SmolVLA on the prepared data using a GPU.
- [ ] Run the trained model in closed loop using current robot observations.
- [ ] Compare learned wrist/object motion with the recorded demonstration and
  simulation reference.
- [ ] Verify grasp, release and placement on held-out variations.

## Reproduce the current reference

Requires Python 3.12, Git, [uv](https://docs.astral.sh/uv/) and FFmpeg.
Tested on Apple Silicon macOS. This replay runs without a rented GPU or licensed
hand-model files.

```bash
git clone https://github.com/Dharun235/video-guided-dexterous-manipulation.git
cd video-guided-dexterous-manipulation
python3 -m scripts.setup_g1
python3 -m scripts.setup_book_g1
make test
.venv-g1/bin/python -m scripts.replay_example --output runs/example-replay
```

Open `runs/example-replay/execution.mp4`. The command applies saved motor targets
through physics and verifies robot motion, object motion and fingertip forces.
Use a new output folder each time. The full training collection and model
weights are separate local artifacts.

## Details

[Reconstruction and setup](docs/running.md) ·
[Data preparation](docs/contact-data-preparation.md) ·
[Data format](docs/data-and-evaluation.md) ·
[GPU training](training/CONTACT_TRAINING.md) ·
[Example files](examples/contact-demo/README.md) ·
[Licenses](THIRD_PARTY_NOTICES.md)

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
