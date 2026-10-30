# Video-Guided Dexterous Manipulation

Transfer a recorded tabletop hand demonstration to a Unitree G1 with an Inspire
five-finger right hand. The pipeline estimates human and object motion, builds a
MuJoCo scene and fits robot joint targets. The current physics experiment lifts, transfers and releases the case using
contact feedback. Full-video imitation remains incomplete.

## Demo

| Recorded demonstration | Physical transfer in MuJoCo |
|---|---|
| [![Source](examples/source-recording/preview.gif)](examples/source-recording/book5.mp4) | [![Physical transfer](examples/physics/preview.gif)](examples/physics/physics.mp4) |

[Source video](examples/source-recording/book5.mp4) ·
[Hand extraction](examples/hand-extraction/README.md) ·
[Case motion video](examples/object-motion/object-motion.mp4) ·
[Physical transfer](examples/physics/physics.mp4)

The physical transfer uses motor controls with a free case. It adapts the recorded
translation task and holds the initial orientation; the ambiguous flip is omitted.
[Results, data and limitations](examples/physics/README.md)

## Pipeline

| Stage | Method | Output |
|---|---|---|
| Hand motion | FastHaMeR / MANO on Core ML | Hand mesh, landmarks and wrist orientation; 683/748 frames detected |
| Scene | TAA, SAM3, DA3 and SAM 3D Objects | Camera/depth estimates and textured MuJoCo geometry with collisions |
| Object motion | TrackCraft3R, rigid fitting and image refinement | 740/748 pose candidates; 310 flagged for review |
| Robot targets | Contact-aware bounded IK; upstream PyRoki adapter available | Grasp and smooth arm targets derived from recorded hand/object estimates |
| Physics | MuJoCo contact feedback; optional upstream MPC | Free-object lift, transport, release and synchronized observations |

Coverage is not pose accuracy. Estimated scale and hidden geometry remain
uncertain. The raw final case estimate is 13.8 mm above the table; control uses an explicit
support-plane correction.
[Measurements and validation](examples/object-motion/README.md)

## Setup

Mac: Apple Silicon, Python 3.12, Git, uv and FFmpeg. Full hand extraction needs
licensed MANO files. GPU reconstruction was tested on Linux with 48 GB NVIDIA
VRAM; saved outputs can be checked on the Mac.

```bash
git clone https://github.com/Dharun235/video-guided-dexterous-manipulation.git
cd video-guided-dexterous-manipulation
make setup
make test
make review-object
make transfer
```

`review-object` validates the included pose data and scene without GPU inference.
For full processing, follow these guides in order:

1. [Hand extraction and camera calibration](docs/running.md)
2. [Scene reconstruction](docs/taa.md)
3. [Object-motion estimation](docs/object-motion.md)
4. [G1/Inspire retargeting](docs/retargeting.md)
5. [Physical transfer and optional automatic refinement](docs/physics.md)

The supplied calibration is specific to Book5. A new recording needs reviewed
scene dimensions, annotations and camera alignment. Generated runs, models and
licensed assets are excluded from Git.

## Tasks

- [x] Extract the recorded right hand and preserve missing detections.
- [x] Reconstruct scene geometry and estimate moving-case poses.
- [x] Integrate the G1/Inspire retargeter and publish stage comparisons.
- [ ] Review uncertain object poses, gaps and final placement.
- [x] Integrate unflagged moving-case anchors and build contact-aware targets.
- [x] Run and measure actuator-driven transfer with a free object.
- [x] Export synchronized simulated cameras, states, actions and contact summaries.
- [ ] Improve robustness and full-motion fidelity before selecting training methods.

## Lessons

- Independent depth batches disrupted tracking at frame 288; joint depth estimation
  within overlapping windows improved continuity.
- Uncertain object orientation prevents reliable imitation of the recorded flip.
- The nominal physical transfer passes; lower friction and disabled force feedback fail.
  These are small deterministic checks, not a generalization benchmark.

## Acknowledgements

Built on [TAA](https://github.com/brains-bots-n-behavior/track-articulate-act),
[TrackCraft3R](https://github.com/cvlab-kaist/TrackCraft3r),
[FastHaMeR](https://github.com/VimalMollyn/fasterhamer),
[humanoid-dexart-retarget](https://github.com/gtziafas/humanoid-dexart-retarget)
and [MuJoCo](https://github.com/google-deepmind/mujoco), with Unitree G1/Inspire assets.
[Full credits](docs/acknowledgements.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)
