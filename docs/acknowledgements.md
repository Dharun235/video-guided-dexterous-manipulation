# Acknowledgements

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
their authors. [Third-party notices](../THIRD_PARTY_NOTICES.md) record applicable
terms; licensed MANO files and model weights are not included.

Physical refinement optionally calls the unmodified [MuJoCo MPC Python predictive sampler](https://github.com/google-deepmind/mujoco_mpc/tree/main/python/mujoco_mpc/demos/predictive_sampling). The local adapter supplies the G1/Inspire model, reference, bounds and reward; it is not the CUDA SPIDER or Do as I Do implementation.
