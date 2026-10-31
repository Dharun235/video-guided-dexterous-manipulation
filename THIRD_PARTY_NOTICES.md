# Third-party sources and asset terms

The root MIT license covers this project's MIT implementation. It does not
replace upstream code, model, dataset or video terms.

| Component | Source | Terms and use |
|---|---|---|
| Unitree G1 MJCF | https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1 | BSD-3-Clause; pinned in humanoid/UPSTREAM.json |
| Inspire right-hand URDF/STL | https://github.com/unitreerobotics/xr_teleoperate/tree/main/assets/inspire_hand | Repository Apache-2.0 terms; revision/conversion in humanoid/BOOK_G1_UPSTREAM.json |
| Official G1 wrist attachment | https://github.com/unitreerobotics/unitree_ros | G1 DFQ attachment reference and downloaded-file hash recorded in humanoid/BOOK_G1_UPSTREAM.json |
| RH56 width planner / reference hand | https://github.com/correlllab/rh56_controller and https://github.com/correlllab/h1_mujoco | MIT; unmodified external planner, local native-FK and simulation feedback adapters; revisions in video_g1/RH56_UPSTREAM.json |
| MuJoCo | https://github.com/google-deepmind/mujoco | Apache-2.0; runtime pinned in humanoid/requirements.lock.txt |
| HaMeR | https://github.com/geopavlakos/hamer | Underlying hand-reconstruction model; original code, model and training-data terms apply |
| FastHaMeR / FastHands | https://github.com/VimalMollyn/fasterhamer and https://github.com/VimalMollyn/Mediapipe-Hands-PyTorch-CoreML | Mac hand-inference packages, pinned in reconstruction/requirements-book-mac.lock.txt; original package/model terms apply; downloaded weights remain local |
| Apple Core ML Tools | https://github.com/apple/coremltools | Mac model conversion/runtime tooling; package pinned in reconstruction/requirements-book-mac.lock.txt |
| humanoid-dexart-retarget | https://github.com/gtziafas/humanoid-dexart-retarget | External checkout only; no repository-level license found at the pinned revision; source hashes in retargeting/UPSTREAM.json; not relicensed under project MIT |
| PyRoki / jaxls | https://github.com/chungmin99/pyroki and https://github.com/brentyi/jaxls | MIT; revisions and dependency versions pinned in retargeting/ |
| MANO hand model | https://mano.is.tue.mpg.de/ | Separate licensed user-supplied assets; not redistributed |
| Track, Articulate, Act | https://github.com/brains-bots-n-behavior/track-articulate-act | External checkout; revision and commands in docs/taa.md; upstream terms apply |
| SAM3 / SAM 3D Objects | https://github.com/facebookresearch/sam3 and https://github.com/facebookresearch/sam-3d-objects | External source and gated model weights; their separate source/model terms apply; no weights redistributed |
| Depth Anything 3 / MoGe | https://github.com/ByteDance-Seed/Depth-Anything-3 and https://github.com/microsoft/MoGe | External geometry models; source/model terms apply; no weights redistributed |
| SPIDER | https://github.com/facebookresearch/spider | External CC-BY-NC source; unmodified GPU sampling/backend code; pinned in scripts/gpu/UPSTREAM.json; not relicensed under this project MIT |
| MuJoCo MPC Python predictive sampler | https://github.com/google-deepmind/mujoco_mpc | Apache-2.0; external unmodified Python module and license, pinned and hash-checked in manipulation/MPC_UPSTREAM.json |
| TrackCraft3R / Wan2.1 | https://github.com/cvlab-kaist/TrackCraft3r and https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B | External tracking/base-model source and weights; pinned in scripts/taa/UPSTREAM.json; original terms apply; no weights redistributed |

Downloaded robot assets retain their upstream licenses. Model weights, caches, licensed MANO files, unpublished recordings and full
generated runs are excluded from source control. The selected silent
Book5 recording and current motion previews are published with user authorization. The source video and synthetic
robot images have distinct provenance; generated episodes are not new human
recordings. No third-party video is bundled into this workflow.
