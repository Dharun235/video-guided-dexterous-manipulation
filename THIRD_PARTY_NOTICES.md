# Third-party sources and asset terms

The root MIT license covers this project's MIT implementation. It does not
replace upstream code, model, dataset or video terms.

| Component | Source | Terms and use |
|---|---|---|
| Unitree G1 MJCF | https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1 | BSD-3-Clause; pinned in humanoid/UPSTREAM.json |
| Inspire right-hand URDF/STL | https://github.com/unitreerobotics/xr_teleoperate/tree/main/assets/inspire_hand | Repository Apache-2.0 terms; revision/conversion in humanoid/BOOK_G1_UPSTREAM.json |
| Official G1 wrist attachment | https://github.com/unitreerobotics/unitree_ros | G1 DFQ attachment reference and downloaded-file hash recorded in humanoid/BOOK_G1_UPSTREAM.json |
| MuJoCo | https://github.com/google-deepmind/mujoco | Apache-2.0; runtime pinned in humanoid/requirements.lock.txt |
| LeRobot / pretrained SmolVLA | https://github.com/huggingface/lerobot and https://huggingface.co/lerobot/smolvla_base | Apache-2.0; model revisions and adaptation in training/UPSTREAM.json; package versions in training/requirements-smolvla-mac.lock.txt |
| FastHaMeR / FastHands | Python packages pinned in reconstruction/requirements-book-mac.lock.txt | Original package/model terms apply; downloaded weights remain local |
| MANO hand model | https://mano.is.tue.mpg.de/ | Separate licensed user-supplied assets; not redistributed |
| M3T | https://github.com/DLR-RM/3DObjectTracking/tree/master/M3T | MIT; optional tracker adapter and macOS build adaptation, revision in reconstruction/BOOK_POSE_UPSTREAM.json |
| ViSP | https://github.com/lagadic/visp | GPL-2.0-or-later; standalone adapter in native/book_pose_tracker is GPL-compatible; revision in reconstruction/BOOK_POSE_UPSTREAM.json |

Downloaded robot assets retain their upstream licenses. Native build outputs,
model weights, caches, licensed MANO files, unpublished recordings and full
simulation collections are excluded from source control. The selected silent
Book5 recording and compact simulation example are published with user authorization. The source video and synthetic
robot images have distinct provenance; generated episodes are not new human
recordings. No third-party video is bundled into this workflow.
