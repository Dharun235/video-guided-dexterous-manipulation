# Project scope

Keep this project focused on the supplied tabletop recording, supported Unitree
G1, native Inspire right hand and MuJoCo. Do not add VLA training, RL, dataset
splits or unrelated robot tracks unless requested. Preserve the source recordings,
licensed assets, required models and selected run inputs.

## Methods and environments

- `.venv-g1`: simulation, contact-aware IK, validation and tests.
- `.venv-book-reconstruction`: FastHaMeR/Core ML hand extraction.
- `.venv-dexart`: pinned humanoid-dexart-retarget/PyRoki kinematic adapter.
- `scripts/gpu/`: experimental SPIDER/MuJoCo Warp adapter; separate CUDA runtime.

Keep external solver source unchanged and honor its pinned revision and license.
The default physical transfer uses our bounded contact IK and force feedback;
it is not the output of the upstream PyRoki solver. The optional CPU MuJoCo MPC
adapter and GPU SPIDER adapter are separate experiments.

## Model and execution invariants

The G1 faces the centre of the table's longer edge. Verify compiled geometry with
`humanoid.layout.verify_layout`. Retain the genuine right-hand geometry,
unmirrored camera convention and supported pelvis. No walking or balance is claimed.

Execute motor controls with the case free. Set object pose only at reset, never
through hidden attachments, object actuators, external case forces or pose playback.
Keep IK/reference workspaces separate from the running simulator. Label robot
bias compensation and all mass/friction/collision approximations explicitly.

Control feedback sums all external finger contacts; object identity is used only
for evaluation. The local interface has 19 right arm/hand joint states, 13 position
commands and five normal/shear force pairs. Commands are simulated controller
targets, not observed human actions or a trained policy.

## Evidence and limits

The raw object candidate has 740/748 poses: 430 initially connected, 310 flagged
relocalizations and eight gaps. Coverage is not accuracy. Preserve NaNs, component
IDs and review flags. Raw final clearance is 13.8 mm; the controller uses a separate
support correction. Do not rewrite observations to match a desired outcome.

The nominal CPU transfer works as a retimed translation-task adaptation with
prescribed phases and fixed initial orientation. Lower friction and disabled
force feedback fail. Do not claim robustness, exact video imitation or training
readiness. The recorded flip remains unresolved. The separate RH56 native-hand grasp-stage
evaluation passes a measured-scene 50 mm lift, including a fresh-plan check; it
is not integrated into full-video imitation. Evidence is under
`examples/rh56-evaluation/`.

The GPU rotation test also fails: mean/max object errors 11.9/82.7 mm versus replay
10.9/80.4 mm. Reuse shifted optimized plans as in SPIDER's runner; do not discard
learned corrections at every planning step. Full command replay matched all saved
position-state vectors in MuJoCo 3.7.0. Keep this experimental result labeled as a
failure, not an improvement over the working CPU adaptation.

## Verification and storage

Run `make test` after code changes. For changes to physical execution, compare
saved numerical rollouts and record material differences. Use a new output folder
for each attempt. Keep the selected successful run and relevant failure evidence;
remove superseded trials and verified duplicate transport files when authorized.
An ignored `docs/folders-and-storage.md` may record the machine-local inventory.

The user authorized the silent Book5 recording and generated previews for public
documentation. Keep credentials, MANO model files, neural weights, downloaded
source and full local runs out of Git. Preserve hashes and provenance of published
artifacts; historical implementation hashes refer to their generating revision.
