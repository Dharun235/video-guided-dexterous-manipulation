# Project scope

This project turns a supplied tabletop recording into a supported Unitree G1
with a five-finger Inspire right hand, physical rollouts and verified robot data.
Keep changes focused on this workflow. Do not reintroduce unrelated robot tracks.

## Runtime and models

Run from the repository root. Use .venv-g1 for simulation and tests, and
.venv-book-reconstruction for Core ML hand reconstruction. Use .venv-smolvla for
policy training and its adapter tests. Keep MuJoCo evaluation in .venv-g1 through
the explicit simulation bridge. Model revisions,
licenses, conversion details and contact assumptions are recorded in
humanoid/UPSTREAM.json, humanoid/BOOK_G1_UPSTREAM.json and
reconstruction/BOOK_POSE_UPSTREAM.json. Do not modify downloaded robot models to
implement features; compose them in the project code.

## Geometry and evidence

The G1 must face the centre of the table's longer edge. Validate the actual
compiled geometry with humanoid.layout.verify_layout. Keep the right-hand model,
phone-aligned egocentric camera, and unmirrored image geometry consistent.
The pelvis is supported and fixed; no balance or walking is claimed.

Retain source checksums and distinguish estimated hand/object poses from measured
robot states. Single-view hidden geometry and scale are uncertain. A prescribed
human replay is not proof of a physical grasp. Robot demonstrations must move the
free object through actuator-driven contact, without hidden object attachment or
object-pose playback. State teaching assumptions explicitly.

## Data and verification

Every new attempt uses a new output directory. Preserve failures and exclude them
from successful imitation data. The user may explicitly request cleanup of old
attempts; retain the current source/evidence dependencies and a recovery archive
of source changes before removing unrelated work.

Run .venv-g1/bin/python -m pytest -q humanoid/tests. Verify episodes using
python3 -m humanoid audit EPISODE --check-render and collections using
python3 -m humanoid check-data COLLECTION. Physics and image rendering need local
graphics access. Do not relax failed task checks to claim success.

Keep visual variants of a movement together in data splits. Use all locked
held-out rollout conditions, including teacher failures, for later policy tests.
Teacher success, dataset consistency and learned-policy success are separate.

## Contact-aware data

Raw robot state remains nineteen joint positions. Optional tactile input appends
ten external fingertip normal/shear force magnitudes for a 29-feature model.
Use the corresponding train-only statistics; do not mix the two contracts.
Exclude robot self-contact from these idealized pad readings, without selecting
contacts by object identity. Verify forces through exact physics replay.
Teacher phases and simulator-derived grasp labels are supervision only, never
actor observations. Recorded executed targets already include expert feedback;
do not add the same correction again without a separately defined contract.

The user authorized the selected silent Book5 recording in examples/source-recording
and the compact simulation replay in examples/contact-demo for public documentation.
Other recordings, MANO files, model weights, caches, generated runs and virtual
environments stay ignored by Git. Do not publish credentials or licensed assets.
Documentation should explain methods, reproducibility and measured limits.
