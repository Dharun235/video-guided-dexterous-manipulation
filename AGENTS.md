# Project scope

This project turns a supplied tabletop recording into a supported Unitree G1
with a five-finger Inspire right hand, physical rollouts and verified robot data.
Keep changes focused on this workflow. Do not reintroduce unrelated robot tracks.

## Runtime and models

Active scope is full-video right-hand extraction and G1/Inspire retargeting,
followed by physical verification. The selected upstream retargeter runs in
.venv-dexart through retargeting.video. Keep upstream files unchanged and record
objective extensions honestly. Do not claim that the upstream kinematic solver produced the separate contact-IK
physical rollout. Do not reintroduce VLA training, RL, collection
splits or policy benchmarks unless the user requests them. Use .venv-g1 for
simulation/tests and .venv-book-reconstruction for FastHaMeR/Core ML extraction.
Preserve official robot models. Whole-body reconstruction is not required for
this close-up hand recording. The original direct IK preview remains kinematic. manipulation/ now implements
a separate actuator-driven translation-task adaptation with a box collision
proxy, external-contact feedback and optional upstream MuJoCo MPC refinement.
Do not describe it as exact full-video imitation; the uncertain flip is omitted.

## Geometry and evidence

Object motion now uses overlapping TrackCraft3R windows, visible-point rigid
fits, bounded image refinement and explicitly flagged hand-prior reinitialisation.
The published candidate has 740/748 poses: 430 connected to the initial anchor,
310 uncertain relocalized poses and eight gaps. Coverage is not accuracy.
Preserve component/review flags and NaNs. Final estimated table clearance is
13.8 mm; contact correction and motor-driven manipulation remain unverified.
Do not treat these object-pose candidates as training-ready robot states/actions.

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

Every attempt uses a new output directory. Preserve source checksums and missing
frame masks. Never silently fill hidden hand/body observations or report a
kinematic fit as physical success. Run .venv-g1/bin/python -m pytest -q humanoid/tests.
Keep the source videos, current full-video runs and required reconstruction/robot
dependencies. Old policy/data runs and their recovery copies were deleted at the
user's request. Only a compact Git source-history backup is retained outside the
repository. Do not regenerate or download old experiments without a new need.

Raw robot state has nineteen right arm/hand joint positions; controls are seven
arm targets plus six Inspire targets. Simulated touch has ten external fingertip
normal/shear summaries. Do not select sensed contacts by object identity.

The user authorized the selected silent Book5 recording and project-generated
previews for documentation. MANO files, weights, downloaded source, caches and
full generated runs remain ignored. Do not publish credentials or licensed assets.

## Storage after cleanup

Superseded object/scene/solver trials and duplicate tracking transport archives
were removed on 2026-10-30. Keep candidate-04, its scene, full depth archive,
extracted tracking windows, full hand/camera inputs and required model caches.
Do not use deleted diagnostic runs as prerequisites. Current cleanup inventory
is local at runs/cleanup-2026-10-30.json.

## Physical transfer

The contact-IK controller uses an observed frame-240 hand seed and video-derived
source/destination positions. Explicit phase timing, clearance margins and fixed
initial orientation are control-design assumptions. Raw pose flags are untouched.
The robot is supported with bias-force compensation; no force or attachment is
applied to the free case. Sensor feedback includes all external finger contacts;
object-specific contacts are used only for evaluation. Store failures alongside
successes. Lower-friction and no-force-feedback trials failed; no generalization
or training-readiness claim is supported. MuJoCo MPC is pinned, unmodified upstream
Python code with our reward/bounds adapter, not the GPU SPIDER backend.
