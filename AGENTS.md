# Project scope

This project turns a supplied tabletop recording into a supported Unitree G1
with a five-finger Inspire right hand, physical rollouts and verified robot data.
Keep changes focused on this workflow. Do not reintroduce unrelated robot tracks.

## Runtime and models

Active scope is full-video right-hand extraction and G1/Inspire retargeting,
followed by physical verification. Do not reintroduce VLA training, RL, collection
splits or policy benchmarks unless the user requests them. Use .venv-g1 for
simulation/tests and .venv-book-reconstruction for FastHaMeR/Core ML extraction.
Preserve official robot models. Whole-body reconstruction is not required for
this close-up hand recording. The new direct IK path is kinematic only; the
included physical reference is an older adapted trajectory.

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

Every attempt uses a new output directory. Preserve source checksums and missing
frame masks. Never silently fill hidden hand/body observations or report a
kinematic fit as physical success. Run .venv-g1/bin/python -m pytest -q humanoid/tests.
Keep the source video, reconstruction dependencies and existing contact example.
Previous policy/data work has been removed from active scope and saved outside
the repository in a recovery archive at the user's request.

Raw robot state has nineteen right arm/hand joint positions; controls are seven
arm targets plus six Inspire targets. Simulated touch has ten external fingertip
normal/shear summaries. Do not select sensed contacts by object identity.

The user authorized the selected silent Book5 recording and project-generated
previews for documentation. MANO files, weights, downloaded source, caches and
full generated runs remain ignored. Do not publish credentials or licensed assets.
