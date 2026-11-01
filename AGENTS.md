# Scope and structure

Maintain one Book5 pipeline: recorded right-hand manipulation → reviewed
observations → supported Unitree G1 with native Inspire hand → MuJoCo motor
execution. Root README is the entry point; keep public guides limited to
`docs/methods.md` and `docs/preparation.md`. Place input data under `data/book5`
and selected output under `examples/book5`. Keep generated runs ignored.

## Code conventions

Use Python 3.12, PEP 8, Ruff settings in pyproject.toml and short docstrings for
public functions. Keep model/kinematics in humanoid, observations in reconstruction
and planning/control in video_g1. Reuse shared contact and IK helpers. Avoid
parallel implementations, unused adapters and documentation that requires task
history to understand.

## Physical invariants

Keep genuine right-hand geometry, native joint coupling and limits, unmirrored
camera convention and supported pelvis. G1 faces the table's longer edge.
Set case state only at reset. Execute motor commands with the case free; no
attachments, object actuators, external object forces or pose playback.
Use robot-only bias compensation. Keep IK workspaces separate from the plant.
Contact feedback may group external body IDs but cannot use the case's semantic
identity; case-specific contacts are evaluation only. Verify all five fingers
load the case during the scored turn and approach without a preclosed fist.

## Data and evidence

Preserve recordings, licensed assets, required models, unique reconstruction
inputs and raw missing/review flags. Separate observations, calibrated priors,
controller commands and actual physical states. Describe measured dimensions
and assumed mass/friction accurately. Do not claim exact human reconstruction
or general-purpose video conversion from a calibrated example.

Run make test and Ruff after changes. For physical/dependency changes run make
video in a new folder and compare numerical rollouts. Keep checksums current.
Never commit credentials, MANO assets, model weights, caches or full local runs.
