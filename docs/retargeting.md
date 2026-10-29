# G1/Inspire retargeting

The project now calls `solve_retargeting` and
`create_laplacian_interaction_mesh` from
[humanoid-dexart-retarget](https://github.com/gtziafas/humanoid-dexart-retarget).
It uses PyRoki forward kinematics and JAX trajectory optimization on the Mac CPU.
Pinned source revisions and hashes are in [UPSTREAM.json](../retargeting/UPSTREAM.json).

## Setup and upstream check

```bash
python3 -m scripts.setup_dexart
.venv-dexart/bin/python -m retargeting.check_example --output runs/dexart-example
```

The check runs four frames of the supplied H2O milk example without the interactive
viewer. It checks finite solver output, not physical task success. The source
expects prepared hand/object tracks; it does not extract them from RGB video.

## Run our recording

After hand extraction, scene fitting and camera registration:

```bash
.venv-dexart/bin/python -m retargeting.video \
  --hands runs/full-hands --scene runs/book5-space \
  --cameras runs/book5-cameras/camera_poses.json \
  --output runs/dexart-retarget --stride 3 --window 24
.venv-g1/bin/python -m scripts.preview_hand_retarget \
  --hands runs/full-hands --retarget runs/dexart-retarget --output runs/dexart-preview
```

The selected observed frames are about 20 Hz at the original timing. All 748
source timestamps remain in the output; unsolved/missing frames have invalid
flags and NaN actions. Window-end padding is a solver implementation detail and
is not exported as new observations. Gaps are not bridged by invented poses.
For full frame-rate fitting use `--stride 1`; the published trial uses stride 3.

## Adapter changes

- Export the current MuJoCo robot to a right-arm/hand URDF. Preserve joint axes,
  wrist attachment and linear finger couplings. Fix the lower body and unused arm.
- Verify landmark forward kinematics between PyRoki and MuJoCo before solving.
- Apply a strong support penalty to the remaining root variable. Export only joint
  controls into the fixed-pelvis MuJoCo scene.
- Use the observed right-hand landmarks and reconstructed static book corners in
  the upstream Laplacian interaction mesh. No left-hand observations are invented.
- Initialize from one observed-frame bounded IK fit. Final trajectory targets come
  from the upstream solver, not the SciPy baseline trajectory.
- Increase the upstream soft joint-limit penalty and explicitly bound exported
  motor targets. Record any raw excess and clipped components.

Upstream files are unchanged. These objective extensions are implemented in our
adapter. Self-collision uses PyRoki capsules fitted to conservative box proxies;
this does not establish collision-free execution in the original MuJoCo scene.

## Current trial

The upstream example passed its finite-output check. Our video trial solved
227 observed frames over the full timeline in 105 seconds on CPU.

| Metric on the same 227 frames | Upstream adaptation | Earlier SciPy baseline |
|---|---:|---:|
| Mean wrist residual | 31.7 mm | 11.7 mm |
| Mean fingertip residual | 24.9 mm | 10.6 mm |

Residuals compare inferred hand targets, not ground truth. The largest exported
target step is 1.73 rad. This trial does not improve the baseline and is not ready
for physical execution. Window continuity and moving-case interaction still need
refinement. The book anchors are not a substitute for the case's pose trajectory.

The model bridge agreed with MuJoCo landmarks within 0.00025 mm in five sampled
joint configurations. This validates coordinate conversion, not manipulation.

## External code

The retargeter stays in ignored `third_party/`; its code is not copied into the
MIT project. No repository-level license was found in the inspected checkout.
Upstream assets and dependencies retain their own terms.
