# G1/Inspire retargeting

The adapter calls `solve_retargeting` and `create_laplacian_interaction_mesh` from
[humanoid-dexart-retarget](https://github.com/gtziafas/humanoid-dexart-retarget).
PyRoki/JAX optimizes the observed right hand against G1/Inspire joint geometry.
Upstream files remain unchanged; revisions are pinned in
[UPSTREAM.json](../retargeting/UPSTREAM.json).

## Setup

```bash
python3 -m scripts.setup_dexart
```

The setup creates a separate CPU environment. The original simulator environment
remains `.venv-g1`. Robot landmarks exported to PyRoki are checked against MuJoCo
forward kinematics before optimization. The pelvis and unobserved joints stay
supported/fixed; no second hand is inferred from the recording.

## Moving-object targets

After full hand extraction and camera calibration:

```bash
.venv-dexart/bin/python -m retargeting.video \
  --hands runs/full-hands --scene runs/book5-space \
  --cameras runs/book5-cameras/camera_poses.json \
  --object-track examples/object-motion \
  --start-frame 228 --end-frame 428 --stride 6 --window 24 \
  --output runs/moving-case-retarget
```

The object input supplies local sampled surface anchors and per-frame world
poses. The adapter excludes invalid and review-flagged object poses; it does not
interpolate them. Without `--object-track`, it uses static book anchors.

The tested 34-frame segment at source frames 228–426 has mean wrist/tip residuals
of 12.1/9.7 mm against estimated human landmarks. Its largest target change is
0.52 rad, so it is not directly executable. A one-frame bounded IK fit at
the selected segment's first observed frame initializes the upstream solver.
[Saved fit report](../examples/physics/moving-case-retarget.json)

## Outputs

- `retarget.npz`: complete source timeline, solved/observed masks, bounded actions,
  target/fitted landmarks and residuals. Unsolved rows remain NaN.
- `scene.xml`: composed G1/Inspire scene.
- `joint-map.json`: named controls and joint-state order.
- `report.json`: method, upstream revision, weights, fit residuals, root drift,
  joint-limit clipping and continuity checks.

The interaction objective is kinematic. Its collision penalties and bounded
joints do not certify a physical grasp. Different windows are solved separately
and their boundaries may be discontinuous. The reliable object segment covers
only part of the recording; the uncertain flip remains unresolved.

The [physical transfer experiment](physics.md) uses a separate contact-aware
bounded IK fit and simulated force feedback. It does not execute the unverified
PyRoki trajectory or claim that the upstream solver produced the physical result.
