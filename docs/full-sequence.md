# Complete Book5 motion

The current controller executes every task section: initial palm/finger gestures,
approach, grasp, transport, approximately 120° in-hand turn, placement, release
and final wrist/finger gestures. The case remains a free rigid body. Execution sets
robot position only at reset, then applies motor commands and robot-only bias
compensation. There are no case attachments, object actuators, pose playback or
external case forces. The pelvis is supported; walking and balance are not tested.

[Comparison video](../examples/video-to-g1/comparison.mp4) ·
[Recording and collection note](../examples/source-recording/README.md) ·
[Machine-readable results](../examples/video-to-g1/report.json)

## Inputs

The recording contains 748 frames at about 60 Hz, captured on an iPhone 14 held
in the left hand while the right hand manipulated the case. Supplied dimensions:
book 22 × 22 × 2 cm, case 6 × 6 × 2 cm, hand approximately 23 cm wrist to middle
tip. The case is approximated by a rigid box; the book is fixed and textured from
the recording. Case mass 55 g, case friction 0.7, hand friction 1.0, table extent
and zero lens distortion are assumptions.

Published inputs include hand-pose proposals and raw observations, book-corner
tracks, initial segmentation, object-pose candidates and the verified RH56 grasp.
The hand observations have 683/748 raw detections. The object candidate contains
740/748 poses, including 310 review flags and eight gaps. Flags and missing values
remain unchanged. Pose coverage is not accuracy.

## Approach and data use

This is a hybrid observation-driven retargeting and feedback-control pipeline.
There is no VLA, reinforcement learning or end-to-end video-to-action model.
`make video` processes published observations and grasp inputs; it does not rerun
neural extraction from the raw video. New observations use the linked extraction
and reconstruction guides.

| Component | Approach | What determines it |
|---|---|---|
| Hand observations | FastHaMeR/MANO on Core ML | Estimated wrist orientation, mesh and 21 hand landmarks from the recording |
| Metric cameras | OpenCV planar PnP and focal-length reprojection fitting | Tracked book corners and supplied 22 cm square book dimensions |
| Physical scene | Measured rigid boxes, initial-mask bounding-box registration and photographed book texture | Supplied dimensions, saved reconstruction inputs and initial segmentation |
| Gesture retargeting | Native hand landmark/palm fitting; saved initial fit and final fitted finger commands | Human hand estimates; missing observations remain flagged |
| Grasp prior | RH56-derived palm path, native finger-row centring and calibrated pre-shapes | Object geometry, native hand coupling and reviewed wrapping pattern |
| Arm and waist | Bounded SciPy nonlinear least squares using MuJoCo forward kinematics | Palm position/orientation targets, seed regularization and soft arm–torso clearance penalties |
| Rotation and timing | SO(3) spherical interpolation, reviewed phases and joint-speed retiming | Selected observed palm rotations plus explicitly adapted timing |
| Physical control | Position servos, robot bias compensation and per-finger pressure feedback | Simulated joint state and external contact forces |

The current physical path uses the local SciPy/MuJoCo IK implementation. The
separately documented humanoid-dexart-retarget/PyRoki adapter is an earlier
kinematic stage, not the controller executing this complete sequence.

During grasp and carry, finger commands come from the geometric grasp prior and
pressure regulation, rather than direct frame-by-frame MANO angle copying.
The 25 mm centring offset, native finger pre-shape coefficients and force targets
are calibrated design choices. The recording determines the demonstrated task,
gesture estimates, turn anchors, placement evidence and scene appearance; it
does not uniquely reveal hidden contacts or physical parameters.

Reviewed or prescribed parts include phase boundaries, the safe transition above
the scene, the increased lift clearance, placement support height and final yaw.
The case translation follows a phase-based transport proposal derived from the
estimated destination, not every raw tracked object pose. Carry rotation follows
selected human-palm anchors, not an independently verified full object trajectory.
Mass and friction are assumed. These choices are separated from the unchanged
observations and their review flags.

## Steps

Run after `make setup`. `make video OUTPUT=runs/video-01` performs steps 1–5;
rendering is optional. Every stage writes into a new output directory.

1. Fit metric cameras from book corners using `video_g1.calibration`. Inputs are
   `configs/book5.json` and `examples/full-video/camera-observations.json`.
2. Build measured book/case collision geometry and fit the initial case mask using
   `video_g1.scene`. The book texture comes from `examples/object-motion/scene`.
3. Generate an open-to-wrap grasp using `video_g1.wrap_grasp`, then all-frame
   controls using `video_g1.book5_sequence`. The native finger row is centred over
   the case and every finger participates. The initial human-derived gesture fit
   is preserved in `examples/full-video/initial-gestures.npz`; final finger motions
   still come from the hand fit. Native coupling and geometry remain unchanged.
4. Execute `video_g1.sequence`. All five fingers must load the same opposed
   external body for 150 ms before transport. Finger pressure on the book cannot
   substitute for pressure on the held body. Contact loss stops arm/waist motion.
   Feedback uses external contacts without case-class labels; explicit case
   identity appears only in scoring. Pressure targets are 1.20/0.48/0.36/0.24/0.12 N
   for thumb/index/middle/ring/pinky. Unloaded fingers close toward contact, then
   regulate pressure; they do not permanently latch an earlier support touch.
   Software limits non-thumb closure to 0.9 rad; native model limits are unchanged.
   These are simulation settings, not measured human or hardware sensor forces.
5. Score carry, turn, placement, release and initial palm motion. Independently
   replay the complete arm/hand/waist motor stream with `video_g1.sequence_replay`.
   No object states are replayed into the plant.
6. Optionally render `video_g1.sequence_preview`. This combines the recording,
   estimated moving phone view, complete robot and close contact view. Only the
   phone panel hides the torso visually; physical collisions remain active.

Individual planning/execution commands after building the scene:

```bash
.venv-g1/bin/python -m video_g1.book5_sequence \
  --scene runs/video-01/scene --output runs/another-reference
.venv-g1/bin/python -m video_g1.sequence \
  --reference runs/another-reference --output runs/another-rollout
```

The initial gesture uses a reachable robot workspace and reduced translation
amplitude, preserving observed palm orientation and fitted finger commands.
The transition rises clear before rotating into the grasp. Carry rotation comes
from reviewed human-palm anchors expressed relative to the native grasp frame.
The placement translation derives from the source object track with explicit
legacy-map scale adaptation. A reviewed final yaw and table support prior replace
uncertain tilt at placement. Clearance is increased to 10 cm for the larger hand.
After release, observed relative wrist turns and fitted finger commands are
retained; a boundary hold is used when the hand leaves the image. These are robot
control proposals, not corrected human/object observations.

Arm and waist IK uses bounded SciPy least squares, palm pose objectives and
shoulder/elbow/wrist clearance from the torso. The three native waist joints
extend the arm workspace through the turn. Collision penalties are soft and cover
selected arm/torso pairs; physical MuJoCo collision geometry remains active.
The hand geometry, actuator limits and mimic equations are unchanged.

## Reviewed phases and timing

| Source frames | Controller section |
|---|---|
| 0–180 | Initial palm/finger gestures |
| 181–255 | Clear transition and approach |
| 256–310 | Coordinated native grasp closure |
| 311–330 | Contact confirmation and settling |
| 331–390 | Lift and transport |
| 391–570 | In-hand turn and return |
| 571–610 | Placement and supported settling |
| 611–635 | All-finger release |
| 636–747 | Retract, then final wrist/finger gestures |

Phase boundaries are annotations from reviewing this recording. They are not
learned event detection. Source indices are never dropped or reordered. Planning
limits command speed to 0.8 rad/s, with initial gestures limited to 0.45 rad/s;
the executor additionally limits arm/hand slew to 1.2 rad/s. These are simulation
design settings, not measured hardware limits. The 12.5 s source becomes 70.5 s of
simulation. The comparison plays at source time and labels execution time.

## Results

| Nominal measurement | Result |
|---|---|
| Source frames retained | 748/748 |
| Initial palm error against extracted estimates | 0.79° mean; 3.09° maximum |
| Final gesture palm tracking against retargeted FK | 0.89° mean; 2.26° maximum |
| Maximum case lift | 103.4 mm |
| Maximum case turn | 121.4° |
| Opposing contact in scored carry, frames 350–595 | 100% |
| Carried position error against adapted target | 4.1 mm mean; 13.9 mm maximum |
| Carried orientation error against adapted target | 2.15° mean; 4.57° maximum |
| Final position/orientation error | 4.9 mm; 2.16° |
| Each finger loads the case in turn frames 405–570 | 100% for all five, above 0.05 N |
| Maximum approach flexion, frames 225–255 | 0.26 rad (14.6°), with no parked fist |
| Final finger contact | None |
| Final position drift, frames 700–747 | 0.0 mm |

A task pass requires lift over 50 mm, turn over 100°, opposing contact in over
99% of the scored carry frames, carried position error below 20/40 mm mean/max,
carried orientation error below 5°/20° mean/max, placement within 10 mm, final drift below 2 mm,
release without finger pressure and no controller abort. The complete-sequence
check additionally requires initial and final palm checks below 10° mean and 30° maximum.
The wrap check also requires approach flexion below 0.5 rad and case-specific
load above 0.05 N on every finger in over 95% of turn samples (405–570). It checks
that fingers close on the object rather than curl away from it. It does not prove
exact human joint-angle reconstruction. These thresholds assess this adapted task, not exact video imitation. Errors
against fitted pose estimates do not establish true human pose accuracy.

The full sequence also passed with initial case x shifted +3 mm and −3 mm.
Final position errors were 6.8 and 3.3 mm, respectively. These are two deterministic
perturbations, not a generalization success rate. Removing force feedback failed
the grasp and the contact gate withheld transport. The separate
[grasp evaluation](../examples/rh56-evaluation/README.md) includes wider parameter
checks; those results must not be treated as full-sequence reliability tests.

The controlled checks can be run separately:

```bash
.venv-g1/bin/python -m video_g1.sequence_audit \
  --reference runs/video-01/reference --output runs/video-audit-01
```

This repeats the two reset-position offsets and the force-feedback negative
control with the same reference. Each condition saves its actual numerical run.

## Outputs

| File | Contents |
|---|---|
| `cameras/cameras.npz` | Estimated intrinsics and per-frame book-relative camera poses |
| `scene/` | Measured collision scene, book texture and approximation metadata |
| `grasp/` | Regenerated native five-finger prior, geometric residuals and provenance |
| `reference/reference.npz` | 748 source samples, 13 arm/hand commands, three waist commands, phase labels, native FK landmarks, object targets and preserved observation flags |
| `reference/fit.json` | IK residuals and retimed duration |
| `rollout/rollout.npz` | Actual full qpos, 19 arm/hand states plus three waist states/velocities, motor streams, external finger forces, case-specific finger loads, contact gate and object/hand poses |
| `rollout/report.json` | Physical task and gesture checks with explicit limitations |
| `pipeline.json` | Per-stage wall times, GPU-use and rendering flags |
| `replay-check.json` | Independent motor-only replay comparison |
| `runtime.json` in stage folders | Input/source hashes and runtime versions |
| `preview/` when rendered | Comparison video, looping GIF, selected frames and playback manifest |

Commands are controller targets, not observed human actions or a learned policy.
The archived numerical output remains labelled unsuitable as a verified learning
dataset until a training/evaluation protocol is selected and broader checks pass.

## Compute and verification

Apple M2, 8 CPU cores, 16 GiB RAM; Python 3.12.13, MuJoCo 3.8.1, NumPy 2.5.4,
SciPy 1.18.1. The current saved-input run, IK, physics and rendering use no GPU.
A saved-input run including fresh planning, physics and independent replay took
44.3 s wall time on this host, excluding setup and rendering. This is one
measurement; stage timings are saved in `pipeline.json`. Dependencies are pinned in `humanoid/requirements.lock.txt`. Rendering needs
local graphics access. New neural hand observations need licensed MANO assets;
new TAA reconstruction uses the separately documented NVIDIA environment.

`make test` checks the numerical utilities, contact readout, native coupling and
grasp-confirmation/loss behavior. CI runs tests and Ruff. CI also downloads the pinned robot assets and runs the complete CPU sequence
and motor replay. A [fresh-checkout check](verification.json) independently
installed the runtime, ran all 50 tests, reproduced the published numerical
arrays exactly and generated the comparison video/GIF on Apple Silicon. The saved result
includes motor-only replay, the two small position checks and the force-feedback
negative control. Checksums identify the published evidence.

## Problems and lessons

- The earlier thumb–index pinch transported the case but parked the other fingers
  at 1.53 rad before contact. The new approach stays below 0.26 rad in the scored
  approach window and all four fingers participate in the grasp. Enlarged source
  frames 450–510 show why a pinching task solution was insufficient.
- A support touch could latch a finger even after it stopped touching the held
  body. Shared-body confirmation and continuous pressure feedback replace that
  behavior. Regression tests reject support-only force and unloaded fingers.

- A direct gesture-to-grasp joint blend swept the case away. Raising clear before
  reorientation preserved the untouched initial case.
- Tested fixed-torso IK fits missed the peak turn. Native waist motion extended
  the reachable pose range while retaining a supported pelvis.
- Opening unused fingers during an earlier release pushed the case about 27 mm.
  The current five-finger controller settles the case on the table, releases all
  grasp fingers, then retracts before the final gestures; final error is below 5 mm.
- Sparse initial-gesture fitting missed brief rotations. Fitting every initial
  frame and slower retiming reduced mean/max palm error to 0.79°/3.09°.

## Execution model

The sequence uses a supported pelvis, native coupled Inspire joints, a measured
rigid-box scene and reviewed motion phases. Grasp geometry, workspace, clearance,
timing and pressure targets are calibrated for Book5. Pose errors are measured
against extracted or adapted references, and mass/friction are simulation
parameters. The saved observations retain their validity and review flags.
