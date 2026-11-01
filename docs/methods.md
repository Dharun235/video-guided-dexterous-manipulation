# Motion planning and physical execution

[Comparison](../examples/book5/demo.mp4) · [Recorded inputs](../data/book5/README.md) ·
[Results](../examples/book5/report.json)

## From observations to commands

`make video OUTPUT=runs/video-01` executes the stages below using supplied inputs.
[Input preparation](preparation.md) covers neural inference and registration.

| Stage | Code | Output |
|---|---|---|
| Metric camera calibration | `video_g1.calibration` | Intrinsics and book-relative camera poses |
| Scene construction | `video_g1.scene` | Measured box collision geometry, initial-mask alignment and book texture |
| Native grasp | `video_g1.wrap_grasp` | Open-to-wrap palm path and six coupled hand commands |
| Full-timeline planning | `video_g1.book5_sequence`, `video_g1.ik` | Arm/hand/waist targets, phases and retimed timestamps |
| Physical execution | `video_g1.sequence`, `video_g1.contacts` | Motor stream, actual joint/object states and finger forces |
| Independent verification | `video_g1.sequence_replay` | Motor-only replay error against all saved position states |
| Comparison rendering | `video_g1.sequence_preview` | Recording, phone view, contact view and full robot |

Book-corner tracking and planar PnP fit the moving camera using the measured
22 cm square book. The initial case mask aligns a measured 6 × 6 × 2 cm rigid
box. The book is fixed. Case mass is assumed 55 g, sliding friction 0.7 and
hand friction 1.0; table footprint is assumed 0.80 × 0.38 m. Camera distortion
is assumed zero. These are explicit settings, not recovered material properties.

FastHaMeR/MANO provides wrist orientation and 21 landmarks. Hand registration
uses image alignment, approximate hand length and inferred support/contact
priors. The initial gesture fit retains all 181 source frames. Final gestures
use fitted finger commands and relative observed wrist turns, with a boundary
hold when the hand leaves view. Gesture translation is reduced to a reachable
robot workspace. Observation gaps and review flags are preserved.

The grasp uses a supplied RH56-derived palm path, a 25 mm finger-row centring
offset and calibrated finger pre-shapes. All five fingers participate. Carry
rotation follows selected observed palm anchors using SO(3) interpolation.
Placement derives from the object-pose candidate with an explicit tracking-map
scale conversion, support height correction and reviewed final yaw. Increased
lift clearance accommodates the robot hand. Phase boundaries, safe transitions,
workspace, pressure targets and retiming are reviewed design choices.

SciPy bounded nonlinear least squares uses MuJoCo forward kinematics to fit
palm pose, seed regularization and soft arm–torso clearance penalties. Three
native waist joints extend the arm workspace. The robot retains its genuine
right-hand geometry, actuator limits and tendon/mimic coupling. The pelvis
is supported. Collisions remain active throughout execution.

## Timing and control

| Source frames | Section |
|---|---|
| 0–180 | Initial palm/finger gestures |
| 181–255 | Clear transition and approach |
| 256–310 | Coordinated grasp closure |
| 311–330 | Contact confirmation and settling |
| 331–390 | Lift and transport |
| 391–570 | In-hand turn and return |
| 571–610 | Placement and supported settling |
| 611–635 | All-finger release |
| 636–747 | Retract and final wrist/finger gestures |

Planning limits command speed to 0.8 rad/s, or 0.45 rad/s for initial gestures.
Execution limits arm/hand command slew to 1.2 rad/s. Source indices remain ordered;
12.5 seconds become 70.5 seconds in simulation. The preview plays at source time
and labels simulation time.

Position servos and robot-only bias compensation drive the robot. The case is
free after reset: no object pose playback, attachments or external object forces.
Contact readout groups normal forces by external body ID. All five fingers must
load the same opposing body for 150 ms before transport. Contact loss pauses
arm/waist motion. Feedback uses contacts without case-class labels; case-specific
loads are computed separately for scoring.

Thumb/index/middle/ring/pinky pressure targets are 1.20/0.48/0.36/0.24/0.12 N.
Unloaded fingers close toward contact, then regulate pressure continuously.
Non-thumb closure has a software ceiling of 0.9 rad; native model limits are
unchanged. These forces are simulated contact measurements.

## Measurements

| Nominal measurement | Result |
|---|---|
| Source timeline | 748/748 frames |
| Initial palm error against extracted estimates | 0.79° mean; 3.09° maximum |
| Final gesture palm error against retargeted FK | 0.89° mean; 2.26° maximum |
| Case lift / turn | 103.4 mm / 121.4° |
| Opposing contact, carry frames 350–595 | 100% |
| Carried position error against adapted target | 4.1 mm mean; 13.9 mm maximum |
| Carried orientation error against adapted target | 2.15° mean; 4.57° maximum |
| Final position / orientation error | 4.9 mm / 2.16° |
| Each finger above 0.05 N, turn frames 405–570 | 100% for all five |
| Maximum approach flexion, frames 225–255 | 0.26 rad (14.6°) |
| Final contact / drift, frames 700–747 | Released / 0.0 mm |

A task pass requires lift >50 mm, turn >100°, opposing carry contact >99%,
carry position error <20/40 mm mean/max, orientation error <5°/20° mean/max,
placement error <10 mm, final drift <2 mm, release and no abort. Gesture mean/max
palm errors must stay below 10°/30°. The wrap check requires approach flexion
<0.5 rad and each finger loading the case >0.05 N in >95% of turn samples.
These assess the adapted task; fitted estimates are not human-pose ground truth.

Reset offsets of +3 mm and −3 mm along case x passed with final errors of
6.8 mm and 3.3 mm. Disabling force feedback failed the grasp and the contact
gate withheld transport. These are controlled checks, not a statistical success
rate. Run them with:

```bash
.venv-g1/bin/python -m video_g1.sequence_audit \
  --reference runs/video-01/reference --output runs/audit-01
```

## Output arrays

`reference/reference.npz` stores source frames/times, 13 arm/hand commands,
three waist commands, phase labels, native FK landmarks, object targets and
observation flags. `rollout/rollout.npz` stores actual full position state,
19 arm/hand states plus three waist states/velocities, motor streams, finger
forces, case-specific loads, contact gate and object/hand poses. Commands are
controller targets; actual states are physical simulation outputs.

`rollout/report.json` scores the task; `replay-check.json` compares independently
replayed motor commands; `pipeline.json` records stage timings. Stage manifests
record runtime versions and input/source hashes. The selected example combines
these checks in one `examples/book5/report.json`, with artifact checksums in
`SHA256SUMS`. Rendering hides the torso only in the phone panel; it does not
change physics.

## Compute and design lessons

The saved-input pipeline uses an Apple M2, 8 CPU cores, 16 GiB RAM, Python
3.12.13, MuJoCo 3.8.1, NumPy 2.5.4 and SciPy 1.18.1. Dependencies are pinned
in `humanoid/requirements.lock.txt`. Neural input preparation uses Core ML on
the Mac and an RTX 6000 Ada 48 GB Linux host for reconstruction/tracking.

- Keep fingers open during approach, then wrap around the object. The measured
  approach stays below 0.26 rad and all five fingers carry load during the turn.
- Group contacts by shared body and regulate continuously. A book touch cannot
  substitute for an opposing grasp on the held body.
- Raise clear before rotating into the grasp; use native waist motion for reach.
- Settle on the support, release all fingers, then retract. Final placement error
  is below 5 mm and the case remains stationary.
- Fit brief gestures at full source rate and retime them for the robot. Retain
  observation uncertainty separately from the commands used to execute them.
