# Method

## Video and scene estimation

A right-hand RGB recording shows a small case being lifted from a paperback
and placed on a table. FastHaMeR/Core ML estimates the hand mesh and landmarks.
Forced right-handedness prevents framewise handedness flips. Missing detections
remain explicit. MANO assets are user-supplied and are not redistributed.

Book corners and assumed dimensions constrain camera alignment and scale.
The workspace uses photographic planes, a rigid book and a rounded convex case.
Mass, friction and hidden geometry are engineering priors. Image consistency
is not independent metric ground truth.

Optional CPU tracking was investigated with ViSP and M3T. Occlusion and depth
ambiguity limited tracking. The retained motion uses conservative image-centre
corrections while preserving prior depth/orientation where visual evidence is
weak. See [tracking details](object-tracking.md).

## Robot retargeting

The embodiment combines the pinned Menagerie G1 model with Unitree's Inspire
right-hand assets and official wrist attachment. Seven arm motors and six hand
motors control the arm and five coupled fingers; nineteen simulated joint
positions include passive mimic joints.

Bounded inverse kinematics maps estimated wrist/palm/finger landmarks to robot
targets. The teacher adds clearance, grasp closure, carrying, lowering, release
and retreat adaptations. These are robot-specific changes, not exact human-pose
recovery. Native joint/effort limits remain active.

The G1 faces the centre of the table's longer edge. The primary camera follows
an unmirrored, phone-aligned view. A compiled-geometry check rejects side-on
layouts. The pelvis is fixed on a support.

## Contact-controlled demonstrations

MuJoCo advances at 1 ms; observations and targets are recorded at 20 Hz.
The case remains a free rigid body. Only robot motor commands drive interaction
after initialization; no object attachment, pose playback or external case
forces produce a grasp.

Five fingers provide summed normal/shear force magnitudes at external
intermediate/distal contacts. Known robot self-contact is excluded using robot
geometry, without selecting contacts by object identity. These are idealized
pad signals, not calibrated taxels or a validated slip sensor.

The expert adjusts bounded thumb/index/middle closure targets from normal
forces. Once contact occurs, further reference closure is cancelled so it does
not overwhelm pressure regulation. Sustained thumb/opposing contact can pause
the lift transition. Opening remains permitted during release.

## Variation and recovery

Case spawn varies independently of the fixed destination and common robot
start. A smooth wrist-offset adaptation approaches the shifted case and returns
to the destination. Small arm errors and short action prefixes from the existing
policy's nominal training-scene rollout create recovery starts. Prefix actions
initialize physics; they are not imitation targets. This is not full DAgger.

Mass, friction, timing, textures, lighting and camera parameters vary by group.
Every physical group has three visual versions, kept together in one split.
Candidates that fail verification are retained separately and excluded from
successful imitation data. Rejection-sampled expert quality is not a policy
success rate.

## Learning

SmolVLA receives RGB, joint positions, an instruction and optionally fingertip
forces. It predicts thirteen actual motor targets. Teacher phase, source time,
object ground truth and audit labels do not enter the actor.

Recorded executed targets already contain expert force corrections. Cloning
those targets and adding the same correction again would change the control
contract. Targets before correction are retained for separate future
hierarchical-controller experiments.

The current tactile-aware dataset is complete; training on it has not started.
[Results](results.md) separate expert quality, original-scene reproduction and
held-out learned-policy performance. [GPU training](../training/CONTACT_TRAINING.md)
defines a matched vision/proprioception versus tactile-input comparison.
