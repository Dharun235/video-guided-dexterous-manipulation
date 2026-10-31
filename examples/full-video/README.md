# Full-video diagnostic

[![Full-timeline comparison](comparison.jpg)](comparison.mp4)

Left to right: recording, fitted hand, physical hand and complete physical robot.
The torso is hidden only in the middle phone-view panels. The case is static in
the kinematic panel and free in the physical panel. The physical transfer fails
to lift it; the preview is a diagnostic, not a successful imitation.

The latest preview uses the approximate 23 cm hand measurement and diagnostic
support/contact refinement. It uses 675 refined registrations from the 683 raw
hand detections. Physical wrist/tip errors are 28.8/35.4 mm against those refined
estimates; the case is not lifted.

All 748 frames remain in the numerical data. The video samples every fourth
frame over the complete timeline. Its playback uses source time; labels show
the separately retimed 25.9-second execution. Missing human observations remain
NaN. No trained policy is included.

The saved raw observations, refined proposals, camera corners, initial segmentation mask and initialization
allow [the full-video commands](../../docs/full-video.md) to run without downloading
neural weights. New hand extraction requires licensed MANO files; those files and
mesh weights are excluded. `reference.npz` contains fitted proposals; `rollout.npz`
contains actual simulated states, motor commands and external finger forces.

The measured-size scene uses a rigid box approximation of the case. This differs
from the earlier successful translation-only demo. Camera residual is 1.28 px;
that does not validate monocular depth. Hand/object alignment, collision avoidance
and the flip remain unresolved. Neither file is approved as policy training data. Earlier baseline reports are
in `baseline/`; the GPU experiment uses a different, failed anchor proposal.
