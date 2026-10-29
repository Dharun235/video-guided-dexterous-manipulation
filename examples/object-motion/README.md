# Case motion candidate

The original recording beside estimated case poses in MuJoCo:

[Full-frame comparison](object-motion.mp4)

- 748 source frames processed; 740 pose candidates (98.9% coverage, not accuracy).
- 430 remain connected to the initial calibration.
- 310 belong to an uncertain component anchored at frame 544 using predicted
  depth, visible case edges and the observed hand orientation relative to frame
  320. Backward tracking recovers earlier frames in that component.
- Eight rejected frames remain NaN. No pose interpolation is used.

Green labels identify tracked estimates, amber labels identify hypotheses that
require review, and red labels identify missing estimates. Every source frame
is shown at the original frame rate.

`object_track.npz` contains source-frame indices, FPS-based timestamps,
world-frame position/rotation, validity, component IDs and review flags. The world
uses the reviewed book plane and estimated metric scale. These are estimated
object poses, not measured robot states or control actions.

`scene/` contains the generated tabletop/case scene, including collision geometry
and assumed mass/friction. It contains no hand-model files or neural weights.
`initial_pose.json` and `case_surface.npy` preserve the anchor used for this run.

```bash
.venv-g1/bin/python -m scripts.taa.check_object_track \
  --track examples/object-motion --scene examples/object-motion/scene/scene.xml \
  --surface examples/object-motion/case_surface.npy
```

The imposed-pose collision check finds five frames with more than 1 mm
penetration, maximum 1.25 mm. Estimated final clearance above the table is
13.8 mm, so placement still needs contact correction. This is a kinematic pose
candidate; actuator-driven grasping and training data are not ready.

[Pipeline and commands](../../docs/object-motion.md) ·
[Pose report](report.json) · [Validation](validation.json)
