# Five-finger grasp prior

`reference.npz` is an open-to-wrap motor proposal for the measured Book5 case.
It replaces the earlier parked fingers and thumb–index pinch. Physical execution
uses the thumb and all four fingers, with contact regulation on one shared
external body. This file alone is not physical success evidence.

The native finger row is centred by a 25 mm task-space offset. Length-dependent
finger pre-shapes replace the old 1.53 rad parking targets. The thumb uses its
native opposition range; native meshes, coupling, joint limits and actuator gains
remain unchanged. Pressure targets are 1.20/0.48/0.36/0.24/0.12 N for thumb/index/
middle/ring/pinky. They are simulation tuning parameters, not video measurements.

Generate the same proposal after scene construction:

```bash
.venv-g1/bin/python -m video_g1.wrap_grasp \
  --scene runs/video-01/scene --output runs/another-wrap-grasp
```

The input is the saved native RH56 width plan under `examples/rh56-evaluation/`.
The generation code is `video_g1/wrap_grasp.py`; its geometric stage is separate
from the complete physical/contact checks in `examples/video-to-g1/`.
