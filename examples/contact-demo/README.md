# Robot demonstration example

`preview.gif` shows a contact-controlled teacher performing the task with
synchronized simulated fingertip forces. It is not a trained-policy result.
The [human recording](../source-recording/README.md) shows the source task.

## Replay through physics

After the robot setup commands in the [main README](../../README.md):

```bash
.venv-g1/bin/python -m scripts.replay_example --output runs/example-replay
```

The example includes a scene, an initial state and a complete saved trajectory.
The command applies its motor targets to the free object in MuJoCo, verifies
state/object/force replay, and saves `execution.mp4` and `replay-check.json`.
It does not move the object by recorded object poses.

Robot meshes are fetched by setup rather than redistributed here. `checksums.json`
verifies those pinned meshes and the supplied fixture. Scene textures/geometry
include generated workspace assets; the robot scene retains the upstream G1
and Inspire terms in `licenses/`. Upstream sources are listed in
[third-party notices](../../THIRD_PARTY_NOTICES.md).

## Inspect one training frame

`sample.json` contains nineteen joint positions, ten fingertip force features,
thirteen executed targets, targets before force correction and the next joint
state. `front.jpg` and `inspection.jpg` are its RGB observations. Names and units
are included. The complete training dataset and model weights remain separate.
