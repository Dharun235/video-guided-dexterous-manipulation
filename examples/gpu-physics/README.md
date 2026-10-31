# GPU physics-retargeting check

This experimental branch uses SPIDER's sampling optimizer and MuJoCo Warp on an
RTX 3090. It follows a grasp-conditioned reference from unflagged source frames
228–426, not the complete recording. Initial height alignment, smoothing and
joint-speed-based retiming produce a 21.76 s reference without hand-timed phases.

| Same cleaned reference | Mean object-position error | Maximum error |
|---|---|---|
| Reference-command replay | 10.9 mm | 80.4 mm |
| GPU optimization with plan reuse | 11.9 mm | 82.7 mm |

The case is lost near the end. This is not an improvement or a successful
manipulation demonstration. The grasp and contact reference need further work;
these limited-budget runs do not evaluate SPIDER's published performance.

The robot uses motor commands with a free case. No object actuator, weld or
external case force is applied. All 10,880 executed commands were independently
replayed in CPU MuJoCo 3.7.0; all 1,088 recorded position-state vectors matched
exactly. This confirms command replay, not task success or training readiness.

[GPU report](optimization.json) · [Reference replay](replay.json) ·
[Independent replay check](command-replay.json) · [Reference assumptions](reference.json)

[Setup, algorithm and commands](../../docs/gpu-physics.md). The working retimed
CPU task adaptation remains in [the physical-transfer example](../physics/README.md).

[Recorded states and full motor commands](trajectory.npz) ·
[Interpolated control reference](control-reference.npz) · [Checksums](provenance.json).
These arrays are estimates/control proposals and failed simulated execution,
not additional human observations or successful training demonstrations.
