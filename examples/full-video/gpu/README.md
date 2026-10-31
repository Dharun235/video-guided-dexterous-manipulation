# Full-video GPU test

The 64-world RTX 4090 run fails to grasp the case. Object-position errors remain
101.8 mm mean and 257.2 mm maximum, effectively identical to reference-command
replay. The case moves upward by only 0.75 mm. The reference contains converted,
uncertain object estimates, including the original gaps and review flags.

The test uses the explicitly translated grasp-registration proposal, rather
than the unanchored hand fit in the parent comparison video. It spans 22.70 s,
8.5 ms shorter than the reference. Setup and optimization took 357 s.

[Optimization](optimization.json) · [Baseline replay](replay.json) ·
[Reference assumptions](reference.json) · [CPU command check](command-replay.json)

`trajectory.npz` contains the executed 2 ms command stream and sampled simulator
states. An independent CPU MuJoCo 3.7.0 process reproduced all 1,135 saved
position-state vectors exactly. These arrays are failed execution evidence,
not successful demonstrations or training data.

[Steps and limitations](../../../docs/full-video.md)
