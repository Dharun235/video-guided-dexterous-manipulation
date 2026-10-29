# Upstream G1/Inspire retargeting trial

[Source beside the fitted robot](kinematic-fit.mp4) · [Measurements](report.json)

The video shows `humanoid-dexart-retarget` through our right-hand adapter.
The upstream Laplacian solver ran on 227 observed frames at the original timing,
about 20 Hz. The case stays static in this kinematic view; no physical grasp or
moving-object trajectory is demonstrated.

The current trial has larger wrist/fingertip residuals than the earlier baseline
and a maximum 1.73 rad target step. It needs interaction and continuity refinement
before physical execution. Missing detections are not filled. The 748-row output
retains the source timeline; only every third observed frame is fitted.

The adapter uses the original MuJoCo robot's joint geometry, fixed unobserved
joints, static book anchors, stronger joint-limit penalties and a supported-root
penalty. It does not invent a second hand. See the
[integration guide](../../docs/retargeting.md) for the algorithm and commands.
