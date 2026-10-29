# Retargeting integration

The active baseline fits the observed right-hand landmarks to the existing
G1/Inspire MuJoCo model with bounded SciPy least squares. It preserves all source
timestamps and marks detection gaps. It does not use the earlier scripted task
trajectory or claim physical execution.

## Interaction-aware candidate

[humanoid-dexart-retarget](https://github.com/gtziafas/humanoid-dexart-retarget)
supports G1 with Inspire RH56DFX. The inspected revision is
`238cd8b45f3969f24f9c66b91eb707ec1b51825a`. Its hand-only H2O example reads
42 hand landmarks per frame, an object mesh and per-frame 4 × 4 object poses.
It optimizes interaction geometry with PyRoki/JAX and scene-specific posture
priors. It is not a raw-video extractor or a physical grasp validator.

Required integration work:

1. Adapt landmark constraints to observed right-hand points only; do not invent
   left-hand observations or use zeros as measured points.
2. Put hand landmarks and object poses in one reviewed metric coordinate frame.
3. Match robot joint names, hand couplings and wrist mounting to our simulator.
4. Keep the pelvis supported and the unused arm parked.
5. Export named controls, source timestamps, validity masks and fit metrics.
6. Verify the resulting trajectory through actuator-driven MuJoCo execution.

The checkout is local and unmodified under `third_party/`; it is not redistributed.
No repository-level license file was found in the inspected checkout. Check
upstream permission before copying its implementation into this MIT project.
The candidate has been inspected, not integrated or benchmarked.

## Other tools

- [FRoM-W1/H-ACT](https://github.com/OpenMOSS/FRoM-W1/tree/main/H-ACT)
  supports G1/Inspire retargeting from structured human motion. Its documented
  starting point is H-GPT action representations, not an ordinary hand video.
- [Mink](https://github.com/kevinzakka/mink) is MuJoCo-based inverse kinematics.
  It is an alternative solver, not a hand/object reconstruction pipeline.
- [WHAM](https://github.com/yohanshin/WHAM) and
  [4D-Humans](https://github.com/shubham-goel/4D-Humans) use SMPL body models;
  they do not automatically supply detailed finger motion for this recording.

Do not install all alternatives into the current environments. Test a selected
candidate separately and compare it against the direct IK baseline first.
