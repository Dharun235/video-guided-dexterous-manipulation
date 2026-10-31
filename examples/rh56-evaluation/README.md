# Native Inspire grasp evaluation

The native Unitree G1/Inspire hand can now grasp and lift the measured case in
MuJoCo. This is a verified grasp-stage controller, not complete Book5 imitation.
The original video's gestures, flip, transport and release are not validated by
this ten-second test. No policy was trained.

[![Physical comparison](comparison.jpg)](comparison.mp4)

Left: native hand with reachable palm targets but no arm-clearance constraint.
Right: native hand with arm clearance and force-stop feedback. Both use a free
case. The right panel holds it about 49 mm above its initial position.

## Decision

Use [Correll Lab's RH56 planner](https://github.com/correlllab/rh56_controller)
for width-dependent grasp geometry, with our native-hand FK adapter, whole-arm
collision-aware IK and simulation force-stop controller. Keep the existing
video extraction and scene estimation. The upstream viewer sets joint positions
directly and is not evidence of physical grasping. Its hardware executor is not
the controller running here.

The upstream reference hand differs from Unitree's native model: finger/thumb
coupling, joint limits, gains and fingertip sites differ. Directly copying its
joint values is unsafe as a simulation integration assumption. At matched
controls, fingertip differences reached 14.7 mm. Our adapter reads the native
model's coupling equations; 25 fingertip comparisons against the full G1 model
agreed within 0.0106 mm. Geometry, native actuator limits and mimic equations
remain unchanged. The pelvis is supported; balance and walking are outside scope.

## Results

Success requires a lift over 25 mm with opposing thumb/finger normals for at
least 0.5 s, support during the final 0.5 s, translation slip below 10 mm and
angular slip below 10 degrees. Opposition means a normal angle over 120 degrees.
These are local evaluation thresholds, not hardware specifications.

| Check | Measured result | Interpretation |
|---|---|---|
| Upstream unit tests | 14 passed | Source planner tests pass on this Mac |
| Upstream sim-core import profile | Passed | Optional plotting dependencies required |
| Internal-width sweep, 4 modes × 7 widths | 28 nominal solutions; 7 within 1 mm width error | Solver feasibility is not grasp success |
| Isolated reference-hand fixture | 0/16 passed; best lift 2.63 mm | Our local fixture did not reproduce a physical grasp |
| Default G1 approach | 0/16; palm errors about 145–151 mm | Unreachable orientation; invalid integration |
| Reachable near-side approach without clearance | 0/16 | Up to 22.6 mm shoulder/torso penetration |
| Native G1 with clearance, without force stop | 0/4 passed | Closure targets alone failed |
| Native G1 with clearance and force stop | 3/4 passed | Useful grasp-stage improvement |
| Selected two-finger grasp | 49.15 mm lift; 1.92 s supported | Free case held physically |
| Selected relative slip | 0.162 mm; 0.333 degrees | Measured relative to the hand during support |
| Saved-command replay | All 500 states exactly matched; 10,000 commands | No object-state playback |
| Fresh plan from published seed | 49.07 mm lift; 1.90 s supported | Generation also works, rather than only saved-plan replay |
| Controlled sensitivity settings | 7/8 passed | Small deterministic checks, not a statistical success rate |

Sensitivity checks changed one setting at a time: case position ±5 mm in either
axis, global friction ×0.5 or ×0.2, and case mass/inertia ×2. All except friction
×0.2 passed. Friction scales all geometry, including supports; this does not
isolate in-hand friction. Nominal mass is assumed 55 g, case friction 0.7 and
hand friction 1.0. These were not measured from the recording.

Additional ablations passed with unused fingers left open and with the palm
fixed during closure. Neither parking unused fingers nor moving the palm
throughout closure is necessary in this nominal setup. The evidence supports
correct final geometry, arm clearance and force feedback, not every optional
feature of the implementation.

The 80 mm requested internal width is fingertip-site separation, not the case's
60 mm outer width. The upstream sweep uses a 20 mm empirical conversion. It is
not a general rule for arbitrary meshes. Two fingers means thumb and index;
the robot still uses the native five-finger Inspire hand.

## Execution

After the root setup, run from the project root:

```bash
make test
make grasp OUTPUT=runs/grasp-01
```

This fits cameras from published observations, builds the measured scene and
physically executes the saved motor plan with feedback. It needs no GPU,
MANO files, neural weights or upstream planner checkout. Outputs include scene
assets, `execution/reference.npz`, `execution/rollout.npz`, `execution/report.json`
and a runtime/input-hash manifest. Use a new output folder for each attempt.
This is a saved grasp plan for Book5, not a universal raw-video converter.

To generate a new plan using the pinned upstream algorithm:

```bash
python3 -m scripts.setup_rh56_eval
.venv-g1/bin/python -m video_g1.rh56_native \
  --scene runs/grasp-01/scene \
  --upstream third_party/rh56-controller-eval \
  --seed examples/physics/hand-seed.npz \
  --fingers 2 --width .08 --azimuth-deg 180 \
  --collision-aware --park-inactive --force-stop \
  --output runs/grasp-generated-01
```

The phase schedule is prescribed: approach 0–2 s, closure 2–5 s, settle/force stop
5–7 s, lift 7–9 s and hold to 10 s. Robot position commands are slew limited;
only robot joints receive bias compensation. Force feedback sums external finger
contacts without case labels. Case identity is used for scoring. The case pose
is set at reset only, with no weld, object actuator or external case force.
The local force-stop adapter latches finger targets when simulated pressure
reaches its threshold; it is not raw 0–1000 hardware sensor calibration.

Compute: Apple M2, 8 CPU cores, 16 GiB RAM, Python 3.12.13, MuJoCo 3.8.1,
NumPy 2.5.4 and SciPy 1.18.1. No GPU was used. Saved-plan physical execution took
2.50 s for ten simulated seconds; fresh planning plus execution took 3.39 s on
this host, excluding setup and rendering. These are single measurements.
The 16 reference-fixture trials took about 204 s. Optional upstream plots/import
checks used isolated Matplotlib 3.11.2 and PyYAML 6.0.3; these are unnecessary
for the two commands above. Exact source revisions are in
[RH56_UPSTREAM.json](../../video_g1/RH56_UPSTREAM.json).

## Alternatives assessed

[DemoGrasp](https://github.com/BeingBeyond/DemoGrasp) supports Inspire but expects
prepared reference motions and uses Isaac Gym/CUDA. It is not a direct raw-video
input replacement. No repository-level reuse license was verified during this
review. [DexGraspNet2](https://github.com/PKU-EPIC/DexGraspNet2) uses LEAP-hand
configuration and CUDA/Isaac Gym dependencies; native Inspire requires embodiment
adaptation and validation. Its source has CC-BY-NC 4.0 terms. Both received source
and dependency review only: neither was run or benchmarked here. They are less
direct choices for the present Mac/MuJoCo grasp bottleneck.

The [RH56 paper](https://arxiv.org/abs/2603.08988) reports hardware experiments.
Our tests neither reproduce nor refute those experiments: our native G1 model,
object, support geometry and local simulation controller differ. Planner/model
code is MIT; the paper has separate publication terms.

## Remaining work

1. Insert the verified grasp stage between the full-video approach and transport.
   Preserve the initial gestures and all source timestamps; label any retiming.
2. Track the recorded six-degree-of-freedom case motion while retaining opposing
   contact. Stop/recover when measured contact is lost, rather than forcing the case.
3. Verify release and final placement over the complete source timeline. Compare
   wrist, fingertips and case position/orientation, with observation gaps visible.
4. Expand perturbation tests after nominal full-motion success. Only then select
   episodes for learning or consider RL. VLA/RL is not required to solve this stage.

The recording remains essential: it supplies the task, scene appearance,
observations, measured-size scene initialization and robot seed. This grasp test
uses prescribed lift motion rather than claiming to reproduce the recorded flip.
It is not approved as a successful full-motion training episode.

## Evidence files

`nominal.json`, `fresh-generation.json`, `command-replay.json`, `sensitivity.json`
and `ablations.json` contain physical results. The three `g1-*.json` summaries
and `reference-hand-physical.json` retain failure evidence. `model-audit.json`,
`native-fk-check.json` and `width-sweep.csv` cover kinematics. `reference.npz`
contains planned controls; `rollout.npz` contains actual simulated states,
external forces and executed controls. `runtime.json` records the generating
implementation and input hashes; its Git revision precedes publication of this
new evaluation. `SHA256SUMS` checks the published artifacts.
