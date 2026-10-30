# Physical transfer

This experiment adapts the recorded pick-and-place task to a supported G1 arm
and Inspire right hand. It is a retimed translation task, not exact full-video
imitation: the uncertain case flip and the early free-hand gestures are omitted.

## Run

```bash
make setup
make transfer
```

Use a new output for each attempt, for example `make transfer OUTPUT=runs/transfer-02`.
This uses the included case scene, calibration, estimated object track and an
observed frame-240 hand seed. No MANO download or GPU inference is required to run
the saved example. Full hand/scene reconstruction remains in the other guides.

To render and export cameras on the Mac:

```bash
.venv-g1/bin/python -m manipulation.preview \
  --rollout runs/transfer/rollout --output runs/transfer/preview
.venv-g1/bin/python -m manipulation.export \
  --rollout runs/transfer/rollout --output runs/transfer/observations
```

Normal macOS graphics access is required. Alternatively use
`python -m manipulation.run --output runs/transfer-03 --render` in the simulation
Python environment to run all stages together.

## What the recording supplies

- Observed right-hand landmarks at source frame 240 and a bounded IK seed.
- Initial case position and orientation from the reconstructed scene.
- Destination XY from the final 30 valid video estimates. These estimates remain
  flagged for review; the destination is an estimated task target.
- Lift height from unflagged video poses. A 25 mm robot-clearance margin is added.

The saved hand seed contains derived landmarks and joint initialization, not
MANO model files or human mesh weights. Its provenance is recorded with the
published source-video checksum.

## Control and physics

1. Project candidate fingertip targets onto an explicit case collision proxy.
2. Fit bounded robot joints using landmark, contact-distance and penetration costs
   in a separate IK workspace.
3. Solve arm targets for approach, lift, carry, placement and retreat. Smoothstep
   timing and the phase durations are prescribed, rather than inferred automatically.
4. Close the hand and require opposing external finger contacts before lifting.
   Adjust closure using normal force and the shear/normal ratio. Commands are
   bounded and limited to 0.8 rad/s.
5. Step MuJoCo with the case free. Set object pose only during reset; use no object
   actuator, weld, scripted pose playback or external object force.
6. Score object-specific contact, lift, slip, release, settling and final position.
   Object identity is used for scoring, not for selecting the controller's contacts.

The pelvis is fixed. Robot bias-force compensation adds gravity/Coriolis torques;
those torques are logged separately from position actuator commands. Native joint
and hand force limits remain present. The case uses an estimated 55 g mass and
friction assumptions. Its collision proxy is a box fitted to reconstructed extents;
the neural mesh remains visual. This avoids treating its unverified uneven hidden
base as measured geometry.

Placement uses the table plane outside the book footprint, with a 12 mm release
clearance. The free case settles onto the table after release. These corrections
are control proposals; they do not overwrite raw object estimates or clear their
uncertainty flags.

## Automatic physics refinement

The optional adapter calls the unmodified Python predictive sampler from
[MuJoCo MPC](https://github.com/google-deepmind/mujoco_mpc/tree/main/python/mujoco_mpc/demos/predictive_sampling).
It evaluates candidate motor commands in forward physics on the CPU. Revision,
file hashes and license are pinned in `manipulation/MPC_UPSTREAM.json`.

```bash
python3 -m scripts.setup_mpc
.venv-g1/bin/python -m manipulation.rollout \
  --fit runs/transfer/grasp --task runs/transfer/targets/task.json \
  --mpc --output runs/transfer-mpc
```

The tested adapter refines the carry phase only, with a 120 ms horizon, twelve
samples and one improvement iteration every 80 ms of simulated time. It bounds
arm changes to 0.003 rad and other changes to 0.06 rad around the nominal command.
It has access to simulated object state. This is model-based refinement, not a
trained policy or an image-only controller. CPU execution is slower than real time.

Earlier MPC trials during grasp acquisition failed. Bounded carry refinement
passes the nominal transfer, but does not improve every metric; it is optional.

[Do as I Do](https://github.com/malik-group/do-as-i-do/tree/main/retargeting) uses
SPIDER-based sampling MPC with MuJoCo Warp and requires NVIDIA CUDA. That is a
closer route to full-trajectory physics optimization, but it is not the backend
executed here. [SPIDER](https://github.com/facebookresearch/spider) also provides
Inspire/G1 examples and reference tracking. No CUDA backend or model weights are
installed by this CPU experiment. None of these controllers removes the need to
review uncertain monocular geometry or supply mass/friction assumptions.

## Outputs and checks

| File | Contents |
|---|---|
| `targets/task.json` | Source-derived goal, support correction, margins and review flags |
| `targets/review.npz` | Original pose arrays plus separate support-correction proposals |
| `grasp/grasp.npz` | Fitted grasp, targets and named model indices |
| `rollout/rollout.npz` | Actual physics states, position commands, forces and phase labels at 50 Hz |
| `rollout/schema.json` | Names and ordering for 19 joint states, 13 controls and five contact groups |
| `rollout/report.json` | Metrics, acceptance thresholds, assumptions and outcome |
| `observations/episode.npz` | Synchronized states/actions and relative image paths at 10 Hz |
| `observations/manifest.json` | Schema, provenance and execution limitations |

Each camera export has 210 images at 256 × 256 for the 21 s example. Contact
observations have shape `(frames, 5, 2)`: external normal/shear force summaries
in newtons. They are simulated contact summaries, not a calibrated hardware
sensor model. The 10 Hz commands are samples of the 1 kHz controller, not a verified 10 Hz
open-loop action sequence. The front camera is robot mounted; the inspection camera is external.

Success requires more than 25 mm lift with opposing case contact for over 0.3 s,
final position error below 40 mm, release and settling. The final 0.5 s must remain
below 0.03 m/s linear and 0.3 rad/s angular speed. Slip is measured as change in
case position relative to the hand during lift/carry; rotation is not included in
that slip statistic. All numerical thresholds are explicit experimental criteria.

[Saved results and data](../examples/physics/README.md). Failures are retained in
the evaluation summary. These few deterministic checks do not establish a
success rate, broad generalization or training readiness.
