# Data contract and evaluation

## Observations and targets

| Field | Shape | Meaning |
|---|---|---|
| `observation.images.front` | 256 × 256 × 3 | RGB; phone-aligned content is letterboxed |
| `observation.images.inspection` | 256 × 256 × 3 | Additional RGB view |
| `observation.state` | 19 | Simulated arm/hand joint positions, radians |
| `observation.tactile` | 10 | Five fingers × external normal/shear force magnitudes, N |
| `action` | 13 | Actual bounded absolute motor-angle targets, radians |
| `action.reference` | 13 | Expert targets before fingertip force correction |

At each control boundary, record the observations, apply the executed target
for fifty 1 ms physics steps, then record next state. Raw joint state remains
nineteen-dimensional. The tactile reader appends ten force values to form a
29-feature model input. The vision/proprioception ablation uses nineteen.

Joint names, actuator names, limits, units and source-video hash accompany the
records. Passive mimic joints are simulated proprioception, not independent
hardware encoders. Preserve the parked arm's background controls.

## Files

| File | Contents |
|---|---|
| `data/{train,validation,test}.jsonl` | Verified rows and image paths |
| `episode.npz` | Precise states, velocities, forces, targets and diagnostic object poses |
| `dataset.json`, `feature-contract.json` | Named feature and control contracts |
| `scene.xml`, `initial_state.npz` | Physics model and replay reset |
| `contacts.json`, `report.json`, `audit.json` | Measured contacts and task/replay verification |
| `auxiliary-labels.npz` | Simulator-derived grasp/placement labels; supervision only |
| `statistics.json`, `statistics-tactile.json` | Separate train-only normalization for 19/29 features |
| `coverage.json`, `episode-quality.csv` | Accepted variation and per-episode quality |
| `evaluation-scenarios.json` | Locked policy tests, including teacher failures |
| `manifest.json` | File checksums |

The project format is JSONL/NPZ/JPEG, not native LeRobot parquet. The provided
adapter trains the upstream policy directly. The published
[representative sample](../examples/contact-demo/README.md) shows the sensor/action
contract without distributing the full local dataset.

## Input isolation

The actor receives RGB, robot state, task text and optional fingertip forces.
Object poses, source timestamps, teacher phase and simulator-derived grasp
labels remain diagnostics or training supervision. Phase-balanced sampling
uses training labels only; it does not provide phase to the actor.

Use executed `action` targets directly. Do not add teacher feedforward or the
same force correction again. A separate hierarchy trained on `action.reference`
requires a separately specified inference contract.

## Physical verification

A successful task requires contact, lift above 2 cm, final planar error below
3 cm, settling below 2 cm/s, table support, released fingers and penetration
below 5 mm. Raised multi-finger contact must last at least 0.5 s. Anatomical
fingers are counted once even when multiple segments touch.

The contact-aware collection additionally verifies raised thumb/opposing-finger
contact, force shape, force replay, images, layout and bounded targets. Its
40 N sampled fingertip-force filter is a collection criterion, not a hardware
safety rating. Exact action replay must reproduce state/object trajectories
within numerical tolerance; tactile replay tolerance is 1e-6 N.

## Splits and policy evaluation

All visual versions of a physical movement stay in one split. Training-only
rows fit normalization. Failed demonstrations remain outside successful
imitation data; their held-out conditions remain in policy rollout evaluation.

Select models using validation, freeze the candidate, then evaluate every
locked test condition. Report individual failures and mean placement error,
not only success counts. Teacher verification does not establish policy skill.

The original reconstructed scene contributed to training, so its comparison is
reproduction. All human motion comes from one recording. Synthetic condition
splits are not independent human demonstrations or real-hardware tests.
