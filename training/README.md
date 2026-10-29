# SmolVLA interface

The project trains upstream SmolVLA through a JSONL/NPZ/JPEG adapter. It does
not require converting the prepared collection into LeRobot parquet.

## Inputs and targets

- Two 256×256 RGB views, task text and nineteen robot joint positions.
- `--tactile` appends ten fingertip normal/shear features for a 29-feature state.
- Thirteen executed absolute motor targets; action chunks remain within episodes.
- `--phase-balanced` balances teacher phases using training labels only.

Use `statistics.json` for nineteen features and `statistics-tactile.json` for
twenty-nine. Both are fitted on training rows only. The visual backbone stays
frozen while the action expert and state/action projections train. Revisions
and dependencies are pinned in [UPSTREAM.json](UPSTREAM.json).

The [GPU recipe](CONTACT_TRAINING.md) is the current training entry point.
The [running guide](../docs/running.md) covers local setup and reconstruction.
[Results](../docs/results.md) distinguish the older trained baseline from the
new contact-aware collection, whose model training has not started.

## Evaluation and packaging

The policy process supplies targets through a separate MuJoCo bridge. Object
poses, teacher actions, source time and audit labels are not actor observations.
Targets are applied directly with native limits and parked-arm controls intact.
No object attachment or pose playback drives a learned rollout.

```bash
python -m training.evaluate \
  --checkpoint runs/new-model/best --dataset runs/book5-contact-dataset-02 \
  --suite validation --noise zero --replan-every 1 \
  --output runs/new-validation
python -m training.package_checkpoint runs/new-model/best
```

On a GPU host, add `--device cuda` to evaluation and provide the simulator
runtime expected by the bridge. Tactile models default to one-step replanning;
use the same rate for both models in the force-input comparison. Simulation
control rate does not establish real-time inference speed.

Select using validation rollouts, freeze the candidate, then test every locked
condition including teacher failures. Original-scene comparisons are
training-derived reproduction. Packaging includes native pre/post processors,
normalization, named features, licenses and checksums.
