# Candidate SmolVLA commands

These commands document an existing implementation option, not an agreed training
plan. Policy choice, training method, compute and evaluation protocol remain
undecided. Training on the new collection has not started. CUDA commands below
apply only if a compatible GPU environment is selected.

See [data preparation](../docs/contact-data-preparation.md) and
[data contracts](../docs/data-and-evaluation.md) for the prepared inputs.

## Prerequisites

Provide the project, a prepared CUDA environment and the verified portable
dataset at `runs/book5-contact-dataset-02`. Full data and pretrained weights
are not bundled in this source repository. The [dataset pointer](contact-dataset.json)
records checksums. [UPSTREAM.json](UPSTREAM.json) pins the policy/backbone revisions
and LeRobot version. Use a CUDA-compatible PyTorch build for the target GPU;
the Mac lock file is provenance, not a universal CUDA installation recipe.

```bash
python -m scripts.verify_contact_data runs/book5-contact-dataset-02
```

Both readiness flags in `preparation.json` must be true, and the collection
audit must pass. Place the pinned public SmolVLA base files in `.cache/smolvla-base`.
First run five steps with batch size two to check gradients, finite losses and
memory before a full run.

## Candidate training runs

```bash
# RGB + nineteen joint positions.
python -m training.smolvla --device cuda \
  --dataset runs/book5-contact-dataset-02 --phase-balanced \
  --chunk-size 10 --steps 5000 --batch-size 4 --lr .00003 \
  --warmup-steps 100 --eval-every 250 --validation-samples 128 \
  --output runs/contact-vision-01

# Same data/targets plus ten fingertip force features.
python -m training.smolvla --device cuda \
  --dataset runs/book5-contact-dataset-02 --phase-balanced --tactile \
  --chunk-size 10 --steps 5000 --batch-size 4 --lr .00003 \
  --warmup-steps 100 --eval-every 250 --validation-samples 128 \
  --output runs/contact-tactile-01
```

These are starting settings, not measured optimal settings. Profile memory
before increasing batch size. Phase balancing uses training labels for sampling
only; phase is not an observation. Normalization uses the matching train-only
19- or 29-feature statistics. Do not mix tactile-free episodes into the tactile
run without recording their force observations.

`action` contains executed targets including expert force correction. Do not
add the same correction again when cloning those targets. `action.reference`
is reserved for separate hierarchical-controller experiments.

## Available evaluation interface

Use validation loss and physical validation rollouts to select a candidate,
then freeze it before testing. Evaluate every locked condition, including
teacher failures. Report placement, raised grasp, release, support, settling,
penetration and force peaks with the unchanged task thresholds.

For the force-input comparison, use `--replan-every 1` for both models. This
supplies fresh observations every 50 ms of simulation, matching the 20 Hz
teacher; five-step replanning observes force at 4 Hz. Inference wall time may
be slower than simulation, so this is not a real-time claim.

The [evaluation interface](README.md) detects the tactile checkpoint contract.
It requires the separate MuJoCo runtime used by the bridge. Original-scene
comparisons use a scene represented in the training data. Reserve new blind seeds after
inspecting a final test set. No RL or hardware transfer is established by this
training recipe.
