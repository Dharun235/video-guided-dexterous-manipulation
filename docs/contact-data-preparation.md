# Contact-aware data preparation

The current local package is `runs/book5-contact-dataset-02`; its portable
archive is about 698 MB. [The dataset pointer](../training/contact-dataset.json)
records checksums and feature dimensions. Full data remains local; the
[published example](../examples/contact-demo/README.md) contains a simulated
preview and one sensor/action sample.

## Design decisions

**Vary physical states, not only appearance.** The case spawn changes relative
to a common robot start while the destination stays fixed. A smooth adaptation
of the recorded wrist trajectory approaches the shifted case and returns to
the destination. Accepted offsets reach roughly 11 mm, compared with the earlier
4 mm range. Coverage reports distinguish proposed bounds from accepted values.

**Include recovery starts.** Small arm errors and 0/10/20/30-step prefixes from
the existing policy's nominal training-scene rollout initialize physical states.
There are eight groups per prefix category. These prefixes are not imitation
targets; held-out policy trajectories are not reused. This is initial-error
recovery, not full DAgger or recovery after arbitrary drops.

**Use touch during execution.** External fingertip normal forces regulate
bounded closure targets. Further reference closure is cancelled after contact
so it does not overwhelm pressure regulation. A sustained thumb/opposing-contact
gate can pause lift. Opening remains permitted during release. Source and
executed targets are saved separately.

**Separate observations from supervision.** RGB, joint positions and optional
normal/shear forces are model inputs. Simulator object states and grasp labels
are auditing/supervision only. Phase balancing uses training labels without
revealing phase at inference.

**Keep validation honest.** Every physical group's three visual versions stay
in one split. Normalization fits training rows only. Rejected candidates remain
separate, and their locked held-out conditions stay in policy evaluation.

## Collection and verification

The sealed collection contains 32 groups and 96 episodes, split into 60 training,
18 validation and 18 test episodes. See [results](results.md) for measured quality
and [the data contract](data-and-evaluation.md) for exact fields and thresholds.

Checks cover motor limits, contact-supported lift, placement, release, settling,
penetration, long-edge layout, complete camera images and state/object/tactile
replay. The additional 40 N sampled fingertip-force filter is a data-quality
criterion, not a hardware safety rating.

`preparation.json` must report both `data_ready` and `tactile_data_ready`, and
`collection-audit.json` must pass. A partial collection is not training-ready.
The archive and all manifest-listed files were verified after packaging.

```bash
python -m scripts.verify_contact_data runs/book5-contact-dataset-02
```

The portable package also provides a standalone `verify.py`.

## Training handoff

Use the [GPU recipe](../training/CONTACT_TRAINING.md) for matched policies with
nineteen state features or twenty-nine including touch. Both use the same
executed targets. Do not add the same expert feedback again at inference.

This data is synthetic augmentation of one personally recorded video. Sensors
are idealized external fingertip force summaries, not calibrated taxels or a
validated slip detector. Teacher quality does not establish learned-policy
success or real-hardware transfer.
