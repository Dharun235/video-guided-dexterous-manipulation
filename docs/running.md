# Setup and reproduction

Run from the project root. The tested local platform is Apple Silicon macOS
with Python 3.12, Git, uv and FFmpeg.

On a Mac with Homebrew installed, `brew install uv ffmpeg git` provides the
command-line dependencies. The setup scripts use uv to create the Python 3.12
simulation environment.

## Simulation setup

```bash
python3 -m scripts.setup_g1
python3 -m scripts.setup_book_g1
make test
python3 -m humanoid --help
```

Setup downloads the pinned G1 and right Inspire assets with their licenses.
Simulation uses `.venv-g1`; reconstruction uses `.venv-book-reconstruction`;
policy training/evaluation uses `.venv-smolvla`.

## Prepared data

The selected [recording](../examples/source-recording/README.md) and a
[replayable simulation example](../examples/contact-demo/README.md) are included.
The full dataset, checkpoints and reconstruction outputs remain local.
[Measured results](results.md) describe the completed experiments.

With the full portable dataset supplied:

```bash
.venv-g1/bin/python -m scripts.verify_contact_data runs/book5-contact-dataset-02
```

Use the [GPU recipe](../training/CONTACT_TRAINING.md) for training. Mac policy
setup is available through `python3 -m scripts.setup_smolvla`.

## Reconstruction from a recording

Use the supplied silent recording or your own video, plus a licensed MANO
archive. The published recording has the same decoded frames as the original.
Install the separate runtime:

```bash
uv venv --python 3.12 .venv-book-reconstruction
uv pip install --python .venv-book-reconstruction/bin/python \
  -r reconstruction/requirements-book-mac.lock.txt

.venv-book-reconstruction/bin/python -m reconstruction.book_local hands \
  examples/source-recording/book5.mp4 runs/book5-local-03 \
  --mano /absolute/path/mano_v1_2.zip --samples 126 --right-hand \
  --cache .cache/hand-models
.venv-book-reconstruction/bin/python -m reconstruction.book_local video \
  examples/source-recording/book5.mp4 runs/book5-local-03
.venv-book-reconstruction/bin/python -m reconstruction.book_space build \
  runs/book5-space-03 --video examples/source-recording/book5.mp4
.venv-book-reconstruction/bin/python -m reconstruction.book_motion recover \
  runs/book5-motion-04 --scene runs/book5-space-03 --hands runs/book5-local-03
```

Review detections, handedness, scene dimensions, book-corner annotations and
source intervals. The calibration is specific to the Book5 case study; this is
not automatic reconstruction of arbitrary videos. Every output must be new.

The retained motion `book5-motion-06` is a reviewed fusion of `book5-motion-04`
and optional M3T attempts. [Tracking instructions](object-tracking.md) describe
those tools and limitations. The robot generator currently expects this reviewed
artifact chain at the named paths. A new recording needs reviewed calibration
and reference-path changes; running the source commands alone does not produce
the full original collection without the reviewed intermediate artifacts.

## Teacher and collection

With the reviewed scene/motion artifacts available:

```bash
python3 -m humanoid simulate runs/new-reference
python3 -m humanoid audit runs/new-reference --check-render
python3 -m humanoid preview runs/new-reference
```

The contact collector additionally uses the audited nominal reference
`runs/book5-g1-ego-09` and early action prefixes from the existing policy's
nominal training-scene rollout in `runs/book5-smolvla-evaluation-02/scenario_00`.
Those local dependencies must be supplied; they are not downloaded by setup.

```bash
.venv-g1/bin/python -m humanoid.contact_collection \
  --groups 32 --variants 3 --max-candidates 32 --output runs/new-contact-data
.venv-g1/bin/python -m humanoid.contact_finalize runs/new-contact-data
.venv-g1/bin/python -m scripts.verify_contact_data runs/new-contact-data
```

Rendering requires graphics access. Resume incomplete collection with `--resume`
without changing its plan. Failed candidates remain separate; do not relax checks
or overwrite sealed data. Finalization packages portable assets, verified split
rows, normalization, force/grasp labels, previews and checksums. It does not
train a policy.
