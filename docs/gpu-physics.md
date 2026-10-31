# GPU physics retargeting

Experimental adapter to [SPIDER](https://github.com/facebookresearch/spider),
the physics-retargeting framework used by Do as I Do. It uses the upstream
sampling optimizer and MuJoCo Warp backend unchanged. This is separate from the
working, retimed CPU pick-and-place example.

## Current result

The 128-world GPU test runs, but does not yet maintain the grasp through the
complete tested rotation. It does not improve over replaying the cleaned
reference commands:

| Same cleaned video reference | Mean object error | Maximum object error |
|---|---|---|
| CPU reference-command replay | 10.9 mm | 80.4 mm |
| GPU optimization, warm-started | 11.9 mm | 82.7 mm |

Errors are against estimated, smoothed reference poses, not ground truth. The
case is lost near the end. These small-budget adapter checks do not evaluate
SPIDER's published performance. They are not successful training demonstrations.

The final 10,880 executed motor commands were independently replayed in MuJoCo
3.7.0. All 1,088 recorded position-state vectors matched exactly. Object pose is set only at reset;
no object actuator, weld or external object force is used. GPU candidate worlds
are synchronized to an independent CPU plant, which executes robot motor commands.
[Reports and replay check](../examples/gpu-physics/README.md)

## Inputs and processing

The test uses source frames 228–426, where the estimated case poses have no
missing/review flags. It does not cover the later uncertain flip or placement.

1. Package the supported G1/Inspire model and the successful CPU grasp states.
2. Select an in-hand initial state matching the source segment's starting height.
3. Smooth position and relative rotation-vector proposals with a seven-sample,
   second-order Savitzky–Golay filter; preserve raw observations unchanged.
4. Fit the arm to the grasp-conditioned 6-DoF case motion. The grip shape is held;
   full human finger motion is not reconstructed by this step.
5. Automatically slow the reference so arm target rates stay at or below 1 rad/s.
   This 3.3 s source segment becomes 21.76 s of control reference.
6. Run receding-horizon physics optimization, shifting the previous optimized
   plan and appending a reference tail as in SPIDER's runner.

The initial grasp comes from the recorded-hand-conditioned CPU experiment; this
is an in-hand tracking test, not independent grasp acquisition. Kinematic residuals
of 2.4 mm mean / 5.8 mm maximum refer to robot grip targets, not human-pose accuracy.
Mass, friction and the box collision approximation remain assumptions. The GPU
optimizer does not infer those physical parameters from monocular video.

## Run

On the Mac, after `make setup` and `make transfer`:

```bash
.venv-g1/bin/python -m scripts.gpu.bundle \
  --rollout runs/transfer/rollout --output runs/gpu-input
.venv-g1/bin/python -m scripts.gpu.video_reference \
  --bundle runs/gpu-input --output runs/gpu-video-input
```

Copy `runs/gpu-video-input/` to the GPU host as `/workspace/physics/input/`, and
copy this repository's `scripts/gpu/` there. Use a CUDA PyTorch image and uv:

```bash
bash scripts/gpu/setup.sh /workspace/physics
/workspace/physics/env/bin/python scripts/gpu/optimize.py \
  --spider /workspace/physics/spider --input /workspace/physics/input \
  --output /workspace/physics/run --seconds 21.76 \
  --samples 128 --iterations 1 --horizon .32 --slew 1.2
```

Use a new output directory for each attempt. Add `--replay-only` for the reference
command baseline. The final test used 2 ms integration, 80 ms replanning, a 320 ms
horizon and 1.2 rad/s motor-target slew limits. Noise is applied only to right-arm
and Inspire controls. The pelvis remains supported; no balance/walking is tested.

The tested RTX 3090 has 24 GB VRAM. First kernel compilation took about three
minutes; the final optimization took four minutes with caches populated. The
pinned source revision, hashes and observed runtime are in
[UPSTREAM.json](../scripts/gpu/UPSTREAM.json). SPIDER retains its CC-BY-NC terms.
No model weights or Hugging Face login are needed for this saved-input stage.
