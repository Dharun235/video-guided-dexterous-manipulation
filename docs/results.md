# Measured results

These results separate expert demonstrations from learned-policy rollouts.
Aggregate values and artifact checksums are in [results.json](results.json).

## Contact-aware dataset

The current collection contains 96 verified episodes in 32 physical groups,
with 24,732 samples. Splits are 60/18/18 episodes for train/validation/test.

| Expert/data metric | Result |
|---|---:|
| Median planar placement error | 6.1 mm |
| Mean planar placement error | 8.6 mm |
| Maximum planar placement error | 26.1 mm |
| Minimum raised thumb/opposing-finger hold | 4.15 s |
| Maximum sampled fingertip normal force | 39.95 N |
| Maximum replay force error | 0 N |

Placement statistics use one trajectory per physical group. All accepted
records pass physical task, image, split and replay checks. Preflight attempted
168 candidates and rejected 136. This is quality-filtered teacher data, not an
unbiased policy success experiment. The new tactile-aware model is not trained.

## Earlier SmolVLA baseline

The baseline was trained on the earlier 36-episode collection, with the nominal
training episode mixed into refinement. A 1,000-step initial run failed grasping;
a 600-step refinement used a ten-action horizon. Validation selected refinement
step 300. These older results must not be attributed to the new 96-episode data.

| Locked suite | Full task success | Mean placement error |
|---|---:|---:|
| Validation | 3/7 | 3.09 cm |
| Test | 1/7 | 4.67 cm |

The original reconstructed scene achieved a 5.73 cm lift and 3.85 s raised hold,
but its 5.57 cm placement error failed the unchanged 3 cm task tolerance.
All seven test cases passed the other grasp/lift/release/support checks.

## Experimental arm correction

A supervised arm-only correction, with kinematic filtering, reduced original-scene
error to 1.04 cm and passed all task checks. It did not modify finger commands.
It was not reinforcement learning.

| Earlier locked suite | Baseline success | Corrected success |
|---|---:|---:|
| Validation | 3/7 | 1/7 |
| Test | 1/7 | 0/7 |

On seven new paired conditions, success changed from 0/7 to 1/7, while mean error
worsened from 4.35 to 8.24 cm. Those paired conditions kept robot initialization
fixed as case spawn changed; the earlier benchmark adjusted initialization.
Compare within each protocol, not across their success fractions.

The correction remains a source-scene reproduction experiment. The baseline
is retained; broader reliability has not improved.

## Limits and next evaluation

No RL, predictive world model, real-robot test or inference speedup is claimed.
The pelvis is fixed, geometry is approximate and touch is simulated.
Next, train matched 19-feature and 29-feature policies on the new data, select
using validation rollouts and compare on identical locked test conditions.
