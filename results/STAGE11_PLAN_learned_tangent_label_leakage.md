# Stage 11 Plan - Learned LoRA-Tangent Label Leakage

## Motivation

Stages 8-10 tried to block B-only and LoRA adaptation by attacking the SGD
trajectory itself. The results were negative:

- v8a B-only CE block preserved ImageNet reasonably but did not slow LoRA.
- v9a single persistent attacker damaged one live attacker state, but fresh
  LoRA restarts still recovered.
- v10a population persistent attackers gave a small B-only delay but damaged
  ImageNet utility and still did not block full LoRA.

The working conclusion is that trajectory-specific CE thresholds are not the
right abstraction. We need to identify what information LoRA can still access
that linear probing cannot.

## First-Principles Hypothesis

Linear probing succeeds when harmful labels are already linearly readable from
the frozen feature vector. LoRA succeeds even when that direct readout is weak
because it can move the feature extractor inside a local adapter tangent space:

```text
f_{theta + BA}(x) ~= f_theta(x) + J_theta(x)[BA]
```

So the harmful task may be encoded not only in the base feature `f_theta(x)`,
but also in low-rank, adapter-reachable derivatives of that feature.

If true, immunization needs to remove or scramble harmful label information in
the LoRA tangent space, not only make one optimizer path inconvenient.

## Diagnostic

Script:

```text
experiments/run_learned_tangent_probe.py
```

Role-lab launcher:

```text
scripts/launch_stage11_learned_tangent_rolelab.sh
```

For each model:

1. Learn several label-informed B-only LoRA directions on Cars.
2. Shrink each learned direction to a small epsilon-norm perturbation.
3. Extract:
   - base features: `f_theta(x)`
   - learned tangent features: `(f_{theta + epsilon * BA}(x) - f_theta(x)) / epsilon`
   - concatenated base + tangent features
4. Train linear probes on each feature set.

This tests whether label signal is present in adapter-reachable first-order
directions.

## Planned Comparisons

- Baseline ImageNet-pretrained ResNet18.
- v9a persistent B-only immunized extractor.
- v10a population persistent B-only immunized extractor.

Initial command shape:

```bash
CUDA_VISIBLE_DEVICES=<gpu> /home/jaisharma/miniconda3/envs/trap/bin/python \
  experiments/run_learned_tangent_probe.py \
  --dataset cars \
  --rank 8 \
  --num-directions 3 \
  --direction-epochs 10 \
  --probe-epochs 100 \
  --run-name learned_tangent_<model>_r8_k3_e10_p100
```

Checkpoint variants add:

```bash
--extractor-checkpoint results/<run>/extractor.pt
```

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=<gpu> /home/jaisharma/miniconda3/envs/trap/bin/python \
  experiments/run_learned_tangent_probe.py \
  --dataset cars \
  --num-workers 0 \
  --train-limit 128 \
  --test-limit 128 \
  --feature-batch-size 16 \
  --probe-batch-size 32 \
  --num-directions 1 \
  --direction-epochs 1 \
  --probe-epochs 2 \
  --run-name smoke_learned_tangent_cars
```

## Success/Failure Readout

Important values:

- `base` probe accuracy: ordinary linear readability.
- `learned_tangent` probe accuracy: harmful label signal in LoRA tangent space.
- `base_learned_tangent` probe accuracy: combined direct and adapter-reachable
  signal.

Expected useful outcome:

- If tangent accuracy is high for baseline and remains high for v9/v10, then
  the failed defenses did not remove LoRA-accessible label signal.
- If a future immunization reduces tangent accuracy while preserving ImageNet,
  that is a stronger mechanistic target than one persistent attacker loss.

## Status

- 2026-05-04: Implemented `experiments/run_learned_tangent_probe.py`.
- 2026-05-04: Local `py_compile` passed.
- 2026-05-04: Added `--train-limit`, `--test-limit`, and
  `--synthetic-smoke` for fast runtime validation.
- 2026-05-04: Local synthetic smoke passed and wrote
  `results/smoke_synthetic_learned_tangent/results.json`.
- 2026-05-04: Local real-Cars smoke is blocked because the local Python
  environment does not have the HuggingFace `datasets` package.
- 2026-05-04: Added role-lab launcher
  `scripts/launch_stage11_learned_tangent_rolelab.sh`; shell syntax check
  passed.
- 2026-05-04: `role-lab.ece.vt.edu` SSH check timed out; smoke and full runs
  are ready to launch once the server is reachable.
- 2026-05-04: Retried role-lab. SSH reachable. Remote Cars smoke passed:
  base 4.69%, learned_tangent 6.25%, base_learned_tangent 8.59% on the
  128/128 smoke subset.
- 2026-05-04: Launched full diagnostics:
  - `learned_tangent_baseline_r8_k3_e10_p100` on GPU 1.
  - `learned_tangent_v9a_r8_k3_e10_p100` on GPU 4.
  - `learned_tangent_v10a_r8_k3_e10_p100` on GPU 5.
- 2026-05-04: Patched launcher to make full-run GPU IDs configurable and to
  use `setsid -f` for clean SSH detachment.
- 2026-05-04: Full diagnostics completed and synced locally. Final
  learned-tangent probe accuracies:
  - baseline: 29.14%.
  - v9a: 26.03%.
  - v10a: 25.52%.
- 2026-05-04: Wrote final report
  `results/STAGE11_learned_tangent_REPORT.md`.
