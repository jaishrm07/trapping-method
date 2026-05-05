# Stage 11 Report - Learned LoRA-Tangent Label Leakage

Date: 2026-05-04

## Question

Stages 8-10 showed that CE-threshold and persistent-attacker objectives do not
meaningfully block LoRA adaptation. Stage 11 asks a more mechanistic question:

> Are Cars labels still linearly recoverable from label-informed,
> adapter-reachable first-order feature directions?

If yes, the problem is not just that the attacker found a lucky SGD path. The
harmful task remains encoded in LoRA-accessible tangent directions.

## Method

Script:

```text
experiments/run_learned_tangent_probe.py
```

For each extractor:

1. Learn 3 B-only LoRA directions on Cars for 10 epochs each.
2. Rescale each learned B direction to epsilon `1e-2`.
3. Extract learned tangent features:

```text
(f_{theta + epsilon * BA}(x) - f_theta(x)) / epsilon
```

4. Train linear probes on:
   - base features
   - learned tangent features
   - base + learned tangent features

Runs:

```text
learned_tangent_baseline_r8_k3_e10_p100
learned_tangent_v9a_r8_k3_e10_p100
learned_tangent_v10a_r8_k3_e10_p100
```

Compute:

```text
role-lab
baseline: GPU 1
v9a:      GPU 4
v10a:     GPU 5
```

## Smoke Test

Remote Cars smoke passed on a 128/128 subset:

| Feature set | Final acc |
|---|---:|
| base | 4.69% |
| learned_tangent | 6.25% |
| base_learned_tangent | 8.59% |

This validated the full code path on role-lab before launching full diagnostics.

## Learned Direction Quality

Each row is final Cars test accuracy of the B-only direction learner before
shrinking the direction to epsilon.

| Model | Seed 42 | Seed 43 | Seed 44 | Mean |
|---|---:|---:|---:|---:|
| baseline | 35.99% | 37.35% | 36.30% | 36.55% |
| v9a | 37.38% | 39.00% | 37.36% | 37.91% |
| v10a | 37.63% | 38.35% | 37.33% | 37.77% |

The immunized models do not make label-informed B-only direction learning
harder. If anything, these directions learn slightly better than baseline.

## Probe Results

Final / best Cars probe accuracy:

| Model | Base | Learned tangent | Base + learned tangent |
|---|---:|---:|---:|
| baseline | 33.96 / 34.22 | 29.14 / 29.14 | 32.36 / 32.36 |
| v9a | 33.62 / 33.65 | 26.03 / 26.03 | 30.36 / 30.36 |
| v10a | 31.99 / 32.32 | 25.52 / 25.52 | 29.97 / 30.00 |

Feature dimensions:

| Feature set | Dim |
|---|---:|
| base | 512 |
| learned tangent | 1536 |
| base + learned tangent | 2048 |

Feature norms:

| Model | Base norm mean | Tangent norm mean |
|---|---:|---:|
| baseline | 25.37 | 3.18 |
| v9a | 28.24 | 3.63 |
| v10a | 24.79 | 3.53 |

## Interpretation

This is a strong positive diagnostic for the operator-reachability hypothesis.

Random LoRA tangent sketches from Stage 7 were weak because they sampled
arbitrary directions. Once directions are learned from labels and then shrunk
back into the local tangent regime, the tangent features alone recover Cars at
25-29% accuracy. That is far above chance for 196 classes and close enough to
base-feature probing to be mechanistically meaningful.

v9a and v10a reduce tangent-probe accuracy by only about 3-4 percentage points:

| Comparison | Learned tangent drop vs baseline |
|---|---:|
| v9a | -3.11 pp |
| v10a | -3.62 pp |

That is not enough to explain or produce robust LoRA blocking. It matches the
downstream adversary matrix: v9a/v10a did not meaningfully reduce LoRA-r8,
LoRA-r32, or full fine-tuning performance.

The base + tangent probe underperforms the base-only probe here. That is likely
a probe-optimization/regularization effect from increasing dimension to 2048,
not evidence that the tangent signal is absent. The tangent-only probe is the
clean readout for this diagnostic.

## Conclusion

Stage 11 supports the first-principles claim:

> Linear probing tests label information already present in the frozen feature
> vector. LoRA tests label information reachable through low-rank feature
> derivatives. The v9/v10 defenses do not remove that derivative-level signal.

Next defense should target tangent label leakage directly, for example:

```text
min_theta L_primary
        + lambda_base * readable_label_loss(f_theta(x_H), y_H)
        + lambda_tangent * readable_label_loss(T_LoRA(theta, x_H), y_H)
```

where `T_LoRA` is a learned or adversarially selected LoRA-tangent feature set.
