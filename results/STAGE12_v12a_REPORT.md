# Stage 12 v12a Report - Tangent Label Removal

Date: 2026-05-04

## Headline

v12a is a useful negative result.

It directly targeted learned LoRA-tangent label leakage, preserved primary
ImageNet utility better than previous attempts, but did not meaningfully block
post-hoc LoRA adaptation or remove the learned tangent signal.

```text
Primary ImageNet top-1: 68.36%
Final RIR: 1.111
LoRA-r8 signed RFD: 0.06
LoRA-r32 signed RFD: 0.23
Learned-tangent probe: 27.93%
```

## Why We Ran It

Stage 11 showed the mechanism:

```text
Cars labels are readable from learned LoRA tangent features:
(f_{theta + epsilon * BA}(x) - f_theta(x)) / epsilon
```

v9/v10 attacked the adapted endpoint or persistent attacker state. v12a instead
attacks the tangent readout itself.

## Objective

For each persistent tangent attacker:

```text
T_i(x) = (f_{theta + epsilon * B_i A_i}(x) - f_theta(x)) / epsilon
```

Each attacker owns:

- 3 fixed random LoRA `A_i` bases
- 3 trainable LoRA `B_i` directions
- one linear head over the concatenated 1536-d tangent feature

The defender blocks the strongest current tangent readout:

```text
L_tangent = softplus(log(C_H) - min_j CE(q_j([T_1, T_2, T_3]), y_H))
```

Full loss:

```text
L = L_primary
  + lambda_well R_well(H_P)
  + lambda_ill R_ill(H_H)
  + lambda_tangent L_tangent
```

## Implementation

```text
experiments/run_immunization_v12_tangent_removal.py
configs/immunize_v12a_tangent_removal.yaml
```

Main settings:

| Setting | Value |
|---|---:|
| attackers | 2 |
| directions per attacker | 3 |
| LoRA rank | 8 |
| epsilon | 1e-2 |
| lambda_tangent | 0.2 |
| iterations | 2500 |

## Smoke Tests

5-step smoke passed:

```text
results/smoke_v12a_tangent_removal/
```

100-step behavior smoke passed:

| Step | Min tangent CE | Tangent loss | Tangent acc |
|---:|---:|---:|---:|
| 0 | 5.279 | 0.693 | 1.56% |
| 50 | 5.226 | 0.720 | 3.12% |

This confirmed the tangent attacker learns below `log(196) ~= 5.278` and
activates the defender block.

## Immunization Run

Remote:

```text
role-lab GPU 5
logs/v12a_tangent_removal_20260504_181500.out
logs/v12a_tangent_removal_20260504_181500.err
```

Final artifacts:

```text
results/trap_v12a_tangent_removal_resnet18_cars/
```

Run completed without NaNs.

| Metric | Value |
|---|---:|
| Final primary acc | 68.36% |
| Final RIR | 1.111 |

Last logged train state:

| Step | Min tangent CE | Tangent loss | Tangent acc | Primary loss |
|---:|---:|---:|---:|---:|
| 2250 | 3.220 | 2.178 | 26.6% | 1.238 |
| 2300 | 3.353 | 2.061 | 29.7% | 1.152 |
| 2350 | 3.358 | 2.057 | 32.8% | 1.037 |
| 2400 | 3.608 | 1.842 | 20.3% | 1.242 |
| 2450 | 3.268 | 2.135 | 28.1% | 1.107 |

The in-training tangent adversary remained strong. The defender did not raise
its CE back toward random.

## Probe Matrix

All probes ran for 50 epochs on Cars.

RFD:

```text
mean_t |baseline_t - immunized_t| / baseline_t * 100
```

Signed RFD:

```text
mean_t (baseline_t - immunized_t) / baseline_t * 100
```

| Probe | Baseline final | v12a final | v12a best | RFD | Signed RFD |
|---|---:|---:|---:|---:|---:|
| linear_probe | 41.23% | 40.70% | 41.00% | 4.03 | 3.98 |
| lora_bonly_r8 | 51.19% | 51.08% | 51.10% | 0.66 | 0.66 |
| lora_r8 | 73.85% | 73.85% | 74.07% | 0.37 | 0.06 |
| lora_r32 | 78.15% | 78.57% | 78.57% | 0.40 | 0.23 |
| full_ft_upper | 80.76% | 80.15% | 81.54% | 0.58 | 0.07 |

Result directories:

```text
results/adv_v12a_linear_probe_cars/
results/adv_v12a_lora_bonly_r8_cars/
results/adv_v12a_lora_r8_cars/
results/adv_v12a_lora_r32_cars/
results/adv_v12a_full_ft_upper_cars/
```

## Learned-Tangent Diagnostic

Run:

```text
results/learned_tangent_v12a_r8_k3_e10_p100/
```

Learned B-only direction quality:

| Model | Seed 42 | Seed 43 | Seed 44 | Mean |
|---|---:|---:|---:|---:|
| baseline | 35.99% | 37.35% | 36.30% | 36.55% |
| v9a | 37.38% | 39.00% | 37.36% | 37.91% |
| v10a | 37.63% | 38.35% | 37.33% | 37.77% |
| v12a | 35.57% | 36.94% | 35.83% | 36.11% |

Probe results:

| Model | Base | Learned tangent | Base + learned tangent |
|---|---:|---:|---:|
| baseline | 33.96% | 29.14% | 32.36% |
| v9a | 33.62% | 26.03% | 30.36% |
| v10a | 31.99% | 25.52% | 29.97% |
| v12a | 32.88% | 27.93% | 31.69% |

v12a reduces learned-tangent accuracy by only 1.21pp relative to baseline. It
does less tangent-signal suppression than v9a/v10a, despite optimizing a direct
tangent objective.

## Interpretation

v12a tells us the issue is not just endpoint-vs-tangent mismatch.

The live training objective did create a strong tangent attacker, but the
defender failed to erase the broad class signal. Fresh post-hoc tangent
directions still reached the same direction-learning band, and LoRA-r8/r32
adaptation recovered almost exactly to baseline.

The likely failure mode is operator overfitting again, but at the tangent level:

```text
The defender pushes against two live tangent heads/direction sets.
Fresh LoRA directions still find nearby label-informative tangent subspaces.
```

This suggests the tangent label signal is broad in the local adapter tangent
kernel, not concentrated in a small number of live directions.

## Conclusion

v12a is not a successful defense, but it sharpens the research claim:

> Directly blocking a small population of learned tangent readouts is still too
> narrow. LoRA-accessible Cars signal appears distributed across many tangent
> directions.

The next useful step is not more endpoint trapping. It is a broader tangent
kernel objective, for example:

```text
min_theta L_primary
        + lambda * max_{A,B,q in a larger tangent family}
          softplus(log(C_H) - CE(q(T_{A,B}(x_H)), y_H))
```

with either a larger attacker ensemble, more inner steps per defender step, or a
closed-form/readout upper bound on tangent features instead of a small live
linear head population.
