# Stage 10 v10a Report - Population Persistent B-only LoRA

Date: 2026-05-04

## Headline

v10a is a negative result.

It produced stronger in-training pressure than v9a, and slightly delayed the
B-only probe, but it did not block full LoRA adaptation and it damaged primary
utility substantially.

```text
Primary ImageNet top-1: 56.54%
Final RIR: 0.657
B-only LoRA-r8 RFD: 2.48, signed RFD: 2.48
LoRA-r8 RFD: 0.98, signed RFD: 0.32
LoRA-r32 RFD: 0.81, signed RFD: 0.50
```

This misses the success bar:

```text
target primary >= 63%
target B-only signed RFD > 10%
observed primary = 56.54%
observed B-only signed RFD = 2.48
```

## Why we ran it

The v9a diagnostic showed attacker-state overfitting:

```text
saved persistent attacker test acc: 36.55%, CE: 6.788
saved-A reset B/head, 10 epochs: mean 33.69%
fresh-A reset B/head, 10 epochs: mean 33.94%
```

So v9a did not damage the saved A basis or fresh A accessibility. It only made
one live B/head state high-loss. v10a tested whether a population of persistent
attackers could prevent that narrow overfit.

## Objective

Population:

```text
4 persistent B-only LoRA attackers
each has independent random A, B/head, optimizer state, and minibatch history
2 attackers updated per outer step
```

Defender block:

```text
L_block = softplus(log(C_H) - min_j CE_H(theta, attacker_j))
```

The defender therefore sees the strongest current population member on each
harmful batch.

## Configuration

```text
experiments/run_immunization_v10_population.py
configs/immunize_v10a_population_bonly.yaml
```

Main settings:

```yaml
run_name: trap_v10a_population_bonly_resnet18_cars
num_attackers: 4
rank: 8
attackers_per_outer: 2
attacker_steps_per_selected: 1
attacker_optimizer: adamw
attacker_lr: 1.0e-3
lambda_block: 0.3
iterations: 2500
eval_every: 0
```

## Immunization Run

Remote:

```text
role-lab GPU 3
logs/v10a_population_bonly_20260504_144545.out
logs/v10a_population_bonly_20260504_144545.err
```

Run completed without NaNs.

| Metric | Value |
|---|---:|
| Final primary acc | 56.54% |
| Final RIR | 0.657 |

Last logged train state:

```text
step=2450
loss_primary=1.176
loss_block=1.604
loss_harm_min=3.899
attacker_acc_min_idx=0.219
```

The population pressure was real: the strongest attacker remained below
random-CE (`log(196) ~= 5.278`) late in training. But the cost was excessive
primary utility damage.

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

| Probe | Baseline final | v10a final | v10a best | RFD | Signed RFD |
|---|---:|---:|---:|---:|---:|
| linear_probe | 41.23% | 43.02% | 43.13% | 4.94 | -2.82 |
| lora_bonly_r8 | 51.19% | 50.75% | 50.75% | 2.48 | 2.48 |
| lora_r8 | 73.85% | 74.22% | 74.59% | 0.98 | 0.32 |
| lora_r32 | 78.15% | 78.87% | 78.87% | 0.81 | 0.50 |
| full_ft_upper | 80.76% | 80.05% | 81.74% | 0.75 | -0.01 |

Result directories:

```text
results/trap_v10a_population_bonly_resnet18_cars/
results/adv_v10a_linear_probe_cars/
results/adv_v10a_lora_bonly_r8_cars/
results/adv_v10a_lora_r8_cars/
results/adv_v10a_lora_r32_cars/
results/adv_v10a_full_ft_upper_cars/
```

## Interpretation

Population pressure helps only slightly, and at the wrong cost.

Compared with v9a:

```text
B-only signed RFD: 0.02 -> 2.48
LoRA-r8 signed RFD: -0.08 -> 0.32
primary: 64.23% -> 56.54%
```

The primary utility drop is much larger than the adaptation slowdown. That is
not a useful immunization tradeoff.

The deeper lesson is that even a population of persistent B-only attackers does
not remove adapter-reachable Cars information. Full LoRA and rank-32 LoRA still
recover to baseline-level performance.

## What This Rules Out

v10a rules out:

```text
A small persistent B-only attacker population is enough to create useful
LoRA robustness while preserving ImageNet utility.
```

It also weakens the overall CE-threshold trajectory-blocking approach:

```text
single attacker: no transfer
population attackers: slight B-only delay, utility collapse, no full-LoRA block
```

## Next Direction

Stop scaling trajectory traps for now.

The next defensible direction is representation-level removal of harmful label
information in adapter-reachable features:

```text
minimize Cars label predictability not only in frozen features,
but also in first-order/LoRA-tangent feature spaces
```

Concrete next experiment:

```text
v11: tangent label-confusion objective
```

Estimate several optimized or random LoRA tangent feature maps and directly
penalize Cars label separability in their span, while preserving ImageNet CE.
This targets the information channel that post-hoc LoRA uses, rather than a
particular attacker trajectory.

