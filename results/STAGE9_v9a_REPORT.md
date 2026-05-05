# Stage 9 v9a Report - Persistent B-only LoRA Attacker

Date: 2026-05-04

## Headline

v9a is a hard negative result for persistent single-attacker immunization.

The full run completed stably and preserved primary utility:

```text
Primary ImageNet top-1: 64.23%
Final RIR: 0.783
```

But post-hoc 50-epoch Cars probes were not blocked:

```text
B-only LoRA-r8 RFD:  0.32%  signed RFD:  0.02%
LoRA-r8 RFD:         0.39%  signed RFD: -0.08%
LoRA-r32 RFD:        0.41%  signed RFD:  0.07%
Linear-probe RFD:    5.11%  signed RFD: -4.46%
```

The persistent attacker objective succeeded locally during immunization, but
did not transfer to a freshly initialized downstream attacker.

## Why we ran it

v8a failed with a fresh short-horizon B-only inner loop:

```text
k=10 B-only CE blocking -> B-only RFD 0.20, LoRA-r8 RFD 0.08
```

The suspected failure mode was horizon mismatch:

```text
defender sees: fresh short inner attacker
evaluation uses: 50-epoch trained attacker
```

v9a removed that mismatch by keeping the B-only LoRA/head attacker alive across
defender steps.

## Objective

Persistent attacker state:

```text
A: fixed random LoRA basis
B: persistent trainable LoRA B factors
omega, bias: persistent harmful classifier head
```

Alternating loop:

```text
1. Attacker update:
   minimize CE_H(theta, A, B, omega, bias)
   over B, omega, bias

2. Defender update:
   minimize primary CE
   + condition-number regularizers
   + lambda_block * softplus(log(C_H) - CE_H(theta, current attacker))
   over theta
```

For Cars:

```text
C_H = 196
log(C_H) ~= 5.278
```

## Configuration

```text
configs/immunize_v9a_persistent_bonly.yaml
experiments/run_immunization_v9_persistent.py
```

Main settings:

```yaml
run_name: trap_v9a_persistent_bonly_resnet18_cars
rank: 8
attacker_steps_per_outer: 3
attacker_optimizer: adamw
attacker_lr: 1.0e-3
lambda_block: 0.3
ce_threshold: null
iterations: 2500
eval_every: 500
```

## Immunization Run

Remote:

```text
role-lab, GPU 1
logs/v9a_persistent_bonly_20260504_122919.out
logs/v9a_persistent_bonly_20260504_122919.err
```

Run completed without NaNs.

| Metric | Value |
|---|---:|
| Final primary acc | 64.23% |
| Final RIR | 0.783 |

Intermediate RIR:

| Step | RIR | kappa_H immu | kappa_P immu |
|---:|---:|---:|---:|
| 500 | 0.549 | 2.195e10 | 3.087e10 |
| 1000 | 0.789 | 3.009e10 | 3.258e10 |
| 1500 | 0.721 | 2.865e10 | 3.623e10 |
| 2000 | 0.942 | 3.472e10 | 2.962e10 |
| final | 0.783 | 3.027e10 | 3.523e10 |

Last logged train entry:

```text
step=2450
loss_primary=0.948
loss_block=0.022
loss_harm_current=9.091
attacker_acc_last=1.000
attacker_acc_defender_batch=0.3125
```

This means the persistent attacker could still fit its own minibatches, but
the defender made the current attacker's evaluated CE high on the defender
harmful batch.

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

| Probe | Baseline final | v9a final | v9a best | RFD | Signed RFD |
|---|---:|---:|---:|---:|---:|
| linear_probe | 41.23% | 42.48% | 43.35% | 5.11 | -4.46 |
| lora_bonly_r8 | 51.19% | 51.18% | 51.22% | 0.32 | 0.02 |
| lora_r8 | 73.85% | 73.32% | 74.06% | 0.39 | -0.08 |
| lora_r32 | 78.15% | 78.76% | 78.76% | 0.41 | 0.07 |
| full_ft_upper | 80.76% | 79.72% | 81.62% | 0.68 | -0.16 |

Result directories:

```text
results/trap_v9a_persistent_bonly_resnet18_cars/
results/adv_v9a_linear_probe_cars/
results/adv_v9a_lora_bonly_r8_cars/
results/adv_v9a_lora_r8_cars/
results/adv_v9a_lora_r32_cars/
results/adv_v9a_full_ft_upper_cars/
```

## Interpretation

v9a fixes the wrong issue.

It solves the short-horizon problem for one persistent attacker instance, but
the final released model is still easy for a fresh attacker to adapt. The
failure is therefore not just:

```text
fresh k=10 inner loop is too short
```

It is more specifically:

```text
single-attacker-state blocking does not generalize across attacker
initializations, minibatch order, and long-horizon optimization paths.
```

The defender appears to overfit to the current B/head state. A new B/head state
finds a normal Cars solution.

This also explains why the in-training harmful CE looked promising while the
post-hoc probes did not move. The model learned to make one live attacker's
current classifier bad; it did not remove the broader adapter-accessible
information about Cars.

## What This Rules Out

v9a rules out:

```text
A single persistent B-only LoRA attacker state is enough to produce
post-hoc B-only LoRA robustness.
```

Together with v8a, it also weakens the broader idea that CE-threshold blocking
alone is sufficient, even when the attacker is persistent.

It does not rule out:

```text
population-based persistent attackers
attacker replay buffers
reset-resistant adversarial immunization
implicit near-optimum attacker objectives
tangent-kernel label-alignment penalties
harmful-information removal objectives
```

## Next Direction

The next defense should target the attacker solution set, not one attacker
trajectory.

Most plausible next experiment:

```text
v10: population/replay persistent B-only LoRA immunization
```

Maintain several independent B-only attackers with different random A bases,
heads, seeds, and minibatch histories. On each defender step, update a subset
of attackers and block the strongest current harmful attacker:

```text
L_block = softplus(log(C_H) - min_j CE_H(theta, attacker_j))
```

This tests whether v9a failed because it overfit to one attacker state.

If v10 also fails, the evidence will point away from trajectory trapping and
toward representation-level objectives:

```text
reduce Cars label information in adapter-reachable tangent features
while preserving ImageNet utility
```
