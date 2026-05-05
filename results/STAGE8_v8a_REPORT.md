# Stage 8 v8a Report - B-only LoRA-aware immunization

Date: 2026-05-04

## Headline

v8a is a clean negative result.

We directly immunized against the Stage 7 bypass operator:

```text
LoRA A frozen random
LoRA B trainable
harmful classifier head trainable
```

The run completed stably and preserved primary utility, but it did not move the
B-only LoRA probe or full LoRA probes.

Key result:

```text
B-only LoRA-r8 signed RFD: 0.20%
LoRA-r8 signed RFD:        0.08%
Linear-probe signed RFD:   5.85%
Primary ImageNet top-1:    64.09%
```

So direct post-attack CE blocking with a short k=10 B-only inner loop is not
enough.

## Why we ran it

Stage 7 showed:

| Operator | v4a final |
|---|---:|
| linear_probe | 21.46% |
| lora_bonly_r8 | 51.10% |
| lora_r8 | 73.01% |

This suggested the linear-probe trap failed because it only damaged frozen
feature readout, not the adapter-basis reachable space.

v8a tested the smallest direct fix: train the defender against the B-only
adapter-basis attacker itself.

## Objective

Inner attacker:

```text
random LoRA A, frozen
zero LoRA B, trainable
harmful head, trainable
k = 10 SGD steps on Cars
eta = 0.01
```

Outer loss:

```text
L_block = softplus(log(C_H) - CE_H(theta, B_k, head_k))
```

For Cars:

```text
C_H = 196
log(C_H) ~= 5.28
```

Interpretation:

If the adapted harmful classifier gets CE below random-guess CE, the defender
is penalized. This avoids unbounded CE maximization while directly targeting
post-attack harmful adaptation.

## Configuration

Config:

```text
configs/immunize_v8a_bonly.yaml
```

Main settings:

```yaml
run_name: trap_v8a_bonly_resnet18_cars
trap_operators:
  - linear_probe
  - lora_bonly_r8
trap_k_inner: 10
trap_eta_inner: 0.01
lambda_trap: 0.3
bonly_ce_threshold: null  # log(num_classes)
bonly_inner_create_graph: false
bonly_detach_inner_updates: true
use_k_inv_preconditioner: true
k_inv_ridge: 1.0e-2
iterations: 2500
```

## Immunization run

Remote:

```text
role-lab, GPU 1
logs/v8a_bonly_20260504_101609.out
logs/v8a_bonly_20260504_101609.err
```

Run completed without NaNs.

| Metric | Value |
|---|---:|
| Final primary acc | 64.09% |
| Final RIR | 0.724 |
| kappa_H immunized | 66838.27 |
| kappa_H baseline | 68752.80 |
| kappa_P immunized | 25893.49 |
| kappa_P baseline | 19294.92 |

Intermediate RIR:

| Step | RIR | kappa_H immu | kappa_P immu |
|---:|---:|---:|---:|
| 500 | 0.316 | 30954.46 | 25812.31 |
| 1000 | 0.397 | 30420.23 | 20941.97 |
| 1500 | 0.415 | 31830.36 | 20548.29 |
| 2000 | 0.465 | 37677.89 | 22095.58 |
| final | 0.724 | 66838.27 | 25893.49 |

Last logged train entry:

```text
step=2450
loss_primary=2.2024
loss_well=0.1764
loss_ill=0.0048
loss_trap=0.1455
```

The B-only trap signal was active during training and sometimes hit the clamp,
but that did not translate into downstream robustness.

## Probe matrix

All probes ran for 50 epochs on Cars.

Signed RFD:

```text
mean_t (baseline_t - immunized_t) / baseline_t * 100
```

| Probe | Baseline final | v8a final | v8a best | Signed RFD |
|---|---:|---:|---:|---:|
| linear_probe | 41.23% | 41.00% | 41.00% | 5.85 |
| lora_bonly_r8 | 51.19% | 51.25% | 51.25% | 0.20 |
| lora_r8 | 73.85% | 73.98% | 73.98% | 0.08 |
| lora_r32 | 78.15% | 78.37% | 78.37% | 0.19 |
| full_ft_upper | 80.76% | 80.60% | 81.45% | 0.14 |

Result directories:

```text
results/trap_v8a_bonly_resnet18_cars/
results/adv_v8a_linear_probe_cars/
results/adv_v8a_lora_bonly_r8_cars/
results/adv_v8a_lora_r8_cars/
results/adv_v8a_lora_r32_cars/
results/adv_v8a_full_ft_upper_cars/
```

## Interpretation

The Stage 7 principle still looks right:

```text
Linear-probe immunization does not defend the adapter-basis reachable space.
```

But v8a tells us the naive direct fix is insufficient:

```text
k=10 B-only inner CE blocking does not change the 50-epoch B-only attacker.
```

The most likely reason is horizon mismatch.

The v8a inner attacker only simulates 10 mini-batch SGD updates from a fresh
random adapter basis. The evaluation attacker trains for 50 full epochs over
Cars. The defender is therefore optimizing against an early local condition,
while the attacker succeeds through long-horizon supervised adaptation.

This is consistent with the whole sequence:

| Stage | Mechanism | LoRA/B-only result |
|---|---|---|
| v4a | FOMAML trap | full LoRA RFD near zero |
| v6 | exact Taylor-HVP | full LoRA RFD near zero |
| v8a | B-only post-attack CE block | B-only and full LoRA RFD near zero |

The failure is no longer "wrong Taylor geometry" or "wrong operator branch."
It is probably that short unrolled inner loops are too weak a surrogate for the
downstream adaptation process.

## What this rules out

v8a rules out:

```text
A short k=10 B-only CE-threshold block loss is enough to immunize the adapter basis.
```

It does not rule out:

```text
Long-horizon B-only adversarial immunization
persistent inner states
implicit/near-optimum LoRA objectives
tangent-kernel label-alignment objectives
```

## Next direction

The next fork should stop sampling fresh short inner attacks.

Two viable directions:

1. Persistent B-only adversary state

Maintain B/head states across defender steps, like adversarial training against
a continuously trained attacker. This better matches the 50-epoch probe.

```text
theta_t update sees attacker phi_t
attacker phi_t keeps training across outer steps
```

2. Adapter tangent-kernel alignment

Avoid unrolling the attacker. Estimate whether harmful labels align with the
B-only LoRA tangent kernel:

```text
K_B(i,j) = <d logits(x_i)/dB, d logits(x_j)/dB>
```

Then penalize harmful label alignment directly. This attacks accessibility as a
representation property rather than a k-step training trajectory.

Recommendation:

Move to tangent-kernel alignment or persistent-attacker training. More short
fresh-inner-loop variants are unlikely to move the metric.

## Files changed for v8a

- `src/trap_loss.py`
- `experiments/run_immunization_cn.py`
- `configs/immunize_v8a_bonly.yaml`
- `results/STAGE8_PLAN_bonly_immunization.md`
- `results/STAGE8_v8a_IMPLEMENTATION_LOG.md`
- `results/STAGE8_v8a_REPORT.md`
