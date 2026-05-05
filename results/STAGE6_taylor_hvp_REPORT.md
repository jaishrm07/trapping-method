# Stage 6 Report - Exact Taylor-HVP LoRA trap

Date: 2026-05-04

## Headline

We implemented and ran the exact LoRA Taylor trap using a Hessian-vector
product in LoRA/head parameter space. It is a clean negative result:

- Linear-probe defense disappeared at this setting.
- LoRA-r8 and LoRA-r32 defense remained effectively zero.
- Primary accuracy stayed healthy, so the failure is not from utility collapse.

## What changed

The new trap computes the local Taylor prediction in attacker parameter
space, where phi contains the LoRA factors plus the harmful classifier head:

```text
Delta L_act = L_H(theta, phi_0) - L_H(theta, phi_k)

Delta L_exp =
  -(g_0^T Delta phi + 0.5 Delta phi^T H_phi_phi Delta phi)

L_trap = softplus(Delta L_act - Delta L_exp)
```

The HVP is computed by differentiating `g_0` with `grad_outputs=Delta phi`.
For this run we used the conservative detached-delta variant:

- `inner_create_graph: false`
- `detach_delta: true`
- `detach_phi_k_for_lk: true`

That makes the trap exact in its forward Taylor calculation, but avoids a
full third-order defender gradient through the unrolled LoRA optimization.

## Configuration

- Config: `configs/immunize_multiop_v6_taylor_hvp.yaml`
- Run: `trap_multiop_v6_taylor_hvp_resnet18_cars`
- Operators during immunization: `[linear_probe, lora_r8]`
- Harmful dataset: Cars
- Primary dataset: ImageNet val subset
- Iterations: 2500
- Inner steps: 3
- Inner LR: 0.1
- `lambda_trap: 0.1`
- `lambda_well: 1.0`
- `lambda_ill: 1.0`

## Immunization result

| Metric | Value |
|---|---:|
| Final primary acc | 65.93% |
| Final RIR | 0.505 |
| kappa_H immunized | 32732.65 |
| kappa_H baseline | 68752.80 |
| kappa_P immunized | 18199.43 |
| kappa_P baseline | 19294.92 |

The run was numerically stable. The final trap term had saturated near zero:

```text
step 2450: loss_primary=2.1250, loss_trap=2.94e-33
```

## Adversary probes

All probes used 50 epochs on Cars. RFD below is signed:
`mean_t (baseline_t - immunized_t) / baseline_t * 100`.

| Adversary | Clean baseline final | v6 final | v6 best | Signed RFD |
|---|---:|---:|---:|---:|
| linear_probe | 41.23% | 43.27% | 43.61% | -1.95 |
| lora_r8 | 73.85% | 73.50% | 74.36% | -0.10 |
| lora_r32 | 78.15% | 78.31% | 78.45% | -0.04 |

## Interpretation

The exact Taylor-HVP calculation did not fix the LoRA failure mode. The LoRA
attack recovers essentially the same final accuracy as the clean baseline, and
the linear probe is slightly better than baseline rather than worse.

This suggests the bottleneck is not just that previous LoRA traps lacked the
quadratic Taylor term. More likely, the k=3 SGD inner adversary is too weak or
too short-horizon to approximate the 50-epoch LoRA attacker, even with the
local Hessian correction.

## Files

- `src/trap_loss.py` - `trap_loss_lora_taylor_hvp`
- `experiments/run_immunization_cn.py` - config wiring for Taylor-HVP flags
- `configs/immunize_multiop_v6_taylor_hvp.yaml` - v6 run config
- `results/trap_multiop_v6_taylor_hvp_resnet18_cars/results.json`
- `results/adv_v6_taylor_hvp_linear_probe_cars/results.json`
- `results/adv_v6_taylor_hvp_lora_r8_cars/results.json`
- `results/adv_v6_taylor_hvp_lora_r32_cars/results.json`

## Next fork

The next useful test should change the attacker model, not only the Taylor
accounting. Two defensible options:

1. Run a stronger inner LoRA adversary: larger k, Adam-style update, or a
   short persistent inner state instead of fresh random LoRA factors every
   defender step.
2. Replace the trajectory trap with an optimum-oriented objective, such as
   penalizing harmful accuracy or harmful loss after a stronger inner solve.

The exact HVP trap is still valuable as an ablation: it rules out "missing
quadratic Taylor geometry" as the main explanation for the LoRA failure.
