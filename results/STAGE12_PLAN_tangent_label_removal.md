# Stage 12 Plan - Tangent Label Removal

Date: 2026-05-04

## Motivation

Stage 11 established the key mechanism:

```text
Cars labels are recoverable from learned LoRA tangent features.
```

The v9/v10 trajectory objectives did not remove that signal. They changed
specific attacker states but left label information in adapter-reachable
first-order directions.

## Objective

For attacker direction `i`:

```text
T_i(x) = (f_{theta + epsilon * B_i A_i}(x) - f_theta(x)) / epsilon
```

Each persistent tangent attacker learns:

- fixed random LoRA `A_i`
- trainable LoRA direction `B_i`
- a linear harmful classifier `q`

on concatenated tangent features:

```text
q([T_1(x), ..., T_m(x)])
```

The defender blocks the strongest current tangent readout:

```text
L_tangent = softplus(log(C_H) - min_j CE(q_j(T_j(x_H)), y_H))
```

Full defender loss:

```text
L = L_primary
  + lambda_well R_well(H_P)
  + lambda_ill R_ill(H_H)
  + lambda_tangent L_tangent
```

## Implementation

Files:

```text
experiments/run_immunization_v12_tangent_removal.py
configs/immunize_v12a_tangent_removal.yaml
```

v12a settings:

| Setting | Value |
|---|---:|
| attackers | 2 |
| directions per attacker | 3 |
| LoRA rank | 8 |
| epsilon | 1e-2 |
| lambda_tangent | 0.2 |
| iterations | 2500 |

This matches the Stage 11 diagnostic shape: three learned directions give a
1536-dimensional tangent feature.

## Evaluation Plan

1. Compile and smoke test.
2. Run full v12a immunization on role-lab.
3. Evaluate:
   - ImageNet primary top-1
   - RIR
   - RFD against `linear_probe`, `lora_bonly_r8`, `lora_r8`, `lora_r32`,
     `full_ft_upper`
   - Stage 11 learned tangent diagnostic against the v12a checkpoint

## Status

- 2026-05-04: Implemented v12a script and config.
- 2026-05-04: Local compile and config parse passed.
- 2026-05-04: role-lab compile passed.
- 2026-05-04: 5-step smoke passed and wrote
  `results/smoke_v12a_tangent_removal/`.
- 2026-05-04: 100-step behavior smoke passed and wrote
  `results/smoke_v12a_tangent_removal_100/`.
  - step 0: tangent CE 5.279, tangent loss 0.693.
  - step 50: tangent CE 5.226, tangent loss 0.720.
  - This confirms the tangent attacker learns below `log(196) ~= 5.278`
    and activates the defender block.
- 2026-05-04: Full v12a immunization completed:
  - run: `trap_v12a_tangent_removal_resnet18_cars`
  - primary ImageNet top-1: 68.36%.
  - final RIR: 1.111.
- 2026-05-04: RFD probe matrix completed:
  - linear signed RFD: 3.98.
  - B-only LoRA-r8 signed RFD: 0.66.
  - LoRA-r8 signed RFD: 0.06.
  - LoRA-r32 signed RFD: 0.23.
  - full upper signed RFD: 0.07.
- 2026-05-04: Learned-tangent diagnostic completed:
  - v12a learned-tangent probe: 27.93%.
  - baseline learned-tangent probe: 29.14%.
  - v12a reduces the diagnostic by only 1.21pp.
- 2026-05-04: Wrote final report `results/STAGE12_v12a_REPORT.md`.
