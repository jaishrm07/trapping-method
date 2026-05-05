# Stage 9 Plan - Persistent B-only LoRA attacker

Date: 2026-05-04

## Why v9 exists

The repeated negative result is now clear:

| Stage | Mechanism | Result |
|---|---|---|
| v4a | FOMAML LoRA trap | LoRA RFD near zero |
| v6 | exact Taylor-HVP LoRA trap | LoRA RFD near zero |
| Stage 7 | B-only diagnostic | fixed-A adapter basis bypasses LP defense |
| v8a | k=10 B-only CE block | B-only RFD 0.20, LoRA-r8 RFD 0.08 |

The likely failure mode is horizon mismatch:

```text
defender training: fresh short inner attacker
evaluation: long-horizon 50-epoch attacker
```

v9 removes the fresh-inner-loop assumption. The attacker state persists across
outer defender steps and keeps training.

## Core idea

Maintain persistent attacker parameters:

```text
A: fixed random LoRA basis
B: persistent trainable LoRA B factors
omega, bias: persistent harmful classifier head
```

Each outer iteration alternates:

```text
1. Attacker update:
   minimize CE_H(theta, A, B, omega, bias)
   over B, omega, bias

2. Defender update:
   minimize primary utility loss
   + condition-number regularizers
   + softplus(log(C_H) - CE_H(theta, A, B_persist, omega_persist, bias_persist))
   over theta
```

The defender therefore sees a trained adapter, not a fresh local probe.

## v9a initial design

Threat operator:

```text
lora_bonly_r8 persistent
```

Initial config:

```yaml
run_name: trap_v9a_persistent_bonly_resnet18_cars
rank: 8
attacker_steps_per_outer: 3
attacker_optimizer: adamw
attacker_lr: 1.0e-3
attacker_weight_decay: 0.0
lambda_block: 0.3
ce_threshold: null  # log(C_H)
lambda_well: 1.0
lambda_ill: 1.0
use_k_inv_preconditioner: true
iterations: 2500
```

## Evaluation matrix

After immunization:

```text
linear_probe
lora_bonly_r8
lora_r8
lora_r32
full_ft_upper
```

## Success criteria

Minimum useful result:

```text
B-only LoRA-r8 signed RFD > 10%
primary ImageNet top-1 >= 63%
```

Strong result:

```text
B-only LoRA-r8 signed RFD > 10%
LoRA-r8 signed RFD clearly above the ~1% ceiling
primary ImageNet top-1 >= 63%
```

Negative but useful:

```text
B-only RFD improves, full LoRA-r8 remains near zero
```

That would mean fixed-A adapter accessibility is defensible, but trainable A is
a separate bypass.

Hard negative:

```text
B-only RFD remains near zero
```

Then persistent supervised adaptation is still enough to route around the
defender, and the next direction should be tangent-kernel label alignment or a
data-distribution-level information removal objective.

## Implementation checklist

1. Done: add `experiments/run_immunization_v9_persistent.py`.
2. Done: add `configs/immunize_v9a_persistent_bonly.yaml`.
3. Done: run local compile checks.
4. Done: sync to role-lab.
5. Done: run remote smoke tests with 5 and 100 iterations.
6. Done: launch full v9a.
7. Done: probe matrix.
8. Done: write `results/STAGE9_v9a_REPORT.md`.

Implementation and smoke-test details:

```text
results/STAGE9_v9a_IMPLEMENTATION_LOG.md
```

Full v9a result:

```text
results/trap_v9a_persistent_bonly_resnet18_cars/
final primary ImageNet top-1: 64.23%
final RIR: 0.783
```

Probe result:

```text
lora_bonly_r8 RFD: 0.32, signed RFD: 0.02
lora_r8 RFD: 0.39, signed RFD: -0.08
```

Conclusion: hard negative for single persistent-attacker immunization.
