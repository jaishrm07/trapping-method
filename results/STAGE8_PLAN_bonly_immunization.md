# Stage 8 Plan - B-only LoRA-aware immunization

Date: 2026-05-04

## Why this is the next step

Stage 7 isolated the failure mode:

| Operator | Baseline final | v4a final | v4a signed RFD |
|---|---:|---:|---:|
| linear_probe | 41.23% | 21.46% | 55.98 |
| lora_bonly_r8 | 51.19% | 51.10% | 0.71 |
| lora_r8 | 73.85% | 73.01% | 0.28 |

v4a damages frozen-feature readout, but it does not damage the fixed LoRA-A
adapter basis. A weak attacker that freezes A and trains only B plus the head
still reaches about 51% Cars accuracy.

Therefore the next defense should directly immunize against the operator that
bypassed v4a:

```text
LoRA A frozen random
LoRA B trainable
harmful classifier head trainable
```

This is the smallest operator-aware step between linear probing and full LoRA.

## Core hypothesis

The current trap family fails under LoRA because it protects:

```text
F_LP(theta) = { W f_theta(x) }
```

but not:

```text
F_B(theta, A) = { W f_{theta + B A}(x) : B trainable, A fixed random }
```

Stage 8 tests whether explicitly defending `F_B(theta, A)` moves the LoRA
robustness ceiling.

## Proposed v8a objective

Use a direct post-attack harmful-loss objective rather than another Taylor trap.

Inner attacker:

```text
Initialize random LoRA A, zero B, random harmful head.
Freeze A.
Train B and head for k steps on D_H.
```

Outer defender:

```text
min_theta
    L_primary(theta)
  + lambda_bonly * L_block(theta, B_k, head_k)
  + lambda_well * R_well(H_P)
  + lambda_ill * R_ill(H_H)
```

where:

```text
L_block = softplus(log(C_H) - CE_H(theta, B_k, head_k))
```

For Cars:

```text
C_H = 196
log(C_H) ~= 5.28
```

Interpretation:

If the adapted harmful classifier achieves CE lower than random-guess CE, the
defender is penalized. This avoids unbounded CE maximization while still
targeting successful harmful adaptation.

## Initial v8a configuration

Start conservative:

```yaml
run_name: trap_v8a_bonly_resnet18_cars
trap_operators:
  - linear_probe
  - lora_bonly_r8
lora_rank: 8
bonly_freeze_A: true
trap_k_inner: 10
trap_eta_inner: 0.01  # also test 0.03 if stable
lambda_bonly: 0.3
lambda_trap_lp: 0.3
lambda_well: 1.0
lambda_ill: 1.0
grad_clip: 1.0
iterations: 2500
```

Why `k=10`:

The B-only attacker is weaker and more stable than full LoRA, and Stage 7 showed
it needs many epochs to climb. A `k=3` inner loop is probably too short to teach
the defender the real failure mode.

## Evaluation matrix

After immunization, run:

| Probe | Purpose |
|---|---|
| `linear_probe` | Ensure LP defense does not disappear |
| `lora_bonly_r8` | Direct target operator |
| `lora_r8` | Test transfer from B-only to full LoRA |
| `lora_r32` | Rank extrapolation / stronger adapter |
| `full_ft_upper` | Check whether defense transfers to partial full FT |

Optional:

| Probe | Purpose |
|---|---|
| `lora_bonly_r4`, `lora_bonly_r16`, `lora_bonly_r32` | Rank generalization inside fixed-A family |

## Success criteria

Minimum useful result:

```text
B-only LoRA-r8 signed RFD > 10%
primary ImageNet top-1 >= 63%
```

Strong result:

```text
B-only LoRA-r8 signed RFD is high
full LoRA-r8 signed RFD moves clearly above the ~1% ceiling
primary ImageNet top-1 >= 63%
```

Negative but useful result:

```text
B-only RFD improves, full LoRA-r8 remains near zero
```

This would show that fixed-A accessibility is only one component of the LoRA
failure, and trainable A opens an additional bypass. That would motivate v9:
full LoRA-aware immunization or a tangent-kernel objective over both A and B.

Hard failure:

```text
B-only RFD remains near zero
```

Then the direct post-attack loss is not enough, and we should pivot to a
kernel/alignment formulation rather than more trajectory traps.

## Implementation checklist

1. Done: add B-only inner-loop loss to `src/trap_loss.py`.
2. Done: add `lora_bonly_r<int>` operator support to `trap_loss_multiop`.
3. Done: add config `configs/immunize_v8a_bonly.yaml`.
4. Done: run smoke test on role-lab.
5. Done: run full v8a immunization.
6. Done: run probe matrix.
7. Done: write `results/STAGE8_v8a_REPORT.md`.

Implementation details and smoke-test output:

```text
results/STAGE8_v8a_IMPLEMENTATION_LOG.md
```

Final v8a result:

```text
results/STAGE8_v8a_REPORT.md
```

## Expected contribution if it works

Stage 8 would support the paper claim:

> Linear-probe immunization fails under LoRA because the harmful task remains
> accessible through adapter-induced reachable directions. Directly immunizing
> against the adapter basis reduces this accessibility and moves robustness
> beyond the linear-probe-only threat model.

This is a stronger and cleaner story than another Taylor-trap variant.
