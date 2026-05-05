# Stage 7 Report - LoRA tangent accessibility

Date: 2026-05-04

## Headline

The first-principles hypothesis is supported by the stronger diagnostic:

> Linear-probe immunization blocks harmful readout from frozen features, but it
> does not block harmful adaptation through the LoRA adapter basis.

The cleanest evidence is v4a:

- v4a linear probe final accuracy: 21.46%
- v4a B-only LoRA-r8 final accuracy: 51.10%
- v4a full LoRA-r8 final accuracy: 73.01%

So even when the frozen representation is made poor for linear probing, a
fixed random LoRA-A basis with trainable B plus a head reopens the Cars task.

## Principle

Linear probing only learns a classifier on frozen features:

```text
x -> W f_theta(x)
```

The defender can damage harmful separability in `f_theta(x)`, because the
attacker has no way to change the representation.

LoRA exposes a larger reachable class:

```text
x -> W f_{theta + Delta_theta(phi)}(x)
```

With standard LoRA init, A is random and B is zero. At the first step, the
nonzero adapter gradient is through B. Freezing A and training only B therefore
tests the initial LoRA adapter basis:

```text
Delta h_l(x) = B_l A_l h_l(x)
```

If B-only LoRA succeeds while linear probing fails, the harmful task is still
accessible through representation-moving adapter directions.

## Decision log

1. We stopped tuning the Taylor trap because v6 showed exact HVP accounting did
   not move LoRA RFD.
2. We added a random finite-difference tangent sketch as a cheap static
   separability assay.
3. We added `lora_bonly_r<int>` to the adversary runner as the more faithful
   optimized tangent-basis probe.
4. We ran baseline, v4a, and v6 under both diagnostics.

## Diagnostic A - Random LoRA tangent sketch

Method:

```text
psi_m(x) = (f_{theta, B=eps v_m}(x) - f_{theta, B=0}(x)) / eps
```

Then train linear classifiers on:

```text
base         = f_theta(x)
tangent      = [psi_1(x), ..., psi_M(x)]
base_tangent = [f_theta(x), psi_1(x), ..., psi_M(x)]
```

Settings:

- LoRA rank: 8
- Epsilon: 1e-2
- M: 8 random B directions for baseline, v4a, v6
- Extra v4a control: M = 32
- Probe optimizer: AdamW on fixed extracted features
- Probe epochs: 200

Results:

| Run | M | Base final | Tangent final | Base + tangent final |
|---|---:|---:|---:|---:|
| baseline | 8 | 34.40% | 21.12% | 23.60% |
| v4a | 8 | 20.35% | 15.47% | 15.91% |
| v6 | 8 | 33.50% | 18.55% | 21.14% |
| v4a | 32 | 20.23% | 10.72% | 10.53% |

Interpretation:

The random sketch is a negative/inconclusive diagnostic. It does not recover the
LoRA gap, and increasing from 8 to 32 random directions makes v4a worse. That
means a small random finite-difference sketch is not a reliable proxy for the
optimized adapter direction. It samples arbitrary directions; the attacker
chooses useful directions.

This result is still useful because it rules out an overly simple claim:
"any random LoRA tangent sketch exposes the harmful task." The right claim is
about optimized accessibility, not arbitrary random accessibility.

## Diagnostic B - B-only LoRA adversary

Method:

We added adversary type:

```text
lora_bonly_r8
```

It freezes the base model and freezes LoRA A, then trains only:

```text
LoRA B factors + harmful classifier head
```

This is weaker than full LoRA because A cannot adapt. It is stronger and more
faithful than the random sketch because B is optimized from labels.

Trainable parameters:

```text
131,268 / 11,504,388 = 1.1%
```

Results, 50 epochs on Cars:

| Operator | Baseline final | v4a final | v6 final | v4a signed RFD | v6 signed RFD |
|---|---:|---:|---:|---:|---:|
| linear_probe | 41.23% | 21.46% | 43.27% | 55.98 | -1.95 |
| lora_bonly_r8 | 51.19% | 51.10% | 51.37% | 0.71 | -0.04 |
| lora_r8 | 73.85% | 73.01% | 73.50% | 0.28 | -0.10 |
| lora_r32 | 78.15% | 79.07% | 78.31% | 0.06 | -0.04 |

Key observation:

v4a strongly suppresses the linear probe, but has essentially zero effect on
B-only LoRA-r8. The final B-only accuracies are nearly identical:

```text
baseline: 51.19%
v4a:      51.10%
v6:       51.37%
```

So the harmful task remains accessible through the adapter basis even after
the frozen feature readout has been damaged.

## What this proves and does not prove

Supported:

> The failure of linear-probe immunization under LoRA is already visible in a
> fixed-A, trainable-B adapter subspace. Full LoRA does not need to reshape both
> A and B to bypass the defense.

Not claimed:

> A 50-epoch B-only run is an infinitesimal local tangent calculation.

After many epochs, B can become large and the network path is no longer purely
local. But the attack remains far weaker than full LoRA because A is fixed. It
is therefore a meaningful intermediate operator between frozen linear probing
and full LoRA.

## Fundamental conclusion

The missing principle is operator reachability:

```text
Defend the harmful task against the attacker's reachable function class,
not only against the frozen feature readout.
```

For linear probing, the reachable class is:

```text
F_LP(theta) = { W f_theta(x) }
```

For B-only LoRA, the reachable class is already:

```text
F_B(theta, A) = { W f_{theta + B A}(x) : B trainable, A fixed random }
```

v4a immunization changes `F_LP(theta)` but does not meaningfully change
`F_B(theta, A)`.

## Next objective

The next defense should be B-only/adapter-basis aware. Two concrete paths:

1. B-only adversarial immunization:

```text
min_theta L_primary(theta)
        + lambda * trap_or_final_loss(theta, B-only LoRA attacker)
```

This directly targets the failure mode we just isolated.

2. LoRA tangent-kernel alignment:

```text
K_B(i,j) = <d logits(x_i) / dB, d logits(x_j) / dB>
```

Then penalize harmful-label alignment with this kernel on minibatches. This is
more principled but requires careful HVP/JVP or sketching machinery.

Recommendation:

Start with B-only adversarial immunization. It is experimentally direct, uses
the same adversary runner logic, and tests whether defending the first-order
adapter basis can move the LoRA RFD ceiling.

## Files

- `experiments/run_tangent_accessibility.py`
- `experiments/run_adversary.py` with `lora_bonly_r<int>`
- `results/tangent_access_baseline_r8_m8/results.json`
- `results/tangent_access_v4a_r8_m8/results.json`
- `results/tangent_access_v6_taylor_hvp_r8_m8/results.json`
- `results/tangent_access_v4a_r8_m32/results.json`
- `results/adv_baseline_lora_bonly_r8_cars/results.json`
- `results/adv_v4a_lora_bonly_r8_cars/results.json`
- `results/adv_v6_taylor_hvp_lora_bonly_r8_cars/results.json`
