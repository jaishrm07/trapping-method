# Thread 03 — DRO / min-max over operator mixtures

**Status:** tested 2026-05-03 — running-mean-trap-value DRO null; see Results section below
**Type:** theory + lit
**Owner:** —

## Question

Plan C samples one operator uniformly per defender step. Should it
instead pick the operator that *currently makes most progress* (DRO /
min-max), so the defender focuses gradient on the weakest spot?

Empirically: in Stage 5 v2, LP RFD = 47, LoRA RFD = 1. If we'd been
preferentially training against LoRA, would the defender close the gap?

## Why this matters

Operator-randomization treats LP and LoRA as symmetric — defender
spends 50% of its capacity on each. But LoRA is the *harder* operator
to bound (richer parameter space). Equal sampling under-allocates
defender effort to where it's needed.

Counter: LP is the operator we *can* defend, so removing LP samples
might lose that defense. We need to characterize the trade-off, not
just pick one extreme.

## Reading list

1. **Sinha, Namkoong, Duchi 2018 — DRO with f-divergence.** Foundation
   for distributionally robust optimization in ML. Inner sup over
   distributions ≈ inner sup over operators in our case.
2. **Sagawa et al. 2020 — Group DRO.** Practical DRO for finite groups
   (= our finite operator set). Implementation tricks, theoretical
   guarantees.
3. **Volpi et al. 2018 — adversarial data augmentation as DRO.** Closest
   analogue to "adversarial operator mixture."
4. **Madry et al. 2018, Tramer & Boneh 2019.** Multi-perturbation
   adversarial training — same flavor problem (defend against a *set*
   of attacks). Their findings on weighted-vs-uniform are directly
   transferable.

## Concrete formulations to consider

### Option A — Group-DRO weighted sampling

Instead of `op = uniform(operators)`, maintain weights `w[op]` updated
per step:
```
w[op] ← w[op] * exp(η_w · trap_loss[op])
```
so operators with higher trap loss (= worse current defense) get
sampled more often. Standard exponentiated-gradient update.

### Option B — Worst-case (true min-max)

Each defender step: compute trap loss for *all* operators, defender
gradient is on the max. Most aggressive; expensive (k× more inner
loops per defender step).

### Option C — Randomized worst-case

Sample top-k operators by recent trap loss, mix only those. Cheaper
than B, more focused than A.

### Option D — Curriculum

Start with LP-only (we know it works), gradually add LoRA-r8, then
LoRA-r32, etc. Matches Antoniou's MAML-stability-by-warmup intuition.

## Open hypotheses to test (post-derivation)

- Does Option A's exponentiated weighting collapse to "always LoRA"
  because LoRA trap loss is structurally larger than LP trap loss
  (the softplus ranges differ)? If yes, we need a normalization.
- Is the right thing to weight by *RFD* (downstream), not trap value
  (proxy)? This would require periodic adversarial-probe estimates —
  expensive.

## Acceptance criterion

Thread closes when we have:
- A specific weighted-sampling formula for `trap_loss_multiop` that's
  derived from a DRO objective, **AND**
- A test plan distinguishing it empirically from uniform sampling.

## Results — running-mean-trap-value DRO null (2026-05-03)

Tested as Stage 5 v5a:
- Sampling: `prob[op] ∝ clamp(running_mean_trap[op], 0.1, 10)`
- Update per defender step: `running_mean[op] = 0.95·old + 0.05·trap_value`
- Min weight floor: 0.1 (so neither operator goes to zero sampling)
- Otherwise identical to v4a (FOMAML, no predictor, λ_trap=0.3, grad_clip=1)

### Outcome

| Adversary | Baseline | v4a (uniform) RFD | v5a (DRO) RFD | Δ |
|---|---|---|---|---|
| linear_probe | 41.23% | 47.95 | **18.02** | **−29.93** ↓↓↓ |
| lora_r8      | 73.85% | 1.14  | 1.04 | −0.10 |
| lora_r32     | 78.15% | −1.18 | −0.99 | +0.19 |
| full_ft_upper| 80.76% | 0.62  | 1.03 | +0.41 |
| full_ft_all  | 80.64% | −0.95 | −0.47 | +0.48 |

**LP defense collapsed; LoRA defense unchanged.**

### Why it failed (diagnosis)

Running-mean DRO assumes a *stationary* trap-value distribution per
operator. In our setting both operators saturate to ~0 trap value over
training:

- **LP saturates from defender success.** As the defender shapes
  features to bound LP, the trap value (= `softplus(ΔL_act)`) drops
  toward 0 because the LP adversary makes essentially zero progress.
- **LoRA saturates from inner-loop overshoot.** With η=0.1 and B=0
  init, the LoRA inner loop occasionally produces ΔL_act < 0 (the
  inner adversary makes the harmful loss WORSE), and `softplus(neg)
  ≈ 0`.

DRO sees both saturations as "this operator is easy" and downweights
sampling. Defender then *forgets* the LP defense it had built up.

### What a non-broken DRO would look like

The DRO signal needs to be a property of the *adversary*, not the
*trap loss*. Concretely:

1. **Adversarial harmful accuracy** — periodically run a short
   simulated adversary (e.g., 5 epochs of LoRA-r8 fine-tune on a
   held-out harmful subset), record its accuracy, weight by that.
   Operators that achieve high accuracy = unbounded → sample more.
   Cost: 5 extra epochs per re-weighting (every N defender steps).
2. **Worst-case (true min-max)** — at each defender step, compute
   trap loss for ALL operators, defender gradient on the max.
   Cost: k× more inner loops per defender step.
3. **Curriculum** — start with LP-only (we know it works), gradually
   add LoRA. Avoids the saturation cycle.

All are more expensive than v5a's running-mean update. None tested.

### Acceptance status

Thread NOT closed. We have a *negative result for one specific DRO
formulation*. The general DRO question (does adversarial-loss-weighted
sampling beat uniform?) remains open — needs a different signal than
running-mean trap value.
