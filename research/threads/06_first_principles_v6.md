# Thread 06 — First-principles reformulation after the LoRA-RFD ceiling

**Status:** open (designed 2026-05-03)
**Type:** theory + design
**Owner:** —

## Why this thread exists

Threads 01–05 each tried to push the trap mechanism harder along a
specific axis (predictor, MAML/FOMAML, DRO weighting, PEFT family,
diagnostics). The empirical record (`empirical_state.md`, decision
log 2026-05-03) shows four orthogonal interventions all hitting the
same LoRA-r8 RFD ceiling near 1.1. That's not a tuning failure — it's
a structural property of the formulation.

This thread starts over from "what is the defender actually trying
to do?" and asks whether a different *kind* of formulation can break
through the ceiling.

## The structural diagnosis

Restate what the trap loss does: at each defender step, simulate one
trajectory of an adversary's k-step inner loop, penalize realized
harmful loss reduction beyond local-geometry prediction. The defender
shapes features so that adversary trajectories look deceptively flat.

This works for LP. Why?

| Adversary | Free parameters | What they CAN change | What they CAN'T change |
|---|---|---|---|
| **LP** | `ω_H` only | Classifier readout from features | The features themselves |
| **LoRA** | `(A, B, ω_H)` | Features themselves (via `W ← W + B@A`) AND readout | (only constraints from rank and norm) |

For LP the defender controls every input to LP's loss landscape (the
features). The trap simulates LP's k-step path through a landscape
the defender shaped — and wins.

For LoRA the adversary's loss is computed on `f(x; W + B@A)`, NOT on
the features the defender shipped. LoRA can move to feature
configurations the defender never trained against. The trap is
simulating one point in a continuous reachable region.

**1-vs-1 (LP) versus 1-vs-∞ (LoRA).** No amount of inner-loop tuning
fixes this; it's an *approximation* of an infinite set with a single
finite simulation.

## The reformulation: robust optimization in the LoRA perturbation space

Current formulation (Zheng + Sarker hybrid):
```
min_θ  L_primary(θ) + λ_well · r_well(H_P(θ)) + λ_ill · r_ill(H_H(θ)) + λ_trap · trap(θ)
```

Proposed v6 formulation (drop trap, add robustness to weight perturbation):
```
min_θ  L_primary(θ)
       + λ_well_robust · max_{Δ ∈ B_r(ε)} r_well(H_P(θ + Δ))
       + λ_ill_robust  · max_{Δ ∈ B_r(ε)} -r_ill(H_H(θ + Δ))
```

where `B_r(ε)` is the ball of rank-r perturbations of θ_upper with
norm ≤ ε. The adversary's "best move" (worst-case Δ) is found by
inner gradient ascent (PGD-style) in this constrained space.

### What this changes mechanically

- **Trap loss removed.** No more `softplus(ΔL_act)` or its variants.
- **Inner loop is now PGD on Δ**, not SGD on `(A, B, ω_H)`. Adversary
  becomes "the worst rank-r weight perturbation" rather than "a
  k-step LoRA fine-tune from random init".
- **Defender's training objective directly bounds post-perturbation
  metrics.** No proxy via softplus saturation behavior.

### Why this might break the 1.1 ceiling

The current trap simulates ONE adversary path; any v6 step regularizes
against the WHOLE rank-r ball at perturbation budget ε. The defender
sees worst-case Δ at each step rather than a single random initialization.
Theoretically this should bound the adversary class, not just one point
in it.

This is exactly Madry et al. 2018 PGD-AT but applied in *weight* space,
restricted to rank-r perturbations — and applied to the condition-number
regularizers from Zheng's work, not to a primary-task loss. As far as
field map (research/threads/04) shows: not done.

## Open questions before implementing

### Q1 — Does the existing v4a defense already generalize across rank?

If v4a's trap, despite being trained against LoRA-r8 only, happens
to work at LoRA-r4, r16, r64 too, then "operator-narrowness" was
overstated and v6's complexity might not be needed.

If v4a's defense is rank-overfit (drops to 0 for r ≠ 8), v6 has
clear contribution.

**Cheapest experiment**: probe v4a's extractor at additional ranks
{1, 2, 4, 16, 64} (we have r8 and r32 already). 4 probes × 15 min =
1 hour on role-lab. **This must be done first.**

### Q2 — Is rank-r projection of Δ tractable?

After PGD ascent, we need to project Δ back onto the rank-r ball.
For a single matrix this is SVD truncation. For our setting Δ is a
*tuple* of per-conv-weight rank-r matrices — each layer projected
separately. SVD per conv weight is fast (≤ 512×512 typical) but we
know SVD fails sometimes in our pipeline.

**Mitigation**: factor Δ_l = B_l A_l^T directly (rank-r by
construction), parameterize PGD on (A_l, B_l) instead of on Δ_l.
No projection needed. Same trick LoRA itself uses.

### Q3 — Will the inner PGD be stable?

We just got bitten by bilevel instability in v4. Different setup
(PGD on Δ vs SGD on adversary params), but same risk.

**Mitigation**: small ε (start at ε=0.01·‖θ_upper‖), small k_pgd (3-5
ascent steps), grad clipping on Δ, and ε-tightening if NaN appears.

### Q4 — Does the outer min-max collapse?

Adversary's PGD might find degenerate Δ that flatten r_ill to 0
trivially (zeroing out all features so the harmful Hessian is the
zero matrix and "ill-conditioning" is undefined). Defender then has
no useful signal.

**Mitigation**: keep the SVD-ridge and try/except in r_well/r_ill
(already done). Also: bound ε so Δ can't fully cancel θ_upper's
features.

## Cost vs benefit

| Run | Cost (role-lab, A6000) | What it tells us |
|---|---|---|
| Rank-extrapolation probe of v4a (Q1) | ~1 hr (4 probes parallel) | Whether v4a is rank-narrow → motivates v6 |
| v6 immunization | ~100 min | First test of robust-optimization formulation |
| v6 probe array (5 ranks) | ~30 min | RFD vs rank curve |

Total to a v6 result: ~3 hours. Manageable.

## Acceptance criterion

Thread closes when we have either:
- A v6 implementation that produces LoRA RFD significantly above the
  v4a 1.1 ceiling on at least one rank, **OR**
- A clear argument (theoretical or empirical) for why the
  robust-optimization reformulation also fails, motivating yet
  another direction.

## Pre-v6 work (committed)

Run the rank-extrapolation probe of v4a at ranks {1, 2, 4, 16, 64}
on role-lab (parallel array) before any v6 implementation. Update
this thread with results before designing v6 code.

## Acknowledged limitations of this design

- This formulation only addresses Class-2 approaches (adversarial
  regularization on perturbation space). Class-1 (restrict threat
  model) and Class-3 (information-theoretic feature engineering)
  are not pursued here — would need separate threads.
- The robustness budget ε is a hyperparameter we don't have a
  principled value for yet. Hayou 2024's intrinsic-LR analysis gives
  a starting point but transfer to ResNet/Cars setting is unclear.
- We're committing to the trap-replacement narrative ("v6 drops the
  trap"). If v6 works, the contribution story is "trap-loss family is
  insufficient for LoRA defense; robust optimization is the right
  framework." That's a stronger claim than we have evidence for yet.
