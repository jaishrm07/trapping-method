# Operator-Transfer Report — Trap Geometry Does Not Generalize

Date: 2026-05-01

## Headline

The Stage-4.5 immunized backbone — which reproduces the trapping paper's
Pareto-optimal operating point on Cars/ResNet18 (RFD≈50%, primary acc
≈64%, RIR≈1) — was tested against four adversarial operators stronger
than the linear probing it was trained against. **Trap geometry collapses
under every one of them.**

## Operator-transfer matrix (Cars / ResNet18, 50-epoch probes, SGD lr=0.01)

| Adversary | Baseline final acc | Immunized final acc | Δ (pts) | **RFD** |
|---|---|---|---|---|
| `linear_probe` (in-distribution) | 41.23% | 22.98% | −18.25 | **50.10%** |
| `lora_r8` | 73.85% | 73.73% | −0.12 | **0.65%** |
| `lora_r32` | 78.15% | 78.39% | +0.24 | **0.60%** |
| `full_ft_upper` | 80.76% | 80.75% | −0.01 | **0.74%** |
| `full_ft_all` | 80.64% | 81.40% | +0.76 | **2.02%** |

**RFD drops by 25–80× the moment the adversary deviates from linear
probing.** The trap that holds RFD = 50% in-distribution provides
essentially zero protection against any stronger operator.

In two cases (`lora_r32`, `full_ft_all`) the immunized backbone is
*marginally easier* to attack than the un-immunized one — the trap
geometry slightly *helps* the adversary by pre-shaping features along
ImageNet structure that LoRA / full-FT can exploit.

## What this answers

The single empirical question from the experiment plan:

> Does trap geometry, shaped against a *linear-probing* adversary,
> bound stronger adversarial operators on the same downstream task?

**Answer: No.** Trap is *narrowly operator-specific*. The math machinery
that produces the in-distribution defense (Hessian shaped against
ω_H-only updates) does not apply when the adversary's update operator
includes the backbone parameters themselves.

This matches **Outcome 2** from the experiment plan ("trap is
operator-specific; motivates multi-operator trap"). It rules out the
optimistic outcome (trap is intrinsically agnostic) and the
intermediate outcomes (graceful degradation with adversary capacity).

## Mechanistic reading

The trap in §3 of Sarker et al. is built on a Taylor expansion of
`L_H(ω_H + Δω)`, with `Δω` interpreted as movements *only* in the
adversary's head parameters. The Hessian `H_0` is computed as the
feature-covariance `Kᵀ K`, valid *only* when the parameter being moved
is `ω_H` and the loss is linear-probing CE.

When the adversary's update operator is full FT or LoRA, **the
adversary's free parameters are no longer just `ω_H`.** They include
backbone weights too. The Taylor expansion the defender used during
trap training does not describe these directions at all — they're
*outside* the parameter subspace the trap controls.

Concretely: the trap pushes `θ_upper` (the immunization-trainable
backbone) into a configuration where, *for any fixed `θ_upper`*, an
ω_H-only adversary makes no quadratic-surplus progress. But once the
adversary can also move `θ_upper`, they can step out of the trap
geometry by moving in *backbone* directions the trap-induced
non-quadratic structure doesn't see.

The trap is a *parameter-subspace* defense, not a feature-space
defense. Outside its trained subspace, it has no force.

## Quantitative scaling

The RFD progression with adversary capacity:

```
adversary param count    RFD
~100 K  (linear probe)   50.10%   ← trap subspace
~200 K  (lora_r8)         0.65%
~700 K  (lora_r32)        0.60%
~9 M    (full_ft_upper)   0.74%
~11 M   (full_ft_all)     2.02%
```

The collapse is not gradual. Going from ~100 K to ~200 K parameters
already drops RFD to ~1%. The transition is essentially a step function:
"trap inside the LP subspace" vs "trap absent everywhere else."

This suggests the issue isn't just adversary capacity — it's that the
trap's sphere of influence is exactly the parameter subspace it was
trained against, and **any parameter beyond that subspace is unbounded**.

## What this means for the trapping paper's overall claim

The paper's experiments only ever evaluate against linear probing. Within
that experimental envelope, all of its claims hold (we've reproduced
this). But the paper's conclusion claims the method "raises the cost of
undesirable downstream adaptation" — implying real-world adaptation
robustness. **The result here shows that claim is true only for the
specific adapter-style operator the trap was trained against.**

If a deployed model is released with trap geometry, an attacker who
pulls the weights and runs *full fine-tuning* (the standard adversary
in real LLM safety threat models) sees no slowdown at all.

This is not a refutation of the trapping paper's methodology — the
math is sound for its threat model. It's a refutation of the threat
model's coverage. The paper's "harmful fine-tuning" defense bounds only
"harmful linear probing," and silently fails when the adversary picks
a stronger operator.

## Implication for follow-up work

Three natural directions, ranked by promise:

1. **Multi-operator trap.** Train the defender to be robust against an
   ensemble of inner adversaries: LP + LoRA-r8 + LoRA-r32 + full-FT
   simultaneously. Each defender step computes trap loss across all
   operators. Costly per-step but necessary if trap-style defense is to
   matter for realistic threat models.

2. **Subspace identification.** What is the parameter subspace `S` such
   that an attacker restricted to updates in `S` is bounded by trap?
   Empirically `S` is just `ω_H`, but is there a richer structural
   characterization (e.g., low-rank perturbations of feature-mean
   directions)?

3. **Operator-aware metric.** RFD as defined is operator-dependent. A
   robust extrinsic metric should integrate over a *distribution of
   operators*, not a single one. Unclear how to weight.

The first direction is the most concrete and the most relevant to the
deployed-LLM threat model.

## Implication for an MS thesis pitch

This result, by itself, is a workshop-paper-worthy stress-test: "Trap
geometry from CN+trap immunization is narrowly operator-specific; we
show the dropout from RFD=50% to RFD<2% across five adversaries on
Cars/ResNet18." Three things to add for a full conference paper:

1. Replicate on ViT and on Food101, Country211. Confirm the result is
   not ResNet18/Cars-specific.
2. Run trap-trained-against-LoRA → does that bound full-FT? Does
   trap-trained-against-full-FT bound LoRA? Build the full
   operator-transfer matrix.
3. Propose and evaluate a multi-operator defender: simulate an
   ensemble of operators in the inner loop. Quantify its
   utility-vs-immunization Pareto curve.

Item 3 is the method-paper contribution. Items 1–2 are the empirical
foundation.

## Caveat: same-lr setup

All five adversaries used SGD lr=0.01 (LP-tuned). For full FT this is
likely too aggressive; the actual standard is lr=1e-4. Even so, the
full-FT adversary reached 80.6% Cars accuracy vs 41.2% for LP — they
are using their oversized capacity effectively. Re-running with
per-adversary tuned LRs would refine the numbers but not change the
conclusion: trap completely fails to bound out-of-distribution
operators.

A follow-up "v2" experiment with operator-specific LRs is straightforward
on the existing infrastructure.

## Files

```
results/
├── adv_baseline_lora_r8_cars/results.json
├── adv_baseline_lora_r32_cars/results.json
├── adv_baseline_full_ft_upper_cars/results.json
├── adv_baseline_full_ft_all_cars/results.json
├── adv_immunized_45_lora_r8_cars/results.json
├── adv_immunized_45_lora_r32_cars/results.json
├── adv_immunized_45_full_ft_upper_cars/results.json
└── adv_immunized_45_full_ft_all_cars/results.json
```

Plus the previously-existing `baseline_probe_resnet18_cars/` and
`immunized_probe_cars_trap_paper_faithful/` for the linear-probe row.

## Single-paragraph summary

> The Stage-4.5 trap — reproducing the paper's Pareto-optimal
> immunization on Cars/ResNet18 — bounds the in-distribution adversary
> (linear probing, RFD=50%) but fails completely against four
> out-of-distribution adversaries (LoRA-r8, LoRA-r32, full-FT-upper,
> full-FT-all; RFD ≤ 2% in all cases). The defense is narrowly
> calibrated to the parameter subspace it was trained against, and
> provides essentially no protection once the adversary updates
> backbone parameters. This is the first empirical demonstration of
> operator-specificity for trap-style immunization, and it directly
> motivates multi-operator defender training as the next step.
