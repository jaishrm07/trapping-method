# Rank-Extrapolation Report — v4a's LoRA defense across the rank distribution

Date: 2026-05-03

## Headline (the most important finding so far)

We probed v4a's immunized backbone against LoRA adversaries across
rank ∈ {1, 2, 4, 8, 16, 32, 64} and compared to the un-immunized
baseline at the same ranks. **The trap mechanism produces ~1pp RFD
at the rank it was trained against (r=8) and essentially 0pp at every
other rank.** The defense does not generalize across the LoRA rank
distribution.

This is **Agent A's null hypothesis #2 (operator-extrapolation null)
empirically confirmed**, and a substantial correction to the earlier
"hard ceiling at LoRA-r8 RFD ≈ 1.1" story: the true ceiling is ~0
RFD across the rank distribution; r=8's ~1pp is a narrow rank-overfit
artifact.

## Setup

- **Immunized backbone**: v4a (Stage 5 v4a, FOMAML inner loop, no
  predictor, λ_trap=0.3, grad_clip=1, trained against `[LP, LoRA-r8]`).
  Same `extractor.pt` used for all rank probes.
- **Baseline backbone**: ImageNet-pretrained ResNet18 (no immunization).
- **Adversary protocol**: 50-epoch fine-tune on Cars; same hyperparameters
  as standard adversary probes (`configs/default.yaml`).
- **Ranks tested**: 1, 2, 4, 8, 16, 32, 64. r=8 and r=32 came from the
  prior v4a probe array; r=1, 2, 4, 16, 64 are new.
- **Compute**: role-lab (8× RTX A6000), GPUs 1–5 in parallel for each
  array, ~15 min per array.

## Results

### Final adversary accuracy after 50 epochs

| Rank | Baseline | v4a | Δ (pp) | RFD |
|---|---|---|---|---|
| 1  | 63.50% | 63.14% | −0.36 | **0.57** |
| 2  | 68.23% | 67.89% | −0.34 | **0.50** |
| 4  | 71.25% | 71.21% | −0.04 | **0.06** |
| 8  | 73.85% | 73.01% | −0.84 | **1.14** ← trained against |
| 16 | 76.93% | 76.84% | −0.09 | **0.12** |
| 32 | 78.15% | 79.07% | +0.92 | **−1.18** |
| 64 | 78.45% | 78.30% | −0.15 | **0.19** |

RFD = (baseline − immunized) / baseline × 100. Higher = stronger defense.

### The curves visualized

```
Adversary accuracy (Cars, 50ep fine-tune)
─────────────────────────────────────────
80% ┤             ╭─── baseline
    │           ╭─╯  ╭─ v4a
    │         ╭─╯  ╭─╯
70% ┤       ╭─╯  ╭─╯
    │     ╭─╯  ╭─╯
    │   ╭─╯  ╭─╯
60% ┼─╭─╯─╭─╯─────────────────────────
    └─┬───┬───┬───┬───┬───┬───┬───────
      1   2   4   8   16  32  64
                      rank →
```

The two curves are visually indistinguishable above the noise floor.
Both rise smoothly from ~63% (r=1) to ~78% (r=64), reflecting
inherent capacity-vs-rank scaling of the LoRA adversary.

## What this changes about previous claims

The "hard ceiling at LoRA-r8 RFD ≈ 1.1" finding from `STAGE5_v5_REPORT.md`
needs amendment:

**Previous framing (overstated):**
> Across four orthogonal interventions (LR, autograd, weighting, operator
> family), LoRA-r8 RFD has not exceeded ~1.2. The bilevel-trap formulation
> has a structural ~1pp LoRA-r8 RFD ceiling.

**Corrected framing:**
> Across four orthogonal interventions, LoRA-r8 RFD has not exceeded
> ~1.2 — but the rank-extrapolation probe shows this 1.1pp is itself a
> narrow rank-overfit artifact, with RFD ≈ 0 at every other rank tested.
> The true claim is: the bilevel-trap formulation produces ~0 generalized
> LoRA defense across the rank distribution, with marginal rank-specific
> overfitting (~1pp) at the trained-against rank.

This is a *stronger* statement than the original ceiling claim. It
rules out a much wider class of "tune the trap better" interventions,
because the ceiling is structural at the formulation level, not just
quantitative.

## Why this matters for v6 design

The first-principles thread (`research/threads/06_*`) proposed v6 as
"robust optimization on the rank-r perturbation ball" precisely because
the trap simulates one path while LoRA lives in a continuous reachable
region. The rank-extrapolation result is direct empirical support:

- If v4a's defense were rank-narrow (peaked at r=8, dropped elsewhere),
  v6's multi-rank robust optimization would be the obvious fix → strong
  contribution angle.
- If v4a's defense were rank-uniform (constant ~1pp across ranks),
  v6 would be solving a smaller ceiling-breaking problem.
- **What we actually see**: v4a's defense is rank-uniform AT ZERO. v6
  is now trying to get *any* defense, not break a small ceiling. Higher
  bar for v6 to be worth doing.

The implication: any positive v6 result — even RFD=2 across ranks —
would be a meaningful improvement over the trap mechanism, since the
trap doesn't cross noise floor at any rank.

## Three null hypotheses — updated state

| Null hypothesis | Status |
|---|---|
| Capacity tradeoff — bounding LoRA-reachable subspace destroys benign LoRA-FT | NOT tested |
| **Operator-extrapolation** — defense at r₀ fails at r₁ ≠ r₀ | **CONFIRMED** by this report. v4a defense (~1pp) only at r=8; ~0 elsewhere |
| Side-door — pruning + untrained low-rank perturbations bypass | NOT tested |

The capacity tradeoff null becomes the next priority. If we ever build
a positive defense (v6 or otherwise), we need to verify that benign LoRA
fine-tuning still works on the immunized backbone. Otherwise we're not
defending — we're destroying utility.

## Files

- `results/adv_v4a_rankprobe_lora_r{1,2,4,16,64}_cars/results.json`
  — five v4a probes at non-{8,32} ranks.
- `results/adv_baseline_lora_r{1,2,4,16,64}_cars/results.json`
  — five baseline probes at non-{8,32} ranks.
- `results/adv_v4a_lora_r{8,32}_cars/results.json`
  — existing v4a probes at r=8 and r=32 (from `STAGE5_v4a_REPORT.md`).
- `results/adv_baseline_lora_r{8,32}_cars/results.json`
  — existing baseline probes at r=8 and r=32 (from earlier
  `OPERATOR_TRANSFER_REPORT.md`).

## Single-paragraph summary

> Probing v4a across LoRA ranks {1, 2, 4, 8, 16, 32, 64} reveals a
> rank-RFD curve that is essentially flat at zero. The defense produces
> ~1pp RFD at the trained-against rank (r=8) and ~0pp elsewhere
> (including −1.18pp at r=32, where the immunized model is slightly
> easier than baseline). The earlier "hard ceiling at 1.1 RFD" claim
> was a rank-overfit artifact; the true claim is "~0 generalized LoRA
> defense, with ~1pp narrow overfitting at the training rank." Agent A's
> operator-extrapolation null hypothesis is empirically confirmed. v6
> (robust optimization in LoRA perturbation space) is now positioned as
> "produce any generalized defense" rather than "break a small ceiling."
