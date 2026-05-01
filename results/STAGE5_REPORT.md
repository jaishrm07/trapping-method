# Stage 5 Report — Plan C (Multi-Operator Trap, v1)

Date: 2026-05-01

## Headline (negative result, with caveats)

We added a LoRA-aware trap (Eq. 6 of the trapping paper, but in the
expanded `(LoRA_A, LoRA_B, ω_H)` parameter space) and combined it with
the existing linear-probing trap via per-step random-operator
selection. **The LoRA defense did not improve. Plan C v1 does not fix
the operator-transfer collapse.**

| Adversary | Baseline | Stage 4.5 (LP-trap) | **Stage 5 (LP+LoRA-r8 multi-op)** |
|---|---|---|---|
| linear_probe | 41.23% | 22.98% (RFD=50.10) | **18.62% (RFD=60.30)** ⬆ |
| lora_r8 | 73.85% | 73.73% (RFD=0.65) | **74.03% (RFD=0.67)** ≈ |
| lora_r32 | 78.15% | 78.39% (RFD=0.60) | 78.54% (RFD=0.60) ≈ |
| full_ft_upper | 80.76% | 80.75% (RFD=0.74) | 80.08% (RFD=0.74) ≈ |
| full_ft_all | 80.64% | 81.40% (RFD=2.02) | 81.27% (RFD=2.02) ≈ |

Stage 5 final immunization metrics: RIR=2.38, primary acc=63.88% — a
profile very close to Stage 4.5's (RIR=1.11, primary 64.13%), so the
overall optimization landed in a similar Pareto neighborhood.

## Two findings, the second more interesting than the first

### Finding 1 (negative): LoRA-r8 RFD didn't budge

The Stage 5 immunized backbone gives the LoRA-r8 adversary essentially
the same final accuracy (74.03%) as the un-immunized backbone (73.85%).
Within run-to-run noise. The trap simulation that targeted LoRA-r8
specifically failed to bound it.

### Finding 2 (positive surprise): LP defense became *stronger*

LP RFD: 50.10 → 60.30. The multi-op formulation, even with a useless
LoRA branch, made LP defense ~10 points stronger than the single-op LP
trap. Most likely explanation: the simulated LoRA inner loop acts as
additional input-noise / regularization for the defender's gradient,
which improves the overall harmful loss landscape against LP without
specifically targeting LoRA.

This is surprising and the kind of mechanism that's worth investigating
on its own. But it doesn't address the operator-transfer question.

## Why the LoRA trap branch didn't work — diagnosis

Most likely cause: **the LoRA inner loop is too weak to produce
meaningful trap signal**. With:

- LoRA `B` initialized to zero (standard LoRA init), the initial weight
  delta is exactly zero.
- After k=3 inner SGD steps with η=0.01, the delta is still very small.
- The realized harmful loss reduction `ΔL_act` is therefore tiny.
- `softplus(small_positive)` ≈ `log(2) + small/2`. Defender gradient
  through this is small in magnitude.
- The defender effectively receives no signal pointing toward "make
  features hard for LoRA to fit" — only "make features hard for LP to
  fit," which it does well.

Verifiable via the per-step `loss_trap` history (logged but not
surfaced in this report). The hypothesis predicts trap_loss values
~log(2) ≈ 0.69 most steps, with little variance — much smaller than
the LP trap values ought to be.

## Three concrete fixes for Plan C v2

Listed in increasing intervention order:

| Fix | Cost | Expected effect |
|---|---|---|
| **A. Larger inner lr** (η=0.01 → 0.1) | Free | LoRA factors move 10× more in 3 steps; ΔL_act becomes meaningful |
| **B. Longer inner loop** (k=3 → k=10) | ~3× memory & compute per defender step | LoRA actually fits some structure |
| **C. Warm-start LoRA** (run 1 epoch outer-loop on LoRA factors before each trap eval) | Modest extra compute | Inner adversary starts from a fitted state, ΔL_act reflects late-stage progress not early-stage stalling |

Plan C v2 starts with **Fix A alone** (cheapest, single hyperparameter
change) and rerun. If LoRA RFD doesn't move, escalate to Fix B; if
still nothing, Fix C is the heavier intervention.

## What this still teaches us

Even as a negative result, Stage 5 is informative:

1. **First empirical evaluation of multi-operator trap immunization.**
   Until this point, the operator-transfer hypothesis was untested.
   Now we know: naive operator randomization with weak inner loops
   does not generalize.
2. **The LP-trap improvement (50.10 → 60.30) is a positive byproduct.**
   Even when one branch of the multi-op trap is useless, the
   additional regularization sharpens the in-distribution defense.
   Worth understanding mechanistically.
3. **The fix space is well-defined and tractable.** The three
   knobs above are all 1-day experiments; we know what to try.

## Files

- `trap_multiop_lp_lora8_resnet18_cars/{results.json, extractor.pt}`
  — Stage 5 immunization run.
- `adv_multiop_*_cars/results.json` — five adversarial probes.

## Single-paragraph summary

> Plan C v1 (linear probe + LoRA-rank-8, per-step random sampling)
> did not bound LoRA adversaries (RFD = 0.67 in-distribution vs
> 0.65 baseline). The simulated LoRA inner loop was too short and
> too gentle to produce meaningful trap signal. As a side effect,
> in-distribution LP defense improved by ~10 RFD points. The next
> experiment increases inner-loop learning rate to 0.1 to test
> whether stronger inner-loop signal is enough to make multi-op
> trap actually work.
