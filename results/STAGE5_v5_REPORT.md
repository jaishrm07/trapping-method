# Stage 5 v5 Report — DRO weighting (v5a) + PEFT-family expansion (v5b)

Date: 2026-05-03

## Headline (two negative results, one important synthesis)

We tested the two cheapest forks from the v4a postmortem
(`STAGE5_v4a_REPORT.md`):

- **v5a (Fork A)** — DRO weighting on operator sampling. Replaces uniform
  random sampling of `[LP, LoRA-r8]` with running-mean-trap-value
  weighting (defender preferentially trained on whichever operator
  *currently* has high trap value).
- **v5b (Fork B)** — PEFT-family expansion. Replaces `[LP, LoRA-r8]`
  with `[LP, LoRA-r4, LoRA-r8, LoRA-r16, LoRA-r32]`, uniform sampling.

**Both fail in different ways**, and the combined evidence across
v1, v2, v4a, v5a defines a **hard ceiling at LoRA-r8 RFD ≈ 1.1**
that four orthogonal interventions cannot break.

## v5a — DRO weighting

### Configuration
- Same as v4a (FOMAML, no predictor, λ_trap=0.3, grad_clip=1)
- One change: `dro_weighting: true`, `dro_decay: 0.95`, min weight floor 0.1
- Sampling: `prob[op] ∝ clamp(running_mean_trap[op], 0.1, 10)`
- Update per defender step: `running_mean[op] = 0.95·old + 0.05·trap_value`

### Run summary

| Metric | v2 | v4a | **v5a** |
|---|---|---|---|
| Final RIR | 1.295 | 1.580 | **0.533** |
| Final primary acc | 64.98% | 65.04% | **65.43%** |
| Wallclock (role-lab A6000, GPU 1) | — | 53 min | 1h 43m¹ |
| NaN events | 0 | 0 | 0 |
| Per-step trap (final) | n/a | 1.19 | **0.0000 (saturated)** |

¹ v5a was slower because GPU 1 was contested with v5b (which ran in parallel
on GPU 2 before being killed at step 1051).

### Probes

| Adversary | Baseline | v5a | v5a RFD | Δ vs v4a |
|---|---|---|---|---|
| linear_probe | 41.23% | **33.80%** | **18.02** | **−29.93** ↓↓↓ |
| lora_r8 | 73.85% | 73.08% | 1.04 | −0.10 |
| lora_r32 | 78.15% | 78.92% | −0.99 | +0.19 |
| full_ft_upper | 80.76% | 79.93% | 1.03 | +0.41 |
| full_ft_all | 80.64% | 81.02% | −0.47 | +0.48 |

### Diagnosis

The DRO mechanism is internally self-defeating in this regime:

1. As LP defense forms (trap value drops to ~0), DRO's running mean
   for LP drops, and `prob[LP]` falls toward the 0.1 floor.
2. Once LP sampling is rare, the defender stops reinforcing LP defense
   — and LP-trap *forgets* what it learned. LP RFD erodes.
3. Meanwhile the LoRA branch ALSO saturates — but to *zero* trap value
   (the LoRA inner loop overshoots, ΔL_act < 0, softplus(neg) ≈ 0).
   So `prob[LoRA]` ALSO drops, and DRO can't focus the defender on
   the operator we wanted it to focus on.
4. Final state: trap=0.0000 per step, both operators sampled near the
   weight floor, defender getting weak signal from both branches. RIR
   plateaus low (0.533 vs v4a's 1.580); LP defense erodes (RFD 47.95 →
   18.02); LoRA defense unchanged (1.14 → 1.04).

The DRO formulation we used assumes a **stationary trap-value
distribution per operator**, but the trap value is non-stationary —
it drops as the defender succeeds, and saturates to zero in two
different regimes (LP success vs LoRA overshoot). Running-mean DRO
treats both saturations as "this operator is easy", which is wrong
for the LoRA case.

A better DRO formulation would weight by *adversarial loss*, not
*trap value* — e.g., final harmful-task accuracy of the inner-loop
adversary, not the softplus output. Outside the scope of this report;
recorded as future work in `research/threads/03_*`.

## v5b — PEFT-family expansion

### Configuration
- Same as v4a, but `trap_operators: [LP, LoRA-r4, LoRA-r8, LoRA-r16, LoRA-r32]`
- `lora_rank_for: {lora_r4: 4, lora_r8: 8, lora_r16: 16, lora_r32: 32}`
- Uniform random sampling (no DRO)

### Run summary

NaN cascade at step 1051 / 2500. Killed manually.

```
step 1050: ill=0.0571 primary=1.484 trap=3.5817 well=0.1307  ← stable
step 1051: ill=nan    primary=nan   trap=nan    well=nan      ← cascade
```

Sudden divergence with no warning, identical signature to the original
v4 NaN failure on Falcon (job 386411).

### Diagnosis (provisional)

- LoRA-r32 has 4× the parameter count of LoRA-r8. Inner-loop gradients
  through rank-32 factors can be substantially larger.
- With FOMAML (`use_predictor=False`, `create_graph=False`), gradient
  flow to defender is `∂L_0/∂θ − ∂L_k/∂θ` — which is bounded for
  any single inner step but can have large cross-Hessian terms when
  the high-rank LoRA factor produces large `‖B@A‖_F`.
- Trap output clamp at 10 bounds the *forward* value, but per-step
  trap=3.5817 was within the unclamped regime — clamp didn't save it.
- We hypothesize the rank-32 branch's per-step gradient direction
  occasionally has very large magnitude and `grad_clip=1.0` after the
  global rescaling stops mattering when one component is much larger
  than the others (clip preserves direction, not magnitudes per-axis).

To make Fork B viable, we would need:
- Per-rank gradient clipping (clip each operator's contribution before
  summing into total trap loss), OR
- rsLoRA-style `α/√r` scaling so high-rank operators don't have larger
  effective per-step magnitude, OR
- Drop LoRA-r32 from the family and test `[LP, LoRA-r4, LoRA-r8, LoRA-r16]`
  as a smaller test of rank-extrapolation.

## The synthesis (originally written 2026-05-03, see correction below)

Across **four orthogonal interventions**, LoRA-r8 RFD has not exceeded
~1.2:

| Run | Lever changed | LoRA-r8 RFD |
|---|---|---|
| Stage 4.5 (LP-trap only) | reference (LP only) | 0.65 |
| Stage 5 v1 (multi-op, η=0.01) | inner-loop LR scale | 0.67 |
| Stage 5 v2 (multi-op, η=0.1) | inner-loop LR scale | 1.16 |
| Stage 5 v4a (multi-op, η=0.1, FOMAML) | autograd discipline | 1.14 |
| **Stage 5 v5a (multi-op, η=0.1, FOMAML, DRO)** | **operator weighting** | **1.04** |

Four levers, all in the same RFD band. **The bilevel-trap formulation
as currently expressed has a structural ~1pp LoRA-r8 RFD ceiling.**
This is the most important finding of the day.

What this rules out:
- Inner-loop LR is not the bottleneck (v1 → v2 already moved it 0.67 → 1.16; further increases blow up training).
- Inner-loop autograd discipline is not the bottleneck (v4a same as v2).
- Operator weighting is not the bottleneck (v5a regressed slightly).

What's left to test:
- **Inner-loop strength** — k_inner under FOMAML (now stable, can run k=10 cheaply); Adam-based inner update; iMAML implicit gradient (adversary's *optimum*, not k SGD steps from random init).
- **Trap formulation entirely** — `softplus(ΔL_act)` may be wrong. Alternatives: `||grad of harmful loss at adversary's optimum||²` (bound the optimum, not the trajectory); `KL(adversary outputs ‖ uniform)` (information-theoretic).
- **Threat-model reframing** — maybe LoRA-r8 RFD on Cars is the wrong metric. Per-rank robustness across `[1, 4, 8, 16, 32, 64]` is more honest about what defenders can offer (Agent A null hypothesis #2).

These are no longer hyperparameter-tuning questions; they're research
direction questions.

## Correction (2026-05-03 evening) — the ceiling story is actually stronger

The rank-extrapolation probe (`STAGE5_rank_extrapolation_REPORT.md`)
landed shortly after this report. It tested v4a against LoRA at ranks
{1, 2, 4, 8, 16, 32, 64} and shows: **the 1.1pp RFD at r=8 is rank-
overfit; v4a produces RFD ≈ 0 at every other rank, and even RFD=−1.18
at r=32**.

| Rank | v4a RFD |
|---|---|
| 1  | 0.57 |
| 2  | 0.50 |
| 4  | 0.06 |
| **8** | **1.14** ← trained against |
| 16 | 0.12 |
| 32 | −1.18 |
| 64 | 0.19 |

The "hard ceiling at LoRA-r8 RFD ≈ 1.1" framing in this report is
quantitatively correct *for r=8 only*. The empirically-honest framing,
incorporating rank extrapolation, is:

> **The bilevel-trap formulation produces ~0 generalized LoRA defense
> across the rank distribution, with marginal rank-specific overfitting
> (~1pp) at the trained-against rank.**

This is a stronger negative result than the original "1pp ceiling"
because it rules out the entire continuous family of LoRA ranks, not
just one operator. v6 (robust optimization in LoRA perturbation space)
is therefore positioned as "produce any generalized defense," not
"break a 1pp ceiling." Higher bar for v6, but also a much cleaner
publishable narrative.

Full analysis: `results/STAGE5_rank_extrapolation_REPORT.md`.

## Files

- `results/trap_multiop_v5a_dro_resnet18_cars/{results.json, extractor.pt}` — v5a immunization
- `results/adv_v5a_*_cars/results.json` — five v5a probes
- `configs/immunize_multiop_v5a_dro.yaml` — v5a config (one-line diff vs v4a: `dro_weighting: true`)
- `configs/immunize_multiop_v5b_peft_family.yaml` — v5b config (NaN'd; not re-runnable as-is)
- `src/trap_loss.py` — `trap_loss_multiop` accepts `op_weights` dict for DRO mode

## Single-paragraph summary

> Stage 5 v5 tested the two cheapest forks from the v4a postmortem.
> v5a (DRO weighting) self-defeated: as both operators saturated to
> zero trap value (LP from defender success, LoRA from inner overshoot),
> DRO sampled neither preferentially, RIR plateaued at 0.533, LP RFD
> collapsed from 47.95 to 18.02, and LoRA-r8 RFD did not move (1.14 → 1.04).
> v5b (PEFT-family expansion to ranks {4, 8, 16, 32}) NaN'd at step
> 1051 — same failure signature as the original v4, likely from
> rank-32's per-step gradient magnitude. Combined across four
> interventions (LR, autograd, weighting, operator family), the
> bilevel-trap formulation hits a hard LoRA-r8 RFD ceiling near 1.1.
> Next directions are no longer hyperparameter tuning but trap
> formulation, inner-adversary strength, or threat-model reframing.
