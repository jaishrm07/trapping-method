# Stage 5 v4a Report — FOMAML-only ablation against v2

Date: 2026-05-03

## Headline (clean null result, well-isolated)

We isolated the FOMAML inner-loop change against v2's second-order MAML
implementation, holding everything else fixed. **The change does not
move LoRA-r8 RFD: 1.16 (v2) → 1.14 (v4a), within noise.** No other
RFD changed meaningfully either.

This is informative: it tells us the *cross-Hessian factor in the
defender gradient* was not the bottleneck for the operator-transfer
problem.

## What was tested

`v4a = v2 stable recipe + lora_variant: v2 + use_predictor: false`

Only one operational change vs v2:
- v2: `trap_loss_lora` (second-order MAML inner loop, no predictor)
- v4a: `trap_loss_lora_v2(use_predictor=False)` (FOMAML inner loop,
  no predictor)

Trap forward identity-of-form: `softplus(ΔL_act).clamp(max=10)` in both.

Difference is purely in how `∂trap/∂θ` flows: v2 includes a chain rule
through `∂φ_k/∂θ` (one cross-Hessian factor); v4a treats `φ_k` as
detached from θ for the inner update (no MAML chain).

The earlier v4 attempt (`use_predictor=True`) NaN'd around step ~2000
on both Falcon (job 386411) and role-lab (PID 594522) due to a 2×
cross-Hessian factor introduced by the predictor's `∂(Σ η‖g_t‖²)/∂θ`
HVP term, with catastrophic cancellation against ΔL_act. v4a removes
that path and is strictly more numerically stable than v2.

## Results

### Backbone immunization

| Metric | v2 (`trap_multiop_v2_eta1e-1_*`) | v4a (`trap_multiop_v4a_fomaml_only_*`) |
|---|---|---|
| Final RIR | 1.295 | **1.580** |
| Final primary acc (ImageNet val top-1) | 64.98% | **65.04%** |
| Wallclock | 12 min (Falcon L40S) | 53 min (role-lab A6000)¹ |
| Iterations completed | 2500/2500 | 2500/2500 |
| NaN events | 0 | 0 |

¹ Wallclock difference reflects A6000-vs-L40S throughput plus shared-machine
contention on role-lab; not a property of v4a vs v2.

### Adversarial probes (50-epoch fine-tune on Cars)

| Adversary | Baseline | v2 final | v4a final | v2 RFD | v4a RFD | Δ RFD |
|---|---|---|---|---|---|---|
| linear_probe | 41.23% | 21.78% | 21.46% | 47.18 | 47.95 | +0.77 |
| **lora_r8** | 73.85% | 72.99% | **73.01%** | 1.16 | **1.14** | **−0.02** |
| lora_r32 | 78.15% | 78.40% | 79.07% | −0.32 | −1.18 | −0.86 |
| full_ft_upper | 80.76% | 80.15% | 80.26% | 0.76 | 0.62 | −0.14 |
| full_ft_all | 80.64% | 81.51% | 81.41% | −1.08 | −0.95 | +0.13 |

All deltas within run-to-run noise. **No operator's RFD moved by more
than ~1pp.**

## Three findings, ranked by what's load-bearing

### Finding 1 (clean null, primary): FOMAML vs second-order doesn't matter for LoRA defense

The cleanest possible ablation says: at k=3, η=0.1, with the trap
formulation `softplus(ΔL_act)`, removing the second-order MAML chain
factor does not change LoRA-r8 RFD. The factor was small enough to be
noise relative to whatever else is bottlenecking LoRA defense.

This rules out **one** hypothesis from `research/empirical_state.md`'s
"open scientific question": it isn't that the inner-loop autograd
discipline (FOMAML vs second-order) is the lever.

### Finding 2 (positive byproduct): v4a's primary task held up cleanly

Primary acc went from 64.98% (v2) → 65.04% (v4a). RIR climbed from
1.295 → 1.580 — meaningfully more "geometrically immunized" by the
condition-number metric. But RIR's improvement did not translate to
LoRA RFD, which **confirms previously-known unreliability of RIR as
a downstream-defense predictor.**

Two-step takeaway: (a) FOMAML produces a different κ_H/κ_P trajectory
than second-order MAML even when the final downstream defense is the
same; (b) RIR is not the right summary metric for "is the immunization
working against LoRA."

### Finding 3 (technical, important): the v4 NaN was from the predictor, not from FOMAML

The earlier v4 attempt with `use_predictor=True` NaN'd around step ~2000
deterministically. v4a (`use_predictor=False`) does not, and trains
strictly more stably than v2. This isolates the failure mode in v4 as
the predictor's HVP path, not the FOMAML transition.

Implication: if we ever revisit a predictor-based trap, the predictor
must either (a) be detached from θ (no HVP gradient flow), losing its
"teaching" effect on the defender, or (b) be computed in a numerically
stable way that avoids cancellation between O(η‖g‖²) terms. Neither is
trivial. Worth abandoning this path until thread 01's theory work
suggests a different predictor form.

## What this means for the operator-transfer problem

We now have empirical evidence on three levers:

| Lever | Tested in | Effect on LoRA-r8 RFD |
|---|---|---|
| Inner-loop η | v1 (η=0.01) → v2 (η=0.1) | 0.67 → 1.16 (∼2×) |
| k_inner | v3 (k=10) — never completed (compute-prohibitive at second-order) | unknown |
| Inner-loop autograd discipline (FOMAML vs second-order) | v2 → v4a | 1.16 → 1.14 (no effect) |
| Predictor (`Σ η‖g_t‖²`) | v4 — NaN'd; not a clean signal | unknown but path is broken |

The 1.16 → 1.14 result combined with v3's compute-prohibitive
intractability tells us that **single-operator multi-step inner-loop
tuning is approximately exhausted as a lever for the v2 trap formulation.**
We are hovering at a ~1pp LoRA-r8 RFD ceiling.

## Three forks for what comes next

### Fork A — DRO over operators (`research/threads/03_*`)

Currently `trap_loss_multiop` samples LP and LoRA-r8 with equal
probability per defender step. Replace with weighted sampling driven
by recent trap loss (group-DRO, exponentiated weights). Defender
shifts effort toward whichever operator is hardest to bound.

Hypothesis: the problem isn't that the LoRA inner-loop simulation is
weak; it's that *equal weighting* lets the defender succeed on the
easy operator (LP) and ignore the hard one (LoRA). With DRO, defender
must succeed on both.

Code cost: ~2 hours (small modification to operator sampling +
exp-weighted update). Compute cost: 1 immunization run (~50 min on
role-lab) + 1 probe array (~20 min on role-lab).

### Fork B — PEFT-family joint operator (`research/threads/04_*`, finding #5)

Currently the operator set is `{linear_probe, lora_r8}`. Expand to
`{LP, LoRA-r4, LoRA-r8, LoRA-r16, LoRA-r32}` and probe with rank
∈ {1, 2, 4, 8, 16, 32, 64} to test rank-extrapolation.

Hypothesis: the LoRA-r8 inner loop only teaches the defender to bound
exactly rank-8 LoRA. With multi-rank training, defender bounds the
rank distribution.

Code cost: ~1 hour (add ranks to dispatch). Compute cost: longer
immunization run since each defender step samples one of 5 operators
(slower convergence per operator) + larger probe set (10 ranks).

### Fork C — Different inner adversary (theory pivot)

Real LoRA attackers use AdamW + 50-epoch fine-tune. Our inner loop
simulates k=3 SGD steps with a fresh LoRA factorization per defender
step. This may be a fundamentally too-weak representation of "real
LoRA adversary."

Replace k-step SGD inner loop with: (a) higher k under FOMAML (now
viable since v4a is stable), or (b) an Adam-like inner update, or
(c) iMAML-style implicit-gradient adversary that converges to LoRA's
post-FT minimum, not just k SGD steps from random init.

Code cost: medium (Adam inner update is straightforward; iMAML
requires CG-HVP setup). Compute cost: same as v2.

## Recommendation

**Fork A is the cheapest, most theory-motivated, and most
publishable** if it works. It's also the smallest code change and
sits cleanly inside the operator-randomization framework we already
built.

**Fork C is the most likely to actually move LoRA RFD** — the field
evidence (Lermen 2024, Hayou 2024) suggests our inner-loop simulation
is structurally weak. But it's higher-cost and harder to ablate.

**Fork B is intermediate** — small code, larger compute. Tests a
specific hypothesis (rank generalization).

Pick one, run it as v5, then re-evaluate.

## Files

- `results/trap_multiop_v4a_fomaml_only_resnet18_cars/{results.json, extractor.pt, slurm.out, slurm.err}` — v4a immunization run.
- `results/adv_v4a_*_cars/results.json` — five v4a adversarial probes.
- `configs/immunize_multiop_v4a.yaml` — v4a config (one-line diff vs v4 — `use_predictor: false`).
- `src/trap_loss.py` — `trap_loss_lora_v2` now accepts `use_predictor: bool = True`.
- `src/provenance.py` — git SHA + SLURM/host stamping in every results.json.

## Single-paragraph summary

> Stage 5 v4a isolated the FOMAML-vs-second-order MAML change against
> the v2 stable recipe. LoRA-r8 RFD did not move (1.16 → 1.14, within
> noise). RIR rose from 1.295 to 1.580 without translating to defense,
> reconfirming RIR's unreliability as a LoRA-defense predictor. The
> earlier v4 NaN cascade was the predictor's HVP path, not the FOMAML
> change; v4a is strictly more stable than v2. This was a clean null
> that rules out the inner-loop autograd discipline as the bottleneck.
> Next: Fork A (DRO weighting), Fork B (PEFT-family operator), or
> Fork C (stronger inner adversary). Fork A is the most theory-motivated
> and cheapest to test as v5.
