# Empirical State — what we know from runs

Snapshot of what the experiments have settled. Update as new runs land.
Source of truth for run details: `trapping-method/experiments/REGISTRY.md`.

Last updated: 2026-05-02 (after Stage 5 v2 probes).

## Settled facts

1. **CN-only immunization (Zheng et al.) reproduces.** With K⁻¹
   preconditioner + paper-faithful settings, ResNet18 / Cars hits the
   paper's Pareto neighborhood (RIR ≈ 1.1, primary acc ≈ 64%).

2. **RIR is unreliable in both directions.** Stage 4 hit RIR = 5707 with
   primary acc ~ 42% (over-aggressive). Stage 4.5 hit RIR = 1.1 with
   excellent LP RFD = 50.10 (paper-Pareto). Same metric gives wildly
   different signals. **RFD on Cars is the load-bearing metric.**

3. **LP trap is operator-narrow.** Stage 4.5 LP-trap immunization gives
   LP RFD = 50, but LoRA-r8/r32/full-FT-* RFDs ≈ 1 each. Confirmed via
   the operator-transfer matrix (`OPERATOR_TRANSFER_REPORT.md`).

4. **Naive multi-op trap (Plan C v1, η=0.01) does not transfer.** LoRA-r8
   inner loop produces softplus values saturating at log(2) ≈ 0.69 — no
   gradient signal for the defender. LP RFD improved as a side effect
   (50 → 60); LoRA RFD unchanged.

5. **Stronger inner loop produces signal but is unstable.** η=0.1 with
   λ_trap=1.0 generates trap values up to 360 from LoRA branch overshoot
   (B-init-zero + k=3 inner steps); these dominate the gradient and
   collapse primary features within ~1500 steps (seen in jobs 385254,
   385298, 385299, 385310, 385314).

6. **Stable multi-op recipe found** (Stage 5 v2, job 385480):
   - η_inner=0.1, k_inner=3
   - λ_trap=0.3 (down from 1.0)
   - grad_clip=1.0 (down from 5.0)
   - softplus output clamp at 10
   - K_inv triple fallback + SVD ridge + svdvals try/except (numerical safety)
   Result: RIR=1.295, primary acc=64.98%. **First stable run.**

7. **Multi-op trap *does* transfer to LoRA — but barely.** Stage 5 v2
   probes:
   - LP RFD: 60.30 (v1) → **47.18** (v2) — softer LP defense, expected
   - LoRA-r8 RFD: 0.67 (v1) → **1.16** (v2) — first non-noise LoRA defense
   - Other operators: unchanged within noise.
   - **Take:** η=0.1 hypothesis confirmed but the absolute LoRA gap is tiny.

## Open scientific question

The 47-vs-1 ratio (LP RFD / LoRA RFD in v2) is too large to be a
hyperparameter problem. Either:
- (a) The trap loss `softplus(ΔL_act)` for LoRA is too weak — missing the
  predictor term means we penalize *any* loss reduction, including the
  ones LoRA would produce on a totally normal feature extractor (so the
  signal is dominated by adversary noise, not by the defender's structure
  choice).
- (b) The inner loop k=3 / η=0.1 with B-init-zero is too short to reach
  the regime where LoRA actually fits structure; we're penalizing the
  initial-randomness regime, which doesn't transfer to a real adversary
  doing 50 epochs.
- (c) Multi-op uniform random sampling is wrong — defender should
  preferentially train against the operator that *currently* makes
  most progress (DRO / min-max).

(a) and (c) are theory questions; (b) is partially answered by Stage 5
v3 (k=10, in flight as job 386356).

## Decision log

- 2026-05-02: Adopted v2 stable recipe (λ_trap=0.3, grad_clip=1, η=0.1,
  k=3, output clamp=10) as the working baseline for Plan C variants.
- 2026-05-02: Standardized provenance stamping (git SHA + SLURM job ID)
  via `src/provenance.py` for all future runs.
- 2026-05-02: Began this research hub before running more experiments —
  more compute won't close the LP/LoRA RFD gap without theory.
- 2026-05-02: Lit reviews complete (agents A + B). Field map at
  `threads/04_hft_defense_landscape.md`, stability at
  `threads/02_bilevel_adversarial_stability.md`, theory derivation at
  `threads/01_lora_trap_theory.md`. Agent A's load-bearing finding:
  **no LLM defense in open lit has been robustly demonstrated against
  the full Lermen 2024 LoRA attack budget** as of mid-2026. Our 1.16
  LoRA RFD is competitive, not embarrassing.
- 2026-05-02: Adopted Stage 5 v4 = v2 + FOMAML inner loop + Form (c)
  per-step predictor (`trap_loss_lora_v2` in `src/trap_loss.py`). Pure
  ablation against v2 — only `lora_variant: v2` differs. Run as job
  TBD when v3 finishes (or submit in parallel).
- 2026-05-03: v4 NaN'd at step ~2000 on Falcon AND role-lab. Diagnosis:
  predictor's `∂(Σ η‖g_t‖²)/∂θ` introduces a 2× cross-Hessian factor
  with catastrophic cancellation against ΔL_act. v4a (use_predictor=
  false) ran cleanly: RIR=1.580, primary acc=65.04%, LoRA-r8 RFD=1.14
  (= v2's 1.16, within noise). FOMAML alone doesn't move LoRA defense.
  Full analysis: `trapping-method/results/STAGE5_v4a_REPORT.md`.
- 2026-05-03: v5a (DRO weighting) and v5b (PEFT-family expansion) both
  failed in different ways. v5a self-defeated — running-mean DRO
  down-weights operators that saturate, but BOTH saturate (LP from
  defender success, LoRA from inner overshoot to negative ΔL_act).
  Result: LP RFD eroded 47.95→18.02 while LoRA-r8 RFD unchanged
  (1.14→1.04). v5b NaN'd at step 1051 — rank-32 LoRA branch's per-step
  gradient magnitude exceeds what FOMAML+grad_clip=1 can absorb.
  Full analysis: `trapping-method/results/STAGE5_v5_REPORT.md`.
- 2026-05-03 (afternoon): Provisional "hard ceiling" framing — across
  four orthogonal interventions, LoRA-r8 RFD ≤ 1.2.
- 2026-05-03 (evening): **Ceiling claim corrected via rank-extrapolation
  probe.** Tested v4a against LoRA at ranks {1, 2, 4, 8, 16, 32, 64} +
  matching baselines. The 1.14 RFD at r=8 is rank-overfit; RFD ≈ 0 at
  every other rank, and **−1.18 at r=32** (immunized backbone slightly
  *easier* for LoRA-r32 than baseline). True empirical statement:
  bilevel-trap formulation produces ~0 generalized LoRA defense across
  the rank distribution, with marginal ~1pp narrow rank-overfitting at
  the trained-against rank. **Agent A's null hypothesis #2 (operator-
  extrapolation null) confirmed.** Full analysis:
  `trapping-method/results/STAGE5_rank_extrapolation_REPORT.md`.
- 2026-05-04: **RIR replication investigation closed.** Cloned Zheng's
  reference impl (`github.com/amberyzheng/model-immunization-cond-num`),
  diff'd against ours, found 5 specific protocol differences (dtype,
  eigvalsh vs svdvals, ridge, σ_min filter, per-group aggregation),
  patched `src/metrics.py` to be Zheng-faithful by default. Re-scored
  7 candidate extractors. **Best Zheng-faithful RIR = 3.07** on B2
  (CN-only, λ_ill=2e8, K⁻¹ off, primary 60.6%). This **matches both
  Zheng Table 3 (RIR=2.39 ± 0.44) and Sarker Table 1 "CN" row
  (RIR=3.52, primary 62.27%) within run-to-run noise** — so our
  CN-only reproduction is solid. Sarker's "Ours" trap-augmented row
  (RIR=43.92) does NOT reproduce: our CN+trap at paper-literal hyperparams
  gives RIR=0.88, primary 65.69% (primary matches paper, RIR 50× off).
  The narrow non-replication is on the metric Sarker themselves flag
  as unreliable (§4.2). LP RFD continues to match (50.10 vs 47.19).
  Full analysis: `trapping-method/results/STAGE_RIR_REPLICATION_REPORT.md`.
- 2026-05-04: **Bug hunt narrowed.** C1/C2 (CN-only at paperexact and
  lill100) confirm: adding our trap loss strictly DECREASES Zheng-faithful
  RIR (paperexact: 0.90 with trap → 3.97 without trap, 4.4× boost from
  *removing* trap). Opposite direction from paper's claimed 12× boost
  *from adding* trap. Tested two implementation-variant suspects via D1
  (K = X^T X without /B) and D2 (no detach on centroid init). Both
  failed to recover the trap-induced RIR boost — D1 made it worse
  (0.60), D2 marginal (0.82). Both ruled out. Remaining suspects:
  softmax-aware Hessian (paper might use `X^T diag(p(1-p)) X` instead
  of `X^T X`), inner-loop η/k specifics, or a deeper gradient-path bug.
  Cannot resolve definitively without Sarker source code (workshop
  paper, no public repo). The narrow non-replication is now precisely
  characterized: **our trap suppresses κ_H growth in our pipeline; paper's
  trap amplifies it**. RFD reproduction (50.10 vs 47.19) remains intact.

## Lit-informed reframing (2026-05-02)

What we now know:

- **The 360-magnitude trap in v2 attempts 1-5 was second-order MAML
  chain-rule blowup**, not a real signal. TAR (Tamirisa 2024) uses
  FOMAML at k=64 stably; we should never have used `create_graph=True`
  on the inner update. v4 fixes this.
- **η=0.1 is ~100× the literature norm** for inner-loop LR (MAML
  α≈1e-2; Lermen attacks use AdamW lr=1e-4). Hayou 2024 proves a hard
  stable-LR ceiling for B=0 LoRA init that η=0.1 likely violates. v4
  keeps η=0.1 because v2's safety stack absorbs the instability — but
  if FOMAML alone solves it, we may be able to drop η back to 1e-2 in
  v5 and remove the safety nets.
- **Robust overfitting (Rice 2020) will hit our trap defender** the
  way it hits PGD-AT. We need held-out attack early-stopping. Not yet
  built; thread 05 D3 covers it.
- **CTRAP (Yi/Huang 2025) is the closest conceptual collision** —
  same word "trap", different mechanism (collapse-on-detection vs
  basin geometry). Must contrast in any paper. SOPHON (Deng S&P 2024)
  is the methodological predecessor that uses "entrap" before Sarker.
- **PEFT family unification for *defense* is unfilled territory**
  (He et al. 2022 unify for capability only). Potential contribution
  angle once we have a stable LoRA defender.

## Three null hypotheses to test before claiming success

From Agent A's analysis. These are red-team checks we should run *before*
publishing any positive LoRA-defense result.

1. **Capacity tradeoff null.** Bounding the LoRA-reachable subspace may
   destroy benign LoRA-FT utility, reducing our contribution to "we
   evaluate per-operator" (an evaluation contribution, not defense).
   Test: probe immunized backbone with a benign downstream task using
   LoRA fine-tuning. If benign accuracy collapses too, the trap is
   blocking *useful* LoRA, not just harmful LoRA.
2. **Operator-extrapolation null.** Defense conditioned on rank=r₀ may
   fail at rank=r₁≠r₀. Test: train v4 against rank-8 in `trap_operators`,
   probe with rank ∈ {1, 4, 16, 32, 64}. Plot RFD vs rank.
3. **Side-door null.** Wei et al. 2024 show pruning ~3% of safety-
   critical weights or applying untrained low-rank perturbations breaks
   alignment without gradient-based fine-tuning. Test: probe v4 with a
   pure pruning attack and a random-LoRA-perturbation attack.

## What v4 can / cannot tell us

**Can:** Whether the predictor + FOMAML closes some of the LP/LoRA RFD
gap (47 vs 1.16) with the in-distribution rank-8 LoRA adversary. Pure
ablation against v2.

**Cannot:** Whether v4 generalizes across rank/lr/optimizer (need null
hypothesis tests). Whether benign LoRA still works (capacity tradeoff).
Whether held-out attacks bypass it (robust overfitting).

**Resolved 2026-05-03:** v4 NaN'd; v4a confirmed predictor not the
lever (LoRA RFD 1.14 vs v2's 1.16). v5a (Fork A / thread 03) and v5b
(Fork B / thread 04 #5) also tested — both failed. See "hard ceiling"
entry in decision log and `trapping-method/results/STAGE5_v5_REPORT.md`.

## The hard-ceiling result and what it implies (CORRECTED 2026-05-03 evening)

After four ablations all converging to LoRA-r8 RFD ≈ 1.1, AND a
rank-extrapolation probe showing this 1.1pp is rank-overfit:

**Ruled out as the bottleneck:**
- Inner-loop LR magnitude (v1 → v2 moved 0.67 → 1.16; further increases blow up)
- Autograd discipline (FOMAML vs second-order MAML; v4a same as v2)
- Operator weighting (DRO failed; v5a same as v2)
- Operator-set width (v5b NaN'd before completing; provisional)
- **Generalized LoRA defense at any rank** (rank-extrapolation: RFD ≈ 0
  for r ∈ {1, 2, 4, 16, 64}, and −1.18 at r=32; only r=8 has ~1pp RFD)

**Not yet tested (open hypotheses):**
- *Inner-loop strength*: longer k under FOMAML (now stable; can run k=10
  cheaply); Adam-based inner update; iMAML implicit-gradient adversary
  at the optimum, not from random init.
- *Trap formulation entirely*: `softplus(ΔL_act)` may be wrong. Alts:
  `‖∇L_harm at adversary's optimum‖²`; KL-based; loss-flatness-based.
- *Threat-model reframing*: maybe LoRA-r8 RFD on Cars is not the right
  metric; report rank-distribution robustness instead (per Agent A
  null hypothesis #2).

These are research-direction questions, not tuning questions. **Stop
running experiments without a fresh formulation.**
If v4 LoRA RFD < v2 LoRA RFD: FOMAML alone weakens the signal → keep
second-order on the predictor branch only, or revert to v2.
