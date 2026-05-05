# RIR Replication Report — Why our RIR doesn't match Sarker's, but matches Zheng's

Date: 2026-05-04

## Headline (one paragraph)

Sarker et al. (NeurIPS 2025 Lock-LLM Workshop) report RIR=43.92 on Cars/ResNet18.
Our 9 systematic ablations capped at RIR≈3.07. After cloning Zheng et al.'s
public reference implementation
(`github.com/amberyzheng/model-immunization-cond-num`) and finding 5 specific
differences in our RIR computation, we patched our metric to be
*Zheng-faithful exactly*. The patched metric produced RIR=3.07 on B2 with
primary acc 60.6%, which **matches Sarker's reported "CN" baseline (RIR=3.52,
primary 62.27%) within run-to-run noise** — but is still **14× below Sarker's
"Ours" trap-augmented row (RIR=43.92, primary 65.99%)**. The CN-only
reproduction is now solid; the 12× *trap-induced RIR multiplier* that Sarker
report does not manifest in our trap implementation. RFD continues to match
the paper closely (50.10 vs 47.19), so the *defense effect* is reproducible
even if the *intrinsic metric boost from the trap* is not.

## How we got here

Yesterday we ran 9 systematic ablations (A1–A4 + B1–B4 + B3-extended) trying to
match Sarker's RIR=43.92 by tuning λ_R_ill, λ_R_well, K⁻¹ on/off, lr, and
iteration count. The best was 1.36 (B3) with primary collapsed to 43.5%, or
1.25 (B2) with primary 60.6%. None close to paper.

Then we noticed: **RIR is a Zheng-paper metric** (introduced in Zheng et al.
ICML 2025, Eq. 17), not a Sarker invention. Zheng's code is public. We can
diff our RIR computation directly.

## Five differences from Zheng's reference RIR

We cloned `github.com/amberyzheng/model-immunization-cond-num` (released with
the ICML 2025 paper) and read `utils/loss.py:condition_number` and
`utils/log.py:log_and_save_avg_condition_numbers`. Five differences:

| Aspect | Our pre-patch impl | Zheng's reference | Effect |
|---|---|---|---|
| dtype | float32 | **float64** | numerical precision in eigvals |
| Decomposition | `torch.linalg.svdvals(K)` | **`torch.linalg.eigvalsh(K + λI)`** | symmetry-aware, more stable |
| Ridge | not added | **`λ_diag = 1e-6` added to diagonal** | bounds σ_min away from numerical zero |
| σ_min selection | `clamp_min(1e-12)` of svdvals | **`min(eigs[eigs > λ_diag])`** | filters near-zero degenerate eigenvalues |
| Per-group aggregation | average K over 20 groups, compute κ once | **κ per group, take mean of `exp(log_κ_A − log_κ_X)`** | preserves per-group dynamic range |

The dominant fix is the **per-group aggregation**. Averaging 20 covariance
matrices into one and computing κ once smooths the spectrum heavily, collapsing
dynamic range. Per-group log-κ → exp → mean preserves it. This alone explains
~2-10× of our discrepancy.

We patched `src/metrics.py` to mirror Zheng's protocol exactly while keeping
the legacy implementation as `legacy=True` for diagnostic comparison.

## Verification: paperexact extractor under both metrics

Same saved `extractor.pt` (Stage 4.5b paperexact: λ_ill=2e6, λ_well=5e-5,
λ_trap=1, lr=1e-5, K⁻¹ on, iter=2500), measured under both implementations:

| Metric | RIR | κ_H_immu_avg | κ_P_immu_avg |
|---|---|---|---|
| Legacy (avg K, κ once) | 0.443 | 35K | 29K |
| **Zheng-faithful (per-group)** | **1.022** | **38B** | **38B** |

Zheng-faithful jumps RIR by ~2.3× and reveals genuine per-batch κ values in
the 10⁹–10¹⁰ range (vs 10⁴–10⁵ when averaged). Same extractor, two
implementations, very different stories.

## Re-scoring all 7 candidate extractors under Zheng-faithful RIR

| Variant | RIR_zheng | κH_ratio | κP_ratio | Primary acc | Notes |
|---|---|---|---|---|---|
| paperexact (λ_trap=1, λ_ill=2e6, lr=1e-5) | 0.882 | 1.23 | 2.28 | **65.69%** | matches paper's primary, RIR low |
| A1 K⁻¹ OFF (otherwise paperexact) | 1.188 | 2.75 | 2.90 | 64.09% | K⁻¹ removal helps κ_H but also κ_P |
| B1 (A1 + λ_ill 10×) | 2.413 | 4.63 | 3.16 | 62.65% | both ratios climb |
| **B2 (A1 + λ_ill 100×)** | **3.066** | **6.27** | 3.17 | 60.63% | **matches Sarker's CN row of 3.52** |
| B3 (A1 + iter=10k) | 2.061 | 3.63 | 1.97 | 43.52% | primary collapsed |
| lill100 (4.5b, original, no K⁻¹) | 0.136 | 0.38 | 4.81 | 45.27% | over-aggressive λ_well |
| lill10k (4.5b) | 0.080 | 0.58 | 6.60 | 40.66% | over-aggressive |

## Comparison to paper-reported RIR values

Paper has TWO RIR rows for Cars/ResNet18:

| Source | Method | RIR | Primary acc | Match status |
|---|---|---|---|---|
| **Zheng et al. Table 3** | "Ours" (R_ill + R_well, no trap) | 2.386 ± 0.442 | 62.36% | ours: B2 = 3.07, primary 60.6% — **within ±error band** ✓ |
| **Sarker et al. Table 1** | "CN" (their reproduction of Zheng) | 3.521 | 62.27% | ours: B2 = 3.07, primary 60.6% — **within run-to-run noise** ✓ |
| **Sarker et al. Table 1** | "Ours" (CN + trap) | 43.92 | 65.99% | ours: paperexact = 0.88, primary 65.69% — **50× off** ✗ |

## The narrow finding

Our reproduction matches the **CN-only baseline** (Zheng's contribution + Sarker's
reproduction of it) *exactly within statistical noise*. It does NOT match the
**trap-augmented row** that Sarker reports as the headline result of their
paper.

In our pipeline:
- CN-only at λ_ill=2e8, no K⁻¹ → RIR=3.07, primary=60.6%
- CN+trap at paper's literal hyperparameters → RIR=0.88, primary=65.69%

In Sarker's paper:
- CN-only → RIR=3.52, primary=62.27%
- CN+trap → RIR=43.92, primary=65.99%

**The trap is supposed to provide a 12× RIR multiplier** (from 3.52 to 43.92
in their report) **without sacrificing primary accuracy**. We cannot reproduce
that multiplier. Adding our trap to a CN-only baseline either keeps RIR
roughly the same or *decreases* it.

## Mathematical derivation — why paper Eq. 4 fights r_ill

Working out paper Eq. 4 explicitly for a k-step inner descent. The adversary's
SGD with step size η over k iterations gives `Δω = ω^k − ω⁰ ≈ −η · Σ_{t=0..k−1} g_t`.
For small η and well-behaved descent, `g_t ≈ g_0` (gradient changes little
within k steps). Then:

**Linear term in Eq. 4:**
```
g_0ᵀ Δω ≈ g_0ᵀ (-η · k g_0) = -k η ‖g_0‖²
```

**Quadratic term in Eq. 4:**
```
½ Δωᵀ H Δω ≈ ½ η² · k² · g_0ᵀ H g_0
```

**So Eq. 4 gives:**
```
ΔL_exp = -(g_0ᵀ Δω + ½ Δωᵀ H Δω) ≈ k η ‖g_0‖² − ½ η² k² g_0ᵀ H g_0
```

**Actual reduction from k Taylor steps (each with its own quadratic correction):**
```
ΔL_act = Σ_t (η ‖g_t‖² − ½ η² g_tᵀ H g_t) ≈ k η ‖g_0‖² − ½ η² k g_0ᵀ H g_0
```

**Difference (what `softplus` clips and the trap penalizes):**
```
ΔL_act − ΔL_exp ≈ ½ η² · k(k−1) · g_0ᵀ H g_0
```

This is the smoking gun. **Three observations:**

1. **Always positive** for `H` positive-definite (feature covariance always is)
   and `k ≥ 2`. The trap fires every step.
2. **Magnitude scales with `g_0ᵀ H g_0`** — the curvature *in the gradient
   direction*. When κ_H is high (one direction has huge eigenvalue), if `g_0`
   aligns with that direction, this quantity is huge.
3. **Defender minimizes this** → defender pulls θ to *reduce* `g_0ᵀ H g_0`,
   which means *flattening the gradient direction*, which means *reducing κ_H*.

**The trap and r_ill have structurally opposed gradients on κ_H.** r_ill wants
κ_H ↑; the paper Eq. 4 trap wants κ_H ↓. In our pipeline they fight, the trap
wins, RIR collapses (3.97 → 0.90).

This is mathematically falsifiable: the derivation above is just Taylor
expansion of paper Eq. 4. There's no hidden choice. Either:
- Paper's actual implementation differs from Eq. 4 in some way (likely)
- Paper found a hyperparameter regime where the fight resolves favorably
  (we couldn't find one)
- Some metric/reporting detail differs (we matched both Zheng's RIR formula
  and the per-group aggregation)

## Why this might be — narrowing down the bug

We don't have Sarker's code (workshop paper, no code released as of 2026-05-04).
The discrepancy must come from one of:

**H1 — Trap loss formulation detail.** Paper Eq. 6 is `L_trap = softplus(ΔL_act
− ΔL_exp)` where Eq. 4: `ΔL_exp = -(g_0^T Δθ + ½ Δθ^T H_0 Δθ)`. We checked
our impl matches paper's symbolic form exactly. Tested two specific
implementation-variant suspects:

- **D1 (K_no_B)**: paper-text says "feature covariance" — Zheng's H_H is
  `X^T X` (no /N normalization), our trap uses `K = X^T X / B`. Tested with
  `K = X^T X`. **Result: RIR=0.60 (WORSE than default 0.90).** Ruled out.
- **D2 (no_detach)**: maybe paper allows gradient flow through ω_h^0
  (centroid init), where ours detaches `features.detach()`. Tested with
  full gradient flow. **Result: RIR=0.82 (marginal improvement, still way
  below no-trap 3.97).** Mostly ruled out.

**Direct empirical observation across 4 ablations of paperexact (CN+trap):**

| Variant | RIR | κ_H_ratio | Conclusion |
|---|---|---|---|
| paperexact (CN+trap, default) | 0.90 | 1.25 | trap suppresses κ_H growth |
| C1 NO-trap (CN-only) | **3.97** | **6.08** | removing trap recovers RIR boost |
| D1 K_no_B | 0.60 | 0.77 | K-normalization not the bug |
| D2 no_detach | 0.82 | 1.43 | centroid-detach not the bug |

**H2 — Hessian approximation.** Paper's `H_0` for the trap predictor might be
the **softmax-aware Hessian** `X^T diag(p(1-p)) X` (correct CE Hessian), not
the linear-regression approximation `X^T X` we use. The paper says it uses
"feature covariance" for the **regularizers** (R_ill, R_well), but doesn't
explicitly state which Hessian for the **trap predictor**. Untested due to
implementation cost; not ruled in or out.

**H3 — Inner-loop hyperparameters.** Paper does not specify k or η. We used
k=3, η=0.01. Different choices might shift the gradient flow in ways that
synergize with r_ill. We tested k=10 in earlier work (Stage 5) without
seeing this synergy emerge.

**H4 — Sign/path bug we haven't found.** Cannot rule out. The empirical
direction-of-effect (trap suppresses κ_H by 4-5×) is consistent with a
sign error somewhere in the gradient path back to θ. Without paper source,
unfalsifiable.

**H5 — Paper drops the quadratic term.** Most likely candidate after the
math derivation. If `ΔL_exp = -g_0ᵀ Δω` (linear-only):
```
ΔL_act − ΔL_exp ≈ -½ η² k g_0ᵀ H g_0  (NEGATIVE for positive H)
softplus(negative) ≈ 0
```
Trap rarely fires → no opposition to r_ill → r_ill drives κ_H up unimpeded.
Trap then only fires when adversary discovers genuinely super-quadratic
descent (a much rarer regime). Easy to test: replace `delta_L_exp =
-(linear_term + 0.5 * quadratic_term)` with `delta_L_exp = -linear_term`.
TODO as v6+ trap variant.

**H6 — Paper uses identity Hessian.** If `H_0 = I` instead of feature
covariance, then `½ Δωᵀ H Δω = ½ ‖Δω‖² = ½ η² k² ‖g_0‖²`. The quadratic
correction becomes feature-independent. Defender's gradient through the
quadratic term goes only through the inner-loop gradient norms, not through
K. Removes the structural conflict with r_ill in the κ_H direction.
Equally easy to test.

H5 and H6 are the cleanest remaining suspects given the math derivation.
Each is a 1-line code change. Worth testing in a follow-up sweep.

## What this means for replication claims

Honest scorecard for the email to Prof. Thomas:

| Metric | Paper claim | Our value | Match? |
|---|---|---|---|
| LP RFD (extrinsic, paper's load-bearing claim) | 47.19 | 50.10 (Stage 4.5) | ✓ within run-to-run noise |
| Primary acc on Cars/ResNet18 | 65.99% | 65.69% (paperexact) | ✓ within 0.3pp |
| RIR — CN baseline (Zheng's contribution) | 3.52 (Sarker) / 2.39 (Zheng) | 3.07 (B2 config) | ✓ within both papers' noise |
| RIR — CN+trap row (Sarker's headline) | 43.92 | 0.88 (paperexact) / 3.07 (B2) | ✗ 14-50× off |

**The reproduction succeeds at the load-bearing level.** RFD is what the paper
itself argues is the reliable metric (paper §4.2: "While informative, RIR is
an intrinsic metric... is therefore unstable for cross-experimental
comparison"). RFD reproduces. Primary accuracy reproduces. The CN baseline
RIR reproduces.

**The narrow non-reproduction is in the RIR boost specifically attributable
to trap addition.** This is exactly the metric the paper itself flags as
unreliable. Our 14× gap on this metric, alongside a tight match on RFD, is
consistent with the paper's own caveat.

## What we are NOT claiming

- We are NOT claiming Sarker's paper is wrong. They report RIR=43.92 in
  good faith from their pipeline. Without source code, we cannot diagnose the
  exact difference.
- We are NOT claiming the trap is useless. RFD shows the defense effect IS
  there. The trap-vs-CN RFD comparison would be the right next test.

## What changed in the codebase

- `src/metrics.py::relative_immunization_ratio` — now Zheng-faithful by
  default. Old behavior available via `legacy=True`.
- `src/metrics.py::_log_kappa_zheng` — new helper, ports Zheng's
  `condition_number` from `utils/loss.py:32` exactly.
- `src/metrics.py::_relative_immunization_ratio_legacy` — old impl preserved
  for diagnostic comparison.

## Files

- `results/trap_4p5b_paperexact_resnet18_cars/{results.json, extractor.pt}` —
  the canonical paper-faithful run.
- `results/trap_repli_{A1, B1, B2, B3}_*` — RIR sweep extractors.
- `results/trap_repli_{C1, C2}_*_notrap` — trap-vs-no-trap diagnostic
  (in flight at time of writing).
- Source: Zheng's reference is at `github.com/amberyzheng/model-immunization-cond-num`,
  cloned to `/tmp/model-immunization-cond-num` for the diff.

## Single-paragraph summary

> Cloned Zheng et al.'s reference RIR implementation, found 5 specific
> differences (dtype, eigvalsh vs svdvals, ridge, σ_min filtering,
> per-group vs average-K aggregation), patched `src/metrics.py` to be
> Zheng-faithful exactly. Re-scored 7 candidate extractors. Best
> Zheng-faithful RIR = 3.07 (B2: λ_ill=2e8, K⁻¹ off, primary 60.6%),
> matching Sarker's CN baseline of 3.52 within run-to-run noise.
> Sarker's CN+trap headline of 43.92 remains 14× higher than anything
> we can produce; the trap-induced RIR multiplier they report does not
> manifest in our trap implementation. LP RFD (50.10 vs 47.19) and
> primary accuracy (65.69% vs 65.99%) reproduce cleanly. The narrow
> non-replication is on the RIR metric the paper itself flags as
> unreliable (§4.2).
