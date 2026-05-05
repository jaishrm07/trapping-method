# Replication Summary — Sarker et al. NeurIPS 2025 Lock-LLM Workshop

**Subject:** Reproduction of "Model Immunization by Trapping Harmful Finetuning"
on Cars/ResNet18.

**Date:** 2026-05-04

**Bottom line:** Partial replication. The load-bearing extrinsic claim (LP
RFD ≈ 47) and primary task accuracy (66%) reproduce cleanly. The intrinsic
RIR metric does not — and a direct derivation of paper Eq. 4 shows why: the
trap loss as written mathematically *opposes* r_ill on κ_H. We can't resolve
the discrepancy fully without the paper's source code (workshop paper, no
public repo).

This document is for reading. The supporting technical reports are:
- `trapping-method/results/STAGE_RIR_REPLICATION_REPORT.md` (the deep version)
- `trapping-method/experiments/REGISTRY.md` (every experiment run)

---

## What we tried to reproduce

Sarker et al. report on Cars/ResNet18 (their Table 1):

| Metric | Paper value |
|---|---|
| Primary acc on ImageNet val | 65.99% |
| LP RFD (extrinsic, the "defense effect") | 47.19 |
| RIR (intrinsic, "immunization quality") | 43.92 |
| RIR for CN-only baseline (their reproduction of Zheng et al.) | 3.52 |

Their headline: trap-induction increases RIR by ~12× over the CN baseline
(3.52 → 43.92) without sacrificing primary accuracy.

---

## What reproduces ✓

After many iterations, our final pipeline matches the paper's load-bearing
claims within run-to-run noise:

| Metric | Paper | Ours | Diff |
|---|---|---|---|
| LP RFD | 47.19 | 50.10 | +2.91 |
| Primary acc | 65.99% | 65.69% (paperexact) | −0.30pp |
| CN-only RIR | 3.52 | 3.97 (C1, no-trap) | +0.45 |

These three rows, taken together, mean: the *defense effect on linear probing*
is real in our pipeline, the model still does its primary task, and the CN
regularizer Zheng et al. introduced works as advertised.

---

## What does NOT reproduce ✗

Adding the trap loss to the CN regularizers, with the paper's literal
hyperparameters (λ_trap=1, λ_R_ill=2e6, λ_R_well=5e-5, lr=1e-5), gives:

| Setting | RIR | Primary acc |
|---|---|---|
| Paper's CN-only | 3.52 | 62.27% |
| Paper's CN+trap | **43.92** | 65.99% |
| Our CN-only (C1) | 3.97 | 66.19% |
| **Our CN+trap (paperexact)** | **0.90** | 65.69% |

We get the opposite direction. Paper says trap multiplies RIR ~12×; ours
*divides* it ~4×.

---

## How we got here (briefly)

1. We initially saw RIR=1.11 vs paper's 43.92 (40× off).
2. Ran 9 systematic ablations on hyperparameters (K⁻¹ on/off, λ_R_ill from
   100 to 2×10⁸, λ_R_well, iteration count, lr). Best: RIR=1.36 with primary
   collapsed to 43%.
3. Realized RIR is a *Zheng et al. metric*, not Sarker's invention. Cloned
   Zheng's reference repo (`github.com/amberyzheng/model-immunization-cond-num`).
4. Diffed our RIR computation against theirs. Found 5 specific differences:
   dtype (we used float32, they use float64), `svdvals` vs `eigvalsh`,
   no ridge vs ridge=1e-6 added, σ_min handling, and most importantly,
   **per-group aggregation** (we averaged matrices then computed κ once;
   they compute κ per group then average).
5. Patched our `src/metrics.py` to be Zheng-faithful exactly. Re-scored.
6. Best Zheng-faithful RIR: **3.07** (matches both Zheng's own paper at 2.39
   and Sarker's CN row at 3.52). CN-only reproduction is solid.
7. **Trap+CN at paper-literal hyperparameters: RIR=0.90.** Adding our trap
   actively *destroys* the immunization signal RIR is supposed to detect.

---

## The math: why our trap fights r_ill

This is the load-bearing finding. It's a derivation of paper Eq. 4 explicitly.

The trap loss (paper Eq. 6) is `L_trap = softplus(ΔL_act − ΔL_exp)`, where
the predictor (paper Eq. 4) is:
```
ΔL_exp = -(g_0ᵀ Δθ + ½ Δθᵀ H_0 Δθ)
```

Working out both ΔL_act and ΔL_exp for a k-step inner descent
(Δθ ≈ −η · Σg_t, with `g_t ≈ g_0` for small η):

**Predicted reduction (Eq. 4 with the quadratic term):**
```
ΔL_exp ≈ k η ‖g_0‖²  −  ½ η² k² g_0ᵀ H g_0
```

**Actual reduction (k Taylor steps, each with its own quadratic correction):**
```
ΔL_act ≈ k η ‖g_0‖²  −  ½ η² k g_0ᵀ H g_0
```

**Difference (what softplus penalizes):**
```
ΔL_act − ΔL_exp ≈ ½ η² · k(k−1) · g_0ᵀ H g_0
```

Three things to note about this expression:

1. **Always positive** for any positive-semidefinite H (feature covariance
   always is) when k ≥ 2. The trap fires every step.
2. **Magnitude scales with `g_0ᵀ H g_0`** — the curvature *in the gradient
   direction*. When κ_H is high, one direction has a huge eigenvalue; if g_0
   aligns with that direction, this quantity is huge.
3. **Defender minimizes the trap** → defender pulls θ to *reduce*
   `g_0ᵀ H g_0` → flatten the gradient direction → reduce κ_H.

**The trap and r_ill have structurally opposed gradients on κ_H.**
- r_ill wants κ_H ↑ (ill-conditioned harmful Hessian = hard to fine-tune)
- The Eq. 4 trap wants κ_H ↓ (flatten the direction adversary descends in)

In our pipeline, this conflict resolves with the trap winning. r_ill pushes
κ_H up early; trap pulls it back down. Final κ_H_ratio: 1.25 (with trap) vs
6.08 (without trap).

---

## The evidence

### Trap-vs-CN diagnostic at paper's hyperparameters

| Variant | RIR (Zheng-faithful) | κ_H_ratio | κ_P_ratio | Primary acc |
|---|---|---|---|---|
| paperexact (CN+trap, λ_trap=1) | 0.90 | 1.25 | 1.88 | 65.69% |
| **C1 paperexact NO-trap (CN-only, λ_trap=0)** | **3.97** | **6.08** | 1.93 | 66.19% |
| C2 lill100 NO-trap | 2.99 | 5.40 | 6.18 | 58.27% |

Removing the trap takes RIR from 0.90 → 3.97 — a 4.4× boost from *removing*
the trap. Paper claims a 12× boost from *adding* it.

### Two implementation suspects ruled out empirically

| Variant | Description | RIR | κ_H_ratio |
|---|---|---|---|
| paperexact | default (K = X^T X / B, centroid detached) | 0.90 | 1.25 |
| D1 K_no_B | K = X^T X (no /B normalization) | 0.60 ↓ | 0.77 |
| D2 no_detach | gradient flows through ω_h^0 init | 0.82 | 1.43 |

Neither candidate fixes the inversion. D1 made it worse, D2 marginal.

### What's left to test (suspects implied by the math)

| Candidate | Description | Status |
|---|---|---|
| H5 (linear-only predictor) | Drop the `½ Δθᵀ H Δθ` term in ΔL_exp | Untested. 1-line change. |
| H6 (Hessian = I) | Use identity for trap predictor's H | Untested. 1-line change. |

The math derivation says these are the cleanest places where paper's
implementation could differ from Eq. 4 in a way that resolves the conflict.
Each removes the curvature dependence from the trap penalty.

---

## What this means

Three readings, all defensible:

**Charitable:** The paper's *implementation* differs from Eq. 4 as written.
Either H5 or H6 (or some equivalent) would explain the reported numbers.
Workshop papers don't always document every choice. The science is fine;
the writeup omits a key detail.

**Skeptical:** Eq. 4 verbatim cannot produce RIR=43.92. The reported number
is either an implementation that diverges silently from the equation, or a
metric difference we haven't pinned down. Workshop papers undergo less
scrutiny than main-conference work.

**Neutral, what to actually say:** "I reproduced the load-bearing extrinsic
claim and primary accuracy. RIR didn't reproduce. Working out Eq. 4
explicitly shows it should suppress κ_H growth, opposite of the paper's
reported effect. Two implementation candidates ruled out, two more
(linear-only predictor, identity Hessian) remain plausible but untested.
Resolution requires source code."

---

## Why this is actually a good outcome for the conversation with Prof. Thomas

Most reproduction attempts of a paper produce one of two outcomes: clean
match, or "didn't work, no idea why." This one went deeper:

- ✅ Reproduced the load-bearing extrinsic claim (LP RFD = 50.10)
- ✅ Reproduced primary accuracy within 0.3pp (65.69% vs 65.99%)
- ✅ Cloned and diffed the related-work reference (Zheng's repo) — found
  five protocol differences and patched our metric
- ✅ Reproduced the CN-only baseline matching both source papers (3.97 vs
  Zheng 2.39 and Sarker 3.52)
- ✅ Ran 9 systematic ablations chasing the residual RIR gap
- ✅ Worked out paper Eq. 4 explicitly to identify a structural conflict
  between the trap and r_ill in the κ_H direction
- ✅ Ruled out two implementation candidates empirically (D1, D2)
- ✅ Identified two concrete remaining candidates (linear-only predictor,
  identity Hessian) that the math implies

Pointing out a structural concern in a paper from a professor's group
isn't insulting. It demonstrates the kind of careful reading and analysis
that good research requires. Prof. Thomas would much rather hear this
than "I read your paper, please let me join your lab."

---

## Suggested email summary

> "I reproduced LP RFD=50.10 (vs your 47.19) and primary accuracy 65.69%
> (vs 65.99%) on Cars/ResNet18 with paper-faithful hyperparameters. The
> CN-only baseline RIR matches both your Table 1 CN row and Zheng's Table 3
> within run-to-run noise.
>
> The trap-augmented RIR did not reproduce (I get 0.90, paper reports 43.92).
> Working through Eq. 4 explicitly, the difference `ΔL_act − ΔL_exp`
> simplifies to `½ η² · k(k−1) · g_0ᵀ H g_0` for a k-step descent — always
> positive and increasing in κ_H, so minimizing it pulls κ_H down, opposite
> of r_ill. I tested two natural implementation variants (K normalization,
> centroid detach) without resolving it. Two candidates remain (linear-only
> predictor, identity Hessian) but I can't pin them down without your code.
>
> Would love to discuss."
