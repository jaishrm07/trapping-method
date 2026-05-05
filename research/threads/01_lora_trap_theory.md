# Thread 01 — LoRA trap theory

**Status:** in progress (Derivation v1 below, 2026-05-02)
**Type:** theory
**Owner:** —

## Question

Can we derive a tractable closed form for the trap loss when the inner
adversary uses LoRA-rank-r fine-tuning? The paper's Eq. 6 was derived
for linear probing where the adversary's free parameter `ω_H` has a
quadratic loss landscape (CE on `features @ ω_Hᵀ`). For LoRA, the
adversary's free params are `(LoRA_A, LoRA_B, ω_H)` — Bᵀ·A factors over
each conv plus a head — and the harmful loss is *not* quadratic in the
factors (it has the bilinear `B@A` coupling).

## Why this matters

In our current implementation `trap_loss_lora` returns
`softplus(ΔL_act)` only — no predictor term. We're penalizing *any*
realized harmful loss reduction over the inner loop, regardless of
whether that reduction matches the local geometry's expectation. This
is plausibly why the LoRA branch produces unstable signal and why the
defender doesn't learn a LoRA-aware feature structure.

## What "tractable" looks like

A predictor `ΔL_exp_LoRA` that is:
- A scalar function of features and a small subset of weights (so it
  can be computed in O(B·D²) per step like Eq. 4 was).
- Differentiable w.r.t. θ_upper (the defender's trainable params).
- Quadratic-or-better approximation to the actual `L(θ⁰) − L(θᵏ)` over
  the LoRA inner loop.

## Candidate approaches

1. **Restrict the Hessian to the LoRA subspace.** The full Hessian over
   `(A, B, ω_H)` is intractable. But the LoRA delta `B@A` lives in a
   rank-r subspace of the conv tensor — there are only `r·(C_in·k² + C_out)`
   free params per conv layer. Per-layer, that's a small enough
   parameter count to admit a Gauss-Newton or empirical-Fisher
   approximation.

2. **Linearize around B=0.** Since LoRA initializes B=0 (so initial
   delta is zero), the first-order behavior of the inner loop is
   determined by `∂(B@A)/∂B = A` and `∂(B@A)/∂A = B = 0` initially. So
   the early inner steps are dominated by B's gradient — `ΔL_act ≈
   −η·||∂L/∂B|²` to first order. This gives a *very* tractable
   predictor: the trap loss becomes `softplus(ΔL_act + η·||∂L/∂B||²)`
   which only requires one extra backward pass through `B`.

3. **Implicit-function-theorem inner solution.** Treat the LoRA inner
   loop as solving an implicit fixed-point and back-prop through the
   solution (iMAML, Rajeswaran et al. 2019). Avoids unrolling the
   inner-loop graph entirely. May be faster than k=10.

4. **Closed-form trap for low-rank linear adversary.** If we restrict
   the LoRA inner loop to a *single* linear layer (no conv, no
   nonlinearity), the loss becomes quadratic in the rank-r factors and
   admits a paper-style Eq. 6 derivation. This is a toy version we
   could solve on paper and use as a sanity check.

## Notes / scratchpad

- Reading list (TODO):
  - Hu et al. 2021 — original LoRA paper, for parameterization details.
  - Finn et al. 2017 (MAML) + Rajeswaran et al. 2019 (iMAML) — bilevel
    backprop tricks.
  - Aghajanyan et al. 2020 — intrinsic dimensionality of fine-tuning.
- Open question: does the LoRA inner-loop's effective rank during the k
  inner steps actually stay rank-r, or does the optimizer find higher-
  rank solutions if the loss favors them? Empirically check.

## Derivation v1 (2026-05-02)

### Setup

- Defender params: θ (the trainable upper backbone weights). Per conv,
  call the unfolded weight `W ∈ ℝ^{C_out × (C_in·k²)}`.
- Frozen lower features `z = lower(x)` for a harmful batch.
- Adversary's free params: `φ = (A, B, ω_H)`, where for each conv:
  - `A ∈ ℝ^{r × C_in·k²}`
  - `B ∈ ℝ^{C_out × r}`
  - and the head `ω_H ∈ ℝ^{n_c × D_hid}`.
- Effective weight under LoRA: `W_eff = W + B A`.
- Inner-loop loss: `L(θ, φ) = CE(features(W_eff)(z) · ω_Hᵀ, y_H)`.
- Standard LoRA init: `A` Kaiming, `B = 0`, `ω_H` from KNN centroid.
- Inner update: `φ^{t+1} = φ^t − η · ∇_φ L(θ, φ^t)` for `t = 0..k−1`.
- Realized reduction: `ΔL_act = L(φ^0) − L(φ^k)`.

### Goal

Find a tractable scalar `ΔL_exp_LoRA(θ, φ^0, η, k)` such that
`softplus(ΔL_act − ΔL_exp)` penalizes only **excess** reduction beyond
what local geometry justifies — analog of the paper's Eq. 6.

### Three candidate forms, in order of cost

**Form (a) — Second-order Taylor (general k).**

Standard expansion:

```
L(φ^k) ≈ L(φ^0) + g_0ᵀ Δφ + ½ Δφᵀ H Δφ
ΔL_exp = −(g_0ᵀ Δφ + ½ Δφᵀ H Δφ)
```

where `g_0 = ∇_φ L|_{φ^0}`, `H = ∇²_φ L|_{φ^0}`. The Hessian `H` is huge
(blocks over A, B, ω_H), but `H Δφ` is computable in two backward passes
via the standard Hessian-vector-product trick:

```python
g0 = autograd.grad(L_0, φ, create_graph=True)
hvp = autograd.grad((g0 * Δφ.detach()).sum(), φ)  # = H @ Δφ_detached
```

`Δφᵀ H Δφ = (Δφ_detached · hvp).sum()`. Cost: ~3× current `trap_loss_lora`
(extra forward + 2 extra backwards). Doable.

**Form (b) — Single-step Taylor (k=1 truncation).**

This is the structurally cleanest case, and it has a striking simplification.

At `φ^0 = (A^0, 0, ω_H^0)` (B=0):

- `∂L/∂A|_{B=0} = 0`. Reason: `W_eff = W + BA`, and any partial through A
  passes through B in the chain rule; with B=0 the contribution vanishes.
- `∂L/∂B|_{B=0} = G · Aᵀ`, where `G = ∂L/∂W_eff` evaluated at `W_eff = W`.
- `∂L/∂ω_H|_{φ^0}` is the standard CE gradient `g_ω`.

So `g_0 = (0, GAᵀ, g_ω)`. Single inner step:
- `A^1 = A^0` (gradient was zero)
- `B^1 = −η G Aᵀ`
- `ω_H^1 = ω_H^0 − η g_ω`

Linear term in Taylor:
```
g_0ᵀ Δφ = 0·ΔA + (GAᵀ)·(−η GAᵀ) + g_ω·(−η g_ω)
        = −η (‖GAᵀ‖_F² + ‖g_ω‖²)
        = −η ‖g_0‖²            ← all blocks summed
```

Therefore `ΔL_exp ≈ η ‖g_0‖²`. The trap loss becomes:

```
trap_loss_lora_v2(k=1) = softplus(ΔL_act − η · ‖g_0‖²)
```

**This is the same form as the paper's LP trap** with the predictor
specialized to a single SGD step. Reading: a step of size η along
gradient g should reduce loss by `η‖g‖²` to first order. If actual
reduction matches, trap saturates at `log(2)` — no defender pressure.
If actual *exceeds* the prediction, the harmful loss has favorable
local curvature for the adversary; the defender pulls features to flatten
that.

Cost: 1 forward + 1 backward to get `g_0`, then 1 forward to get
`L(φ^1)`. **Cheaper than current k=3 implementation.**

**Form (c) — Per-step linear bound, summed (general k).**

Generalize form (b) by accumulating the predictor over k inner steps:

```
ΔL_exp ≈ Σ_{t=0..k−1} η · ‖g_t‖²
```

Each `g_t` is the inner-loop gradient at step t — already computed by
the inner-loop autograd unroll. So this requires **zero extra compute**:
just track the sum of squared gradient norms during the inner loop.

This is loose for k>1 (ignores cross-step curvature) but cheap. It also
has a clean interpretation: defender penalizes any reduction that exceeds
the cumulative "first-order budget" along the actual SGD path.

### Recommendation — what to implement first

Implement **Form (b)** as `trap_loss_lora_v2`, run as Stage 5 v4. Three
reasons:

1. Cheapest to add — ~10 lines diff vs current.
2. Cheapest to run — fewer ops than current k=3 trap.
3. Pure ablation: differs from current implementation by exactly one
   change (presence of the predictor term).

Test plan (v4):
- Same v2 stable recipe (η=0.1, λ_trap=0.3, grad_clip=1, output clamp 10).
- Replace `trap_loss_lora` with `trap_loss_lora_v2` (k=1, with predictor).
- Compare LoRA RFD vs v3 (k=10, no predictor) and v2 (k=3, no predictor).

Outcomes interpret as:
| v4 vs others | What it tells us |
|---|---|
| v4 > v3 ≈ v2 | predictor matters more than k. Implement (a) for further gains. |
| v3 > v4 ≈ v2 | k matters more than predictor; keep v3-style. |
| v4 ≈ v3 ≈ v2 | both axes are saturated; pivot to thread 03 (DRO weighting) or 04 (different operator family). |

### Open questions left to thread 01

- Is **Form (a)** strictly better than (b)+(c) for k>1? Worth implementing
  once (b) and (c) results are in.
- Does the LoRA inner loop's effective rank stay rank-r empirically, or
  does the optimizer find higher-rank solutions? Connect to thread 05's
  D5 diagnostic (delta-norm tracking).
- Is there a closed-form *minimization* of L over (A, B, ω_H) for fixed
  features, by treating B as an unknown linear regression coefficient on
  a fixed A? If yes, defender could match against the *optimum* adversary
  rather than a finite-k inner loop.

## Acceptance criterion

Thread closes when we have either:
- A derivation of `ΔL_exp_LoRA` that we can plug into `trap_loss_lora`
  in 1–2 hours of coding, **OR**
- A clear argument for why no such tractable form exists, motivating a
  different approach (DRO, ensemble, etc.).

**Status as of 2026-05-02:** First criterion partially met by Form (b) above.
Awaiting Stage 5 v4 empirical test. Form (a) reserved as escalation.
