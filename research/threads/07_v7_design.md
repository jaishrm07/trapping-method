# Thread 07 — v7 design: robust optimization on the LoRA-r ball

**Status:** committed 2026-05-03 — designing
**Type:** theory + implementation
**Owner:** —

## What v7 is, in one paragraph

v7 replaces the trap loss with **adversarial training in LoRA-bounded
weight space**. At each defender step, an inner PGD finds the rank-r
LoRA factors `(A, B)` that *maximize* a harmful adversary's success at
the perturbed weights `θ + B@A`. The defender then optimizes θ to
*reduce* the worst-case harmful success. This is Madry-style PGD-AT
applied to weight space, restricted to the rank-r manifold, and
targeting harmful-classifier accuracy as the inner objective.

It is **structurally different** from v1–v5 in three ways:

1. The inner loop optimizes **the perturbation Δ**, not the adversary's
   classifier ω_H or LoRA factors as part of an SGD trajectory. We're
   finding the *worst* perturbation, not simulating *one*.
2. The defender's loss has **no `softplus` form, no trap mechanic**.
   It's the harmful classifier's loss at the post-PGD perturbation,
   which the defender wants HIGH.
3. PGD provides **bound-style guarantees within the rank-r ball**, not
   trajectory-level guarantees on a single SGD path.

## The formulation

### Setup

- Defender's parameters: θ (specifically the conv weights of layer3+layer4 in our ResNet18 setting; lower frozen, primary head trainable)
- Adversary's parameters per LoRA application: `A ∈ ℝ^{r × C_in·k²}` and `B ∈ ℝ^{C_out × r}` per conv layer; head `ω_H ∈ ℝ^{n_classes × D}`
- Effective weight under LoRA: `W_eff = W + B@A` (per layer)
- Harmful classifier's loss: `L_harm(θ', ω_H) = CE(features(θ')(z_h) · ω_Hᵀ, y_h)`

### Objective

Defender's training loss at each outer step:

```
L_total(θ) = L_primary(θ)  −  λ · L_harm(θ + Δ*, ω_H*(θ + Δ*))
```

where:
- `Δ* = argmin_{Δ : rank-r, ||Δ|| ≤ ε} L_harm(θ + Δ, ω_H*(θ + Δ))`  ← inner PGD
- `ω_H*(θ') = argmin_{ω_H} L_harm(θ', ω_H) ≈ ridge_solve(features(θ'), y_h)`  ← LP optimum given features (closed form)

In words:

- The adversary's best move is the rank-r weight perturbation Δ that
  minimizes harmful loss when paired with the LP-optimal ω_H given the
  perturbed features.
- The defender's training step pushes θ to *increase* harmful loss at
  the adversary's best move (i.e., minimize the negation in `L_total`).

The minus sign in `L_total` makes this a **gradient descent on `L_primary
− λ · L_harm(adversary's optimum)`**, equivalent to **gradient descent on `L_primary` and gradient ASCENT on `L_harm` evaluated at the adversary's
optimum**. Madry-style adversarial training on weights.

### Why PGD on Δ specifically

We have two equivalent ways to parameterize a rank-r perturbation:

**Direct:** Δ ∈ ℝ^{C_out × C_in·k²} with rank ≤ r constraint. Project to
rank-r via SVD truncation each PGD step. Numerically fragile (we know
SVD fails sometimes).

**Factored:** Δ = B@A with A ∈ ℝ^{r × C_in·k²}, B ∈ ℝ^{C_out × r}. Rank
constraint trivially satisfied. PGD on (A, B) directly.

We commit to **factored**. Norm constraint `||B@A|| ≤ ε` enforced via
post-step rescaling: if ||B@A|| > ε, multiply both A and B by
`sqrt(ε / ||B@A||)`. Standard rsLoRA-style symmetric scaling.

### Why ω_H* via ridge regression (not SGD)

We want the LP optimum at the perturbed features, not a simulation
of an SGD trajectory toward it. With ridge regression,

```
ω_H* = argmin_{ω_H} ||features · ω_Hᵀ − one_hot(y_h)||² + γ ||ω_H||²
     = (features^T · features + γ · I)^{-1} · features^T · one_hot(y_h)
```

is a closed form per defender step. Cost: one D×D matrix inverse per step
(~5ms on A6000 for D=512).

This sidesteps a major issue of v1-v5: simulating ω_H training via SGD
introduces budget mismatch with the real attacker. The ridge optimum is
a tight upper bound on the LP attacker's reachable performance — there
is no "simulate harder" lever the attacker can pull on the linear
classifier specifically.

### Algorithm

```python
for outer_step in range(N):
    x_p, y_p = primary_batch
    x_h, y_h = harmful_batch
    z_p = lower(x_p); z_h = lower(x_h)
    
    # Standard primary loss
    feat_p = upper(z_p)
    L_primary = F.cross_entropy(primary_head(feat_p), y_p)
    
    # Inner PGD: find worst-case rank-r LoRA perturbation
    A_dict, B_dict = init_lora_factors(upper, rank=r)  # B = 0, A = Kaiming
    for inner in range(k_pgd):
        feat_h = forward_with_lora(z_h, upper, A_dict, B_dict)
        omega_H = ridge_solve(feat_h, y_h, gamma=γ)  # closed form
        L_adv = F.cross_entropy(feat_h @ omega_H.T, y_h)
        gA, gB = autograd.grad(L_adv, [*A_dict.values(), *B_dict.values()])
        # Adversary descends: gradient descent step
        for name in A_dict:
            A_dict[name] = A_dict[name] - eta_pgd * gA[name]
            B_dict[name] = B_dict[name] - eta_pgd * gB[name]
            # Projection: rescale to ||B@A|| ≤ ε
            scale = min(1, sqrt(eps / ||B_dict[name] @ A_dict[name]||))
            A_dict[name] *= scale
            B_dict[name] *= scale
    
    # Detach (FOMAML — no second-order through PGD ascent)
    A_dict = {k: v.detach() for k, v in A_dict.items()}
    B_dict = {k: v.detach() for k, v in B_dict.items()}
    
    # Defender's loss at the adversary's worst-case Δ
    feat_h_adv = forward_with_lora(z_h, upper, A_dict, B_dict)
    omega_H_adv = ridge_solve(feat_h_adv.detach(), y_h, gamma=γ)
    L_post_pgd = F.cross_entropy(feat_h_adv @ omega_H_adv.T, y_h)
    
    # Total loss: minimize L_primary, MAXIMIZE L_post_pgd
    L_total = L_primary - lambda_robust * L_post_pgd
    
    optim.zero_grad()
    L_total.backward()
    optim.step()
```

### Hyperparameters

| Symbol | Initial value | Reasoning |
|---|---|---|
| `r` (LoRA rank) | 8 | Matches our prior trap experiments and Lermen's typical rank |
| `ε` (norm budget) | TBD | Need to calibrate; start with 0.01 · ||θ_upper||, sweep |
| `k_pgd` (inner ascent steps) | 5 | Matches PGD-AT image-space (Madry uses 7) |
| `η_pgd` (inner step size) | ε / 4 | Madry standard: step ≈ ε/4 |
| `γ` (ridge regularization) | 1e-3 | Standard for closed-form LP |
| `λ_robust` (defender weight) | 0.3 | Matches v4a's λ_trap; can sweep |
| `lr_outer` | 1e-4 SGD | Matches v4a |

`ε` is the single most important hyperparameter. We don't have a
principled value yet. Plan: sweep `ε ∈ {0.001, 0.01, 0.1, 1.0} · ||θ_upper||`
in the first ablation set.

## What's structurally different from v1–v5 (in plain language)

| | v1–v5 (trap) | v7 (this) |
|---|---|---|
| What inner loop simulates | A specific adversary's k-step SGD trajectory | The adversary's *optimum* within the rank-r ball |
| What inner loop optimizes | LoRA factors + ω_H jointly via SGD | LoRA factors via PGD; ω_H closed-form |
| What outer loop minimizes | Softplus of (realized − predicted) reduction | Negation of harmful loss at adversary's worst Δ |
| Trajectory or bound? | Trajectory simulation (one path) | Ball-bound (worst point in r-ball) |
| Connection to known frameworks | Bilevel meta-learning (MAML-family) | Robust optimization (Madry PGD-AT-family) |
| Generalization across rank? | Empirically: rank-overfit at trained rank | Hypothesis: bounded across the entire r-ball |

## What we predict (the testable claims)

**P1 (the core claim):** v7 will produce **non-trivial generalized
rank-distribution RFD** — i.e., RFD > 2 across ranks 1, 2, 4, 8, 16 at
least, where v4a has RFD ≈ 0.

**P2 (the curve shape):** v7's RFD curve will be roughly *flat* across
ranks within the trained ε ball, then degrade smoothly as rank grows
beyond the trained ε. We predict a Pareto curve: as ε grows, primary
acc drops and rank-distribution RFD rises.

**P3 (vs TAR):** v7 will be more robust than TAR-style FOMAML inner
adversary at the same compute budget, because PGD on Δ converges to
worst-case Δ in the rank-r ball whereas FOMAML simulates a specific
trajectory.

**P4 (the negative bound):** v7 will fail beyond ε. An attacker with
LoRA budget > ε will recover harmful performance. This is not a flaw —
it's the inherent statement of the robustness budget.

If P1 fails (v7's RFD is also ≈ 0), our hypothesis is wrong and we'll
need to pivot to capacity-restriction or pre-training-filter
approaches.

## Two-week implementation plan

### Week 1 — math, prototype, first results

**Day 1 (today):** Lock formulation (this document). No code.

**Day 2:** Implement `ridge_solve` (LP closed-form), `init_lora_factors`,
`forward_with_lora_factored` (parametrize by A, B not Δ), `project_rank_r_ball`.
Self-tests on toy data.

**Day 3:** Implement v7 outer loop in `experiments/run_immunization_v7.py`
(separate file, don't fork run_immunization_cn.py — too much new
machinery). Smoke test: 100 outer steps on tiny data, check it doesn't
NaN.

**Day 4:** First full run on role-lab. ε from a coarse sweep
{0.001, 0.01, 0.1, 1.0} · ||θ_upper||. 4 parallel runs across 4 GPUs.
Pick the ε that gives the cleanest train trajectory (no NaN, primary
acc > 60%).

**Day 5:** Probe v7 on rank-distribution {1, 2, 4, 8, 16, 32, 64} ×
baseline rank-distribution. Compare to v4a's rank curve. **Critical
decision point:** does v7 produce non-trivial generalized RFD? If yes,
continue. If no, document and pivot.

### Week 2 — iteration, write-up, position

**Day 6-7:** Iterate on hyperparameters. λ_robust, k_pgd, ε. Try
multi-rank training (sample r from a distribution rather than fixed 8).

**Day 8:** Run capacity-tradeoff null hypothesis test: does benign LoRA
fine-tune on a different downstream task still work? (Critical for
contribution claims — if benign LoRA breaks, we're not defending,
we're capacity-restricting.)

**Day 9:** Run side-door null hypothesis test: probe v7 with Wei et
al. 2024-style untrained low-rank perturbations and pruning.

**Day 10:** Write `STAGE7_v7_REPORT.md` with full results.

**Day 11-12:** Sketch paper outline. Compare to TAR, IMMA, CTRAP. The
narrative is "robust optimization in LoRA-r weight space gives
generalized rank-distribution defense; trap-mechanism doesn't."

**Day 13-14:** Email Chris Thomas with v7 results + paper sketch.
Schedule a meeting if he's responsive.

### Risks / contingencies

| Risk | Likelihood | Mitigation |
|---|---|---|
| Inner PGD NaNs | medium | We've handled SVD fragility before; reuse k_inv-style fallback |
| Ridge solve singular | low | Diag-scaled ridge γ; fp32 + try/except |
| Out-of-memory at k_pgd=5 + r=8 | low | A6000 has 48GB; v4a used <30GB; we have budget |
| Min-max collapse (Δ flattens features to 0) | medium | Bound ε so Δ can't fully cancel θ_upper; add a feature-norm regularizer |
| v7 RFD also ≈ 0 (hypothesis wrong) | maybe 40% | Documented negative result is itself a contribution; pivot to capacity restriction |

## Acceptance criterion

Thread closes when:
- v7 produces RFD > 2 generalized across ranks 1-16 (P1 confirmed), **OR**
- v7 produces RFD ≈ 0 across ranks (P1 disconfirmed); pivot or document negative result

Either outcome is publishable.

## Open questions to revisit before Day 4

1. **What's the right scope for ε**? Per-layer? Per-tensor? Global? My
   default is per-conv-layer (each layer has its own rank-r ball with
   the same ε). Alternative: global rank-r ball over the concatenated
   weight tensor.

2. **Can the inner PGD be replaced with iMAML-style implicit
   gradients**? (Rajeswaran 2019.) Would give a true "adversary's
   optimum" without simulation budget mismatch. Possibly stronger but
   more complex. Defer to Week 2 if v7 with PGD shows promise.

3. **Is the ω_H closed-form ridge actually a tight upper bound?** Or
   should we also fine-tune ω_H jointly with (A, B) via PGD? Standard
   PGD-AT in image space doesn't have this question because there's no
   "classifier" inside the inner loop. Our setting does. Need to check.

4. **Should we add a primary-task robustness term too?** Currently we
   only require *worst-case* harmful loss to be high. We could also
   require *worst-case* primary acc to be high (full Pareto). Defer.
