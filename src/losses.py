"""Condition-number regularizers from Zheng et al. (ICML 2025).

Two regularizers, both differentiable through `torch.linalg.svdvals`:

- `r_well(S)` — Eq. 3 of Zheng (originally Nenov et al. 2024). Minimizes κ(S)
  by penalizing spectral-vs-Frobenius norm imbalance. Used on the primary
  task's Hessian H_P to keep adaptation easy.

- `r_ill(S)` — Eq. 12 of Zheng. Maximizes κ(S) by inflating ||S||_F^2 relative
  to (σ_min)^2. Used on the harmful task's Hessian H_H to make adaptation
  hard.

We use PyTorch autograd through SVD rather than the closed-form gradients
(Theorem 4.2). Faithful to the paper's optimization objective; trades a bit of
speed for far simpler code. SVD on a 512×512 matrix is ~ms on L40S.

NOTE on the K^{-1} preconditioner: Algorithm 1 line 6 multiplies the
regularizer gradients by K^{-1} to preserve the monotonicity guarantee
(Theorem 4.3). The current Stage-2 implementation **omits the preconditioner**
and lets standard autograd compute the update. This is a known simplification:
- Pro: no custom autograd Function needed; trivial to integrate with optimizer.
- Con: the closed-form monotonic increase/decrease in κ is no longer guaranteed
  step-by-step. Empirically we expect immunization to still occur, with
  possibly larger λ_well, λ_ill needed and slightly worse RIR.

If RIR comes in much below ~3.5 on Cars/ResNet18, revisit this and add a
"dummy layer" with K^{-1}-multiplied backward (paper §4.4).
"""
from __future__ import annotations

import torch


def _trace_normalize(S: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Divide a square PSD-ish matrix by its trace.

    The Stage-2 smoke test on real ResNet18 features showed ||H||_F ≈ 10⁴,
    making R_well ≈ 22 600 — three orders of magnitude above the cross-entropy
    term, which would force λ_well near 1e-5 to avoid the regularizer
    steamrolling primary-task accuracy.

    Both R_well and R_ill are homogeneous in S: r_well(αS) = α² · r_well(S)
    and r_ill(αS) = α⁻² · r_ill(S). Dividing by tr(S) (= Σ σ_i for PSD)
    gives a scale-invariant input where both regularizers live in O(1) and
    the default λ_well = λ_ill = 1.0 in the config behaves sanely.

    Note this rescales the gradient flowing into θ by 1/tr(S), so the
    regularizer effectively self-tunes its learning rate to feature magnitude.
    """
    trace = torch.diagonal(S).sum()
    return S / (trace + eps)


def r_well(S: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """κ-minimizing regularizer (Eq. 3 of Zheng et al.), trace-normalized.

        R_well(S̃) = ½ ||S̃||₂² − (1/(2p)) ||S̃||_F²,   where S̃ = S / tr(S)

    Nonnegative; zero iff κ(S) = 1.

    Args:
        S: a 2-D PSD-ish matrix, typically a Hessian approximation [D_hid, D_hid].
        eps: trace-normalization stability term.

    Returns:
        A scalar tensor (differentiable). Smaller → better-conditioned S.
    """
    if S.dim() != 2:
        raise ValueError(f"r_well expects a 2-D matrix, got shape {tuple(S.shape)}")
    S = _trace_normalize(S, eps=eps)
    sigmas = torch.linalg.svdvals(S)
    spectral_sq = sigmas[0].pow(2)         # ||S||₂²
    frob_sq = (S ** 2).sum()               # ||S||_F² = Σ σ_i²
    p = min(S.shape[0], S.shape[1])
    return 0.5 * spectral_sq - frob_sq / (2.0 * p)


def r_ill(S: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """κ-maximizing regularizer (Eq. 12 of Zheng et al.), *not* trace-normalized.

        R_ill(S) = 1 / [(1/(2k)) ||S||_F² − ½ (σ_S^min)²]

    Reading: minimizing R_ill (the value shrinks) requires the denominator
    to grow — pushing σ_min toward 0 and ||S||_F² up, increasing κ.

    NOTE: unlike `r_well`, this function does *not* trace-normalize S. We
    tried trace-normalization in commit efbf99e and the Stage-2 run on
    Cars/ResNet18 showed r_ill stuck near its lower bound (~1000 for
    k=512), with no descent over 500 steps. The cause: with S/tr(S), both
    ||S̃||_F² ∈ [1/k, 1] and σ_min²/2 are bounded by 1/k², which makes the
    denominator collapse into a tiny `1/(2k²)`-scaled range. Real
    pretrained features start near rank-1 (eigenvalues concentrated), which
    is already at the lower bound of trace-normalized r_ill. Skipping
    normalization restores the meaningful gradient signal: as θ updates
    push features further apart, ||S||_F grows, denom grows, r_ill shrinks.
    Tune λ_ill in the config if magnitudes get unbalanced with λ_well.
    """
    if S.dim() != 2:
        raise ValueError(f"r_ill expects a 2-D matrix, got shape {tuple(S.shape)}")
    sigmas = torch.linalg.svdvals(S)
    sigma_min_sq = sigmas[-1].pow(2)
    frob_sq = (S ** 2).sum()
    k = float(sigmas.numel())
    denom = frob_sq / (2.0 * k) - 0.5 * sigma_min_sq
    return 1.0 / (denom + eps)


# -----------------------------------------------------------------------------
# Self-tests — run with: python -m src.losses
# -----------------------------------------------------------------------------

def _self_test() -> None:
    torch.manual_seed(0)

    # 1. r_well on identity → should be zero (κ(I) = 1, perfectly conditioned).
    I = torch.eye(8)
    val = r_well(I).item()
    assert abs(val) < 1e-6, f"r_well(I) should be 0, got {val}"

    # 2. r_well on a clearly ill-conditioned matrix → should be positive.
    S = torch.diag(torch.tensor([10.0, 1.0, 0.1]))
    assert r_well(S).item() > 0, "r_well of ill-conditioned matrix must be positive"

    # 3. Autograd through r_well: gradient should drive S toward better conditioning.
    S = torch.diag(torch.tensor([5.0, 0.5, 0.1])).clone().requires_grad_(True)
    kappa_before = (torch.linalg.svdvals(S)[0] / torch.linalg.svdvals(S)[-1]).item()
    optim = torch.optim.SGD([S], lr=0.05)
    for _ in range(200):
        optim.zero_grad()
        r_well(S).backward()
        optim.step()
    kappa_after = (torch.linalg.svdvals(S)[0] / torch.linalg.svdvals(S)[-1]).item()
    assert kappa_after < kappa_before, (
        f"r_well descent should reduce κ, but got κ_before={kappa_before:.3f}, κ_after={kappa_after:.3f}"
    )

    # 4. r_ill: descent should INCREASE κ.
    S = torch.diag(torch.tensor([2.0, 1.5, 1.0])).clone().requires_grad_(True)
    kappa_before = (torch.linalg.svdvals(S)[0] / torch.linalg.svdvals(S)[-1]).item()
    optim = torch.optim.SGD([S], lr=0.05)
    for _ in range(200):
        optim.zero_grad()
        r_ill(S).backward()
        optim.step()
    kappa_after = (torch.linalg.svdvals(S)[0] / torch.linalg.svdvals(S)[-1]).item()
    assert kappa_after > kappa_before, (
        f"r_ill descent should increase κ, but got κ_before={kappa_before:.3f}, κ_after={kappa_after:.3f}"
    )

    # 5. Shape check.
    try:
        r_well(torch.zeros(3))
    except ValueError:
        pass
    else:
        raise AssertionError("r_well should reject 1-D input")

    print("losses.py self-test passed")
    print(f"  r_well: κ from {kappa_before:.2f} → {kappa_after:.2f}  (test 4 — r_ill expectation)")


if __name__ == "__main__":
    _self_test()
