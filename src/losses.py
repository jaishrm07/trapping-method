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


def r_well(S: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """κ-minimizing regularizer (Eq. 3 of Zheng et al.).

        R_well(S) = ½ ||S||₂² − (1/(2p)) ||S||_F²

    where ||S||₂ is the spectral norm (largest singular value) and p =
    min(p_r, p_c) is the smaller dimension. Nonnegative; zero iff κ(S) = 1.

    Args:
        S: a 2-D matrix tensor, typically a Hessian approximation [D_hid, D_hid].
        eps: numerical guard (not used in the formula but accepted for symmetry
            with r_ill).

    Returns:
        A scalar tensor (differentiable). Smaller → better-conditioned S.
    """
    if S.dim() != 2:
        raise ValueError(f"r_well expects a 2-D matrix, got shape {tuple(S.shape)}")
    sigmas = torch.linalg.svdvals(S)
    spectral_sq = sigmas[0].pow(2)         # ||S||₂²
    frob_sq = (S ** 2).sum()               # ||S||_F² = Σ σ_i²
    p = min(S.shape[0], S.shape[1])
    return 0.5 * spectral_sq - frob_sq / (2.0 * p)


def r_ill(S: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """κ-maximizing regularizer (Eq. 12 of Zheng et al.).

        R_ill(S) = 1 / [(1/(2k)) ||S||_F² − ½ (σ_S^min)²]

    Reading: minimizing R_ill (so the *value* shrinks) requires the denominator
    to *grow* — pushing σ_min toward 0 and ||S||_F² up, increasing κ.

    Args:
        S: a 2-D matrix tensor, typically a Hessian approximation [D_hid, D_hid].
        eps: small additive constant inside the reciprocal to avoid blow-up
            when the denominator approaches 0 (early in training).

    Returns:
        A scalar tensor (differentiable). Smaller → ill-conditioned S → harder
        for adversary to fit.
    """
    if S.dim() != 2:
        raise ValueError(f"r_ill expects a 2-D matrix, got shape {tuple(S.shape)}")
    sigmas = torch.linalg.svdvals(S)
    sigma_min_sq = sigmas[-1].pow(2)
    frob_sq = (S ** 2).sum()
    k = float(sigmas.numel())  # rank upper-bounded by # of singular values
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
