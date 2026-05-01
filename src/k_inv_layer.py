"""K⁻¹-preconditioned dummy layer (Zheng et al. ICML 2025 §4.4).

Algorithm 1 line 6 multiplies the regularizer gradient by K⁻¹:

    θ ← θ − η · λ_P · K_P⁻¹ · ∇_θ R_well(H_P(θ))

Mathematical purpose: cancels the K factor that appears in ∇_θ R_well
(Theorem 4.2's gradient is `2·K·θ·(...)`), leaving an update operator that
takes simple, K-independent steps in θ-space. Without this preconditioner,
the regularizer gradient magnitude scales with ||K|| (≈ ||features||²) — so
the same `lr` becomes much more aggressive for the regularizer than for the
primary CE, blowing past the paper's monotonic-κ guarantees.

The dummy layer trick: insert an autograd Function with identity forward and
K⁻¹-multiplied backward into the regularizer computation graph. A single
backward pass then yields the preconditioned θ gradient — no custom
optimizer needed.

For non-linear extractors, K is the *current-batch* feature covariance
(matching what the regularizer also sees), with a small ridge term for
numerical stability.
"""
from __future__ import annotations

import torch


class _KInvBackward(torch.autograd.Function):
    """Identity forward; backward multiplies gradient by `K_inv`."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, K_inv: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(K_inv)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (K_inv,) = ctx.saved_tensors
        # grad_output shape [..., D_hid]; multiply along last axis.
        return grad_output @ K_inv, None


def k_inv_dummy_layer(features: torch.Tensor, *, ridge: float = 1e-3) -> torch.Tensor:
    """Wrap `features` so backward through the regularizer applies K⁻¹.

    Computes `K = features^T features / B + ridge·I` on the *current batch*,
    inverts it (with a ridge term for stability), and pre-applies the dummy
    layer. Returns features unchanged in the forward pass; gradients flowing
    back through this point will be multiplied by K⁻¹ exactly once.

    Use only on the path that leads to a regularizer (R_well or R_ill) — not
    on the primary CE path.

    Args:
        features: [B, D_hid] feature matrix from the immunization-trainable
            extractor. Must require_grad.
        ridge: small scalar added to the diagonal of K before inversion to
            avoid blow-up on near-singular feature covariances.

    Returns:
        A tensor identical in value and shape to `features`, with
        K⁻¹-preconditioned backward.
    """
    if features.dim() != 2:
        raise ValueError(f"features must be [B, D_hid], got {tuple(features.shape)}")
    B, D = features.shape
    # K is detached (we don't backpropagate through the inversion itself —
    # paper's dummy layer treats K as constant for the backward).
    with torch.no_grad():
        K = features.T @ features / B
        K = K + ridge * torch.eye(D, device=K.device, dtype=K.dtype)
        K_inv = torch.linalg.inv(K)
    return _KInvBackward.apply(features, K_inv)


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _self_test() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Simple sanity: forward is identity.
    feat = torch.randn(16, 32, device=device, requires_grad=True)
    feat_pre = k_inv_dummy_layer(feat, ridge=1e-3)
    assert torch.allclose(feat_pre, feat), "Forward pass must be identity"

    # Backward: compare grad with and without preconditioner.
    # Loss = sum(features^2). ∂L/∂features = 2·features.
    # With dummy: ∂L/∂features = 2·features @ K_inv.
    loss_pre = (feat_pre ** 2).sum()
    grad_pre = torch.autograd.grad(loss_pre, feat, retain_graph=True)[0]

    # Compute reference K_inv to verify
    with torch.no_grad():
        K_ref = feat.T @ feat / feat.shape[0] + 1e-3 * torch.eye(32, device=device)
        K_inv_ref = torch.linalg.inv(K_ref)
        expected = (2 * feat) @ K_inv_ref

    assert torch.allclose(grad_pre, expected, rtol=1e-4, atol=1e-5), \
        "Backward should match (2·features) @ K_inv"

    print("k_inv_layer.py self-test passed")


if __name__ == "__main__":
    _self_test()
