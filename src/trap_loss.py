"""Trap-inducing loss from Sarker et al. NeurIPS 2025 Lock-LLM Workshop §3.1.

The trap loss penalizes any *realized* multi-step harmful loss reduction
that exceeds what the local quadratic geometry predicts. Over iterations,
this carves the harmful-task landscape into deceptive plateaus where
gradient descent makes apparent progress but never actually descends much.

Eq. 4 (predicted reduction, second-order Taylor expansion):
    ΔL_exp = −(g_0ᵀ Δθ + ½ Δθᵀ H_0 Δθ)

Eq. 5 (realized reduction after k inner adversary steps):
    ΔL_act = L_H(θ⁰) − L_H(θᵏ)

Eq. 6 (trap loss):
    L_trap(θ⁰) = softplus(ΔL_act − ΔL_exp)

In the linear-probing setting we adopt (matching Zheng et al. + the trapping
paper's experimental setup), the adversary's free parameter is the harmful
classifier head ω_H ∈ ℝ^{n_classes × D_hid}. The inner k-step loop is plain
SGD on CE loss with respect to ω_H, with `create_graph=True` so gradients
flow back to the defender's θ_upper.

Hessian H_0 of L_H w.r.t. ω_H is approximated by the feature covariance
K = X̃ᵀX̃/B (Zheng §3.1, exact for ℓ₂ loss; same-spirit approximation for CE,
matching the trapping paper's Appendix B).

KNN-cluster init for ω_H_0 follows the paper's Appendix B: use per-class
mean features as the initial classifier rows so gradients on the inner loop
are meaningful from step 0 (random init gives near-flat gradient signal).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def knn_centroid_init(features: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Per-class mean of `features`, returned as a [num_classes, D_hid] tensor.

    Classes that have no examples in the batch fall back to the global mean.
    """
    if features.dim() != 2:
        raise ValueError(f"features must be [B, D_hid], got {tuple(features.shape)}")
    D = features.shape[1]
    centroids = torch.zeros(num_classes, D, device=features.device, dtype=features.dtype)
    global_mean = features.mean(dim=0)
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            centroids[c] = features[mask].mean(dim=0)
        else:
            centroids[c] = global_mean
    return centroids


def trap_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    k_inner: int = 3,
    eta_inner: float = 0.01,
) -> torch.Tensor:
    """Compute Eq. 6 of the trapping paper for one defender training step.

    Args:
        features: [B, D_hid] — output of the immunization-trainable extractor
            on a harmful batch. Must have requires_grad=True (we differentiate
            through this).
        labels: [B] long-tensor of harmful-task labels (0..num_classes-1).
        num_classes: dimensionality of the adversary's classifier head.
        k_inner: number of adversary SGD steps simulated per defender step.
            Larger k → tighter trap, but k× memory through the inner graph.
        eta_inner: SGD learning rate for the inner adversary loop. Should
            roughly match the realistic adversary's per-step magnitude.

    Returns:
        Scalar tensor (differentiable) with `softplus(ΔL_act − ΔL_exp)`.
        Smaller → trap is biting (realized reduction matches or undershoots
        local quadratic prediction). Larger → adversary is making faster
        progress than local geometry suggests, and the defender should pull
        θ_upper to suppress this surplus.
    """
    if features.dim() != 2:
        raise ValueError(f"features must be [B, D_hid], got {tuple(features.shape)}")

    B, D = features.shape

    # 1. KNN centroid init (no-grad — treat ω_H^0 as a fixed function of features)
    omega_h_0 = knn_centroid_init(features.detach(), labels, num_classes).clone()
    omega_h_0.requires_grad_(True)

    # 2. Initial loss + gradient at ω_H^0 (this is g_0 in the Taylor expansion)
    logits_0 = features @ omega_h_0.T            # [B, num_classes]
    L_H_0 = F.cross_entropy(logits_0, labels)
    g_0 = torch.autograd.grad(L_H_0, omega_h_0, create_graph=True)[0]   # [num_classes, D]

    # 3. k-step inner SGD unroll — simulates an adversary doing linear probing.
    omega_h_t = omega_h_0
    for _ in range(k_inner):
        logits_t = features @ omega_h_t.T
        L_H_t = F.cross_entropy(logits_t, labels)
        grad_t = torch.autograd.grad(L_H_t, omega_h_t, create_graph=True)[0]
        omega_h_t = omega_h_t - eta_inner * grad_t

    # 4. Realized loss at ω_H^k
    logits_k = features @ omega_h_t.T
    L_H_k = F.cross_entropy(logits_k, labels)

    # 5. ΔL_act = L(ω_0) − L(ω_k)
    delta_L_act = L_H_0 - L_H_k

    # 6. ΔL_exp via local quadratic. Hessian ≈ feature covariance (Zheng §3.1).
    delta_omega = omega_h_t - omega_h_0           # [num_classes, D]
    K = features.T @ features / B                 # [D, D] feature-covariance approx
    # Linear term: g_0ᵀ Δω  → (num_classes, D) ⊙ (num_classes, D) → scalar
    linear_term = (g_0 * delta_omega).sum()
    # Quadratic term: ½ Δω H Δωᵀ. With per-class Kronecker-decomposed H ≈ K⊗I,
    # this becomes ½ Σ_c Δω_c K Δω_cᵀ.
    quadratic_term = torch.einsum('cd,de,ce->', delta_omega, K, delta_omega)
    delta_L_exp = -(linear_term + 0.5 * quadratic_term)

    # 7. Trap loss = softplus(ΔL_act − ΔL_exp)
    return F.softplus(delta_L_act - delta_L_exp)


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _self_test() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, D, C = 16, 32, 5
    # Simple case: linear features through a learnable θ
    theta = torch.randn(D, D, device=device, requires_grad=True)
    raw_features = torch.randn(B, D, device=device)
    labels = torch.randint(0, C, (B,), device=device)

    features = raw_features @ theta
    loss = trap_loss(features, labels, num_classes=C, k_inner=3, eta_inner=0.01)
    assert loss.requires_grad
    assert loss.numel() == 1
    assert loss.item() >= 0  # softplus is non-negative

    # Should have gradient w.r.t. theta (the immunization-trainable param)
    grad = torch.autograd.grad(loss, theta)[0]
    assert grad.shape == theta.shape
    assert torch.isfinite(grad).all(), "Gradient must be finite"

    print(f"trap_loss.py self-test passed (loss={loss.item():.6f}, ||grad||={grad.norm().item():.6f})")


if __name__ == "__main__":
    _self_test()
