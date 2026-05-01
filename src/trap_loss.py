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
classifier head ω_H ∈ ℝ^{n_classes × D_hid}.

Two trap variants in this module:

- `trap_loss(features, labels, ...)` — paper-faithful, linear-probing
  inner loop. Uses full Taylor expansion (Eq. 4). The adversary's free
  parameter is ω_H only.

- `trap_loss_lora(upper, z_h, labels, ...)` — LoRA-aware, simulates a k-step
  adversary doing LoRA-rank-r fine-tuning of `upper` plus head training.
  Uses simplified `softplus(ΔL_act)` because the Hessian over the
  expanded {LoRA_A, LoRA_B, ω_H} parameter space is intractable. Strictest
  trap (no allowance for predicted reduction) but tractable.

- `trap_loss_multiop(...)` — randomly samples one operator per defender
  step (Plan C). The trap is shaped against a *mixture* of adversaries.
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn
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
# LoRA-aware trap (operator-transfer fix)
# -----------------------------------------------------------------------------

def trap_loss_lora(
    upper: nn.Module,
    z_h: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    k_inner: int = 3,
    eta_inner: float = 0.01,
    lora_rank: int = 8,
) -> torch.Tensor:
    """Trap loss simulating a k-step LoRA-rank-r adversary.

    The adversary's free parameters expand from `ω_H` (head only) to the
    triple `(LoRA_A, LoRA_B, ω_H)`. LoRA factors are added to every conv
    in `upper` via the standard B@A weight-delta decomposition.

    Forward pass inside the inner loop uses `torch.func.functional_call`
    with substituted weights `W' = W + B@A` so PyTorch's autograd handles
    gradient flow back to θ_upper through the residual + LoRA paths.

    Returns `softplus(ΔL_act)` — the strict-trap variant. Penalizes any
    realized harmful loss reduction without subtracting a quadratic
    prediction. Stronger than the paper's `softplus(ΔL_act − ΔL_exp)`,
    but tractable in the expanded parameter space (computing H_0 over
    {LoRA_A, LoRA_B, ω_H} is infeasible).
    """
    if z_h.dim() != 4:
        raise ValueError(f"z_h must be [B, C, H, W], got {tuple(z_h.shape)}")
    device = z_h.device
    dtype = z_h.dtype

    conv_specs = [(name, mod) for name, mod in upper.named_modules() if isinstance(mod, nn.Conv2d)]

    # LoRA factors fresh per defender step — A kaiming, B zero (standard init).
    lora_A, lora_B = {}, {}
    for name, conv in conv_specs:
        Cout, Cin = conv.out_channels, conv.in_channels
        K1, K2 = conv.kernel_size
        A = torch.empty(lora_rank, Cin, K1, K2, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(A, a=5 ** 0.5)
        B = torch.zeros(Cout, lora_rank, 1, 1, device=device, dtype=dtype)
        lora_A[name] = A.requires_grad_(True)
        lora_B[name] = B.requires_grad_(True)

    with torch.no_grad():
        feat_init = upper(z_h)
        omega_h_init = knn_centroid_init(feat_init, labels, num_classes)
    omega_h_0 = omega_h_init.detach().clone().requires_grad_(True)

    def forward_with_lora(lora_A_, lora_B_, omega_h_):
        param_dict = dict(upper.named_parameters())
        for cname, _conv in conv_specs:
            base_w = param_dict[cname + ".weight"]
            A_ = lora_A_[cname]
            B_ = lora_B_[cname]
            delta_w = (B_.flatten(1) @ A_.flatten(1)).view_as(base_w)
            param_dict[cname + ".weight"] = base_w + delta_w
        feat = torch.func.functional_call(upper, param_dict, z_h)
        logits = feat @ omega_h_.T
        return F.cross_entropy(logits, labels)

    L_H_0 = forward_with_lora(lora_A, lora_B, omega_h_0)

    cur_A, cur_B, cur_omega = dict(lora_A), dict(lora_B), omega_h_0
    n_A = len(cur_A)
    for _ in range(k_inner):
        L_H_t = forward_with_lora(cur_A, cur_B, cur_omega)
        param_list = list(cur_A.values()) + list(cur_B.values()) + [cur_omega]
        grads = torch.autograd.grad(L_H_t, param_list, create_graph=True)
        new_A, new_B = {}, {}
        for i, (name, _) in enumerate(conv_specs):
            new_A[name] = cur_A[name] - eta_inner * grads[i]
            new_B[name] = cur_B[name] - eta_inner * grads[n_A + i]
        cur_A, cur_B = new_A, new_B
        cur_omega = cur_omega - eta_inner * grads[-1]

    L_H_k = forward_with_lora(cur_A, cur_B, cur_omega)
    delta_L_act = L_H_0 - L_H_k
    return F.softplus(delta_L_act)


# -----------------------------------------------------------------------------
# Multi-operator trap (Plan C — operator randomization)
# -----------------------------------------------------------------------------

def trap_loss_multiop(
    upper: nn.Module,
    feat_h: torch.Tensor,
    z_h: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    operators: list,
    k_inner: int = 3,
    eta_inner: float = 0.01,
    lora_rank_for: dict = None,
) -> torch.Tensor:
    """Plan C: per-step random operator selection.

    Samples one operator from `operators` uniformly, runs the corresponding
    inner-loop simulation, returns its trap loss. Over many defender steps
    the trap is shaped against the operator mixture.

    Currently supports:
        "linear_probe" → trap_loss(feat_h, labels, ...)
        "lora_r8"      → trap_loss_lora(upper, z_h, labels, lora_rank=8, ...)
        "lora_r32"     → trap_loss_lora(upper, z_h, labels, lora_rank=32, ...)
    """
    if not operators:
        raise ValueError("operators list must be non-empty")
    if lora_rank_for is None:
        lora_rank_for = {"lora_r8": 8, "lora_r32": 32}

    op = random.choice(operators)

    if op == "linear_probe":
        return trap_loss(feat_h, labels, num_classes=num_classes,
                         k_inner=k_inner, eta_inner=eta_inner)
    elif op in lora_rank_for:
        return trap_loss_lora(upper, z_h, labels, num_classes=num_classes,
                              k_inner=k_inner, eta_inner=eta_inner,
                              lora_rank=lora_rank_for[op])
    else:
        raise ValueError(f"Unknown operator: {op}. Supported: linear_probe, "
                         f"and any key in lora_rank_for ({list(lora_rank_for)})")


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _self_test() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Test 1: original trap_loss with linear features --------------------
    B, D, C = 16, 32, 5
    theta = torch.randn(D, D, device=device, requires_grad=True)
    raw_features = torch.randn(B, D, device=device)
    labels = torch.randint(0, C, (B,), device=device)

    features = raw_features @ theta
    loss = trap_loss(features, labels, num_classes=C, k_inner=3, eta_inner=0.01)
    assert loss.requires_grad and loss.numel() == 1 and loss.item() >= 0
    grad = torch.autograd.grad(loss, theta)[0]
    assert grad.shape == theta.shape and torch.isfinite(grad).all()
    print(f"trap_loss (LP)   passed (loss={loss.item():.6f}, ||grad||={grad.norm().item():.4f})")

    # ---- Test 2: trap_loss_lora with a tiny conv-stack upper ----------------
    class TinyUpper(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(8, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            return F.adaptive_avg_pool2d(x, 1).flatten(1)

    torch.manual_seed(1)
    upper = TinyUpper().to(device)
    z_h = torch.randn(8, 8, 4, 4, device=device)        # frozen-lower output
    labels2 = torch.randint(0, C, (8,), device=device)

    loss_lora = trap_loss_lora(upper, z_h, labels2, num_classes=C,
                               k_inner=2, eta_inner=0.01, lora_rank=4)
    assert loss_lora.requires_grad and loss_lora.numel() == 1 and loss_lora.item() >= 0
    grad_lora = torch.autograd.grad(loss_lora, list(upper.parameters()), allow_unused=False)
    grad_norms = [g.norm().item() if g is not None else 0.0 for g in grad_lora]
    assert all(torch.isfinite(g).all() for g in grad_lora if g is not None)
    print(f"trap_loss_lora   passed (loss={loss_lora.item():.6f}, max||grad||={max(grad_norms):.4f})")

    # ---- Test 3: trap_loss_multiop dispatches without crash -----------------
    feat_h = upper(z_h)
    torch.manual_seed(2)
    losses = []
    for _ in range(5):  # multiple samples to exercise both branches
        l = trap_loss_multiop(upper, feat_h, z_h, labels2,
                              num_classes=C, operators=["linear_probe", "lora_r8"],
                              k_inner=2, eta_inner=0.01,
                              lora_rank_for={"lora_r8": 4})
        assert l.requires_grad and l.numel() == 1 and l.item() >= 0
        losses.append(l.item())
    print(f"trap_loss_multiop passed (5 samples, range=[{min(losses):.4f}, {max(losses):.4f}])")


if __name__ == "__main__":
    _self_test()
