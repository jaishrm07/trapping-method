"""v7 building blocks — robust optimization in LoRA-r weight ball.

Replaces the trap mechanism with adversarial training: at each defender
step, an inner PGD finds the rank-r LoRA factors (A, B) maximizing
harmful classifier success at the perturbed weights θ + B@A. Defender
optimizes θ to reduce the worst-case harmful success.

See `chris-thomas/research/threads/07_v7_design.md` for the full design.

Four building blocks:
    ridge_solve(features, labels, num_classes, gamma) → ω
        Closed-form linear classifier optimum (ridge regression on
        one-hot labels). Tight upper bound on LP attainable
        performance.

    init_lora_factors(upper, rank, device, dtype) → (A_dict, B_dict)
        Fresh LoRA factors (B=0, A Kaiming) per conv layer, with
        requires_grad=True for PGD ascent.

    forward_with_lora_factored(z, upper, A_dict, B_dict) → features
        Forward pass through `upper` with W ← W + B@A substituted
        per conv layer via torch.func.functional_call.

    project_rank_r_ball(A_dict, B_dict, eps) → (in-place rescale)
        Per-layer symmetric rescaling so ||B@A||_F ≤ eps. Inside
        torch.no_grad(); leaf tensors stay leaves.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# 1. Closed-form linear classifier (ridge regression on one-hot)
# -----------------------------------------------------------------------------

def ridge_solve(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    gamma: float = 1e-3,
) -> torch.Tensor:
    """Closed-form optimum for ω* = argmin_ω ||X ω^T − Y||_F^2 + γ||ω||_F^2.

    `features` shape [B, D], `labels` shape [B], target one-hot Y is
    [B, num_classes]. Returns ω of shape [num_classes, D].

    The ridge ω* is a tight upper bound on what a linear classifier can
    achieve given these features (no SGD trajectory simulation; no
    "simulate harder" lever for the adversary on the linear head).

    Numerical stability: scales the ridge by mean diagonal of X^T X so
    γ stays effective when features have large magnitude. Falls back to
    torch.linalg.lstsq if the explicit inverse fails.
    """
    if features.dim() != 2:
        raise ValueError(f"features must be [B, D], got {tuple(features.shape)}")
    B, D = features.shape
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    Y = F.one_hot(labels, num_classes).to(features.dtype)  # [B, num_classes]
    XtX = features.T @ features  # [D, D]
    diag_mean = XtX.diagonal().mean().clamp_min(1.0).detach()
    XtX_reg = XtX + (gamma * diag_mean) * torch.eye(D, device=features.device, dtype=features.dtype)
    XtY = features.T @ Y  # [D, num_classes]

    try:
        omega_T = torch.linalg.solve(XtX_reg, XtY)  # [D, num_classes]
    except torch._C._LinAlgError:
        # Fallback: lstsq with bigger ridge in fp64
        XtX_big = XtX_reg.to(torch.float64) + (99.0 * gamma * diag_mean) * torch.eye(D, device=features.device, dtype=torch.float64)
        omega_T = torch.linalg.solve(XtX_big, XtY.to(torch.float64)).to(features.dtype)

    return omega_T.T  # [num_classes, D]


# -----------------------------------------------------------------------------
# 2. LoRA factor initialization
# -----------------------------------------------------------------------------

def init_lora_factors(
    upper: nn.Module,
    rank: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[dict, dict]:
    """Initialize per-conv-layer rank-r LoRA factors.

    A: Kaiming-uniform-init, shape [r, C_in, K, K] per conv.
    B: zero-init, shape [C_out, r, 1, 1] per conv.

    Both have requires_grad=True (leaf tensors for PGD).

    Returns:
        A_dict: {conv_name: A}
        B_dict: {conv_name: B}
    """
    A_dict, B_dict = {}, {}
    for name, mod in upper.named_modules():
        if isinstance(mod, nn.Conv2d):
            Cout, Cin = mod.out_channels, mod.in_channels
            K1, K2 = mod.kernel_size
            A = torch.empty(rank, Cin, K1, K2, device=device, dtype=dtype)
            nn.init.kaiming_uniform_(A, a=5 ** 0.5)
            B = torch.zeros(Cout, rank, 1, 1, device=device, dtype=dtype)
            A_dict[name] = A.requires_grad_(True)
            B_dict[name] = B.requires_grad_(True)
    if not A_dict:
        raise ValueError("upper has no Conv2d submodules; v7 expects a conv-stack")
    return A_dict, B_dict


# -----------------------------------------------------------------------------
# 3. Forward pass with LoRA factors substituted
# -----------------------------------------------------------------------------

def forward_with_lora_factored(
    z: torch.Tensor,
    upper: nn.Module,
    A_dict: dict,
    B_dict: dict,
) -> torch.Tensor:
    """Forward pass through `upper` with W ← W + B@A substituted per conv.

    Uses torch.func.functional_call to swap weights at call time without
    mutating the module. Both `upper`'s grad path and the LoRA factors'
    grad path stay live for autograd.
    """
    param_dict = dict(upper.named_parameters())
    for name in A_dict:
        base_w = param_dict[name + ".weight"]
        A = A_dict[name]
        B = B_dict[name]
        # B: [C_out, r, 1, 1] → flatten to [C_out, r]
        # A: [r, C_in, K, K] → flatten to [r, C_in*K*K]
        # B@A: [C_out, C_in*K*K] reshape to base_w shape [C_out, C_in, K, K]
        delta_w = (B.flatten(1) @ A.flatten(1)).view_as(base_w)
        param_dict[name + ".weight"] = base_w + delta_w
    return torch.func.functional_call(upper, param_dict, z)


# -----------------------------------------------------------------------------
# 4. Rank-r ball projection (in-place rescale of leaf tensors)
# -----------------------------------------------------------------------------

def project_rank_r_ball(
    A_dict: dict,
    B_dict: dict,
    eps: float,
) -> None:
    """Symmetric rescale of (A, B) so per-layer ||B@A||_F ≤ eps.

    In-place on leaf tensors (requires `with torch.no_grad():` context
    or in_place semantics on leaves; we wrap internally). After the
    rescale, leaves still have requires_grad=True.

    Symmetry: both A and B are multiplied by sqrt(eps/||B@A||) so the
    factorization remains balanced (rsLoRA-style).
    """
    with torch.no_grad():
        for name in A_dict:
            A = A_dict[name]
            B = B_dict[name]
            BA = B.flatten(1) @ A.flatten(1)
            norm = BA.norm()
            if norm > eps:
                scale = (eps / norm.clamp_min(1e-12)).sqrt()
                A.mul_(scale)
                B.mul_(scale)


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _self_test() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Test 1: ridge_solve recovers near-optimal classifier on linear data ----
    B, D, C = 256, 32, 5
    features = torch.randn(B, D, device=device)
    true_omega = torch.randn(C, D, device=device)
    logits = features @ true_omega.T
    labels = logits.argmax(dim=-1)
    omega = ridge_solve(features, labels, num_classes=C, gamma=1e-3)
    pred_acc = (features @ omega.T).argmax(dim=-1).eq(labels).float().mean().item()
    assert pred_acc > 0.85, f"ridge_solve recovered acc={pred_acc:.3f} on linearly-separable data, expected > 0.85"
    print(f"  ridge_solve   passed (linear-separable acc={pred_acc:.3f})")

    # ---- Test 2: init_lora_factors gives correct shapes ----
    class TinyUpper(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(8, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            return F.adaptive_avg_pool2d(x, 1).flatten(1)

    upper = TinyUpper().to(device)
    A_dict, B_dict = init_lora_factors(upper, rank=4, device=device, dtype=torch.float32)
    assert set(A_dict.keys()) == {"conv1", "conv2"}, f"expected conv1/conv2 keys, got {A_dict.keys()}"
    assert A_dict["conv1"].shape == (4, 8, 3, 3)
    assert B_dict["conv1"].shape == (16, 4, 1, 1)
    assert (B_dict["conv1"] == 0).all(), "B should be zero-init"
    assert A_dict["conv1"].requires_grad and B_dict["conv1"].requires_grad
    print(f"  init_lora_factors passed (2 convs, A Kaiming, B=0)")

    # ---- Test 3: forward_with_lora at B=0 == upper(z) ----
    z = torch.randn(8, 8, 4, 4, device=device)
    feat_lora = forward_with_lora_factored(z, upper, A_dict, B_dict)
    feat_base = upper(z)
    assert torch.allclose(feat_lora, feat_base, atol=1e-5), \
        f"forward_with_lora at B=0 should match base; max diff={(feat_lora - feat_base).abs().max():.2e}"
    print(f"  forward_with_lora_factored passed (B=0 → identity)")

    # ---- Test 4: forward_with_lora with B != 0 changes features and grad flows ----
    with torch.no_grad():
        for name in B_dict:
            B_dict[name].add_(0.01 * torch.randn_like(B_dict[name]))
    feat_perturbed = forward_with_lora_factored(z, upper, A_dict, B_dict)
    assert not torch.allclose(feat_perturbed, feat_base, atol=1e-3), \
        "B != 0 should change features"
    # Verify grad flows to A, B and to upper params
    loss = feat_perturbed.sum()
    grads_AB = torch.autograd.grad(loss, [A_dict["conv1"], B_dict["conv1"]], retain_graph=True)
    assert all(g is not None and torch.isfinite(g).all() for g in grads_AB)
    grads_theta = torch.autograd.grad(loss, list(upper.parameters()), allow_unused=False)
    assert all(g is not None and torch.isfinite(g).all() for g in grads_theta)
    print(f"  grad flow OK (∂loss/∂A, ∂loss/∂B, ∂loss/∂θ all finite)")

    # ---- Test 5: project_rank_r_ball enforces norm bound ----
    # Fresh factors, set B large so norm exceeds eps
    A_dict2, B_dict2 = init_lora_factors(upper, rank=4, device=device, dtype=torch.float32)
    with torch.no_grad():
        for name in B_dict2:
            B_dict2[name].add_(2.0 * torch.randn_like(B_dict2[name]))
    eps = 0.1
    project_rank_r_ball(A_dict2, B_dict2, eps=eps)
    for name in A_dict2:
        BA = B_dict2[name].flatten(1) @ A_dict2[name].flatten(1)
        norm = BA.norm().item()
        assert norm <= eps + 1e-5, f"after projection, ||B@A||={norm:.4f} > eps={eps}"
    # Leaves still have requires_grad=True after projection
    assert A_dict2["conv1"].requires_grad and B_dict2["conv1"].requires_grad
    print(f"  project_rank_r_ball passed (norm bounded at eps={eps}, leaves preserved)")

    # ---- Test 6: end-to-end PGD ascent step ----
    # Simulate one inner-loop iteration: forward, ridge_solve, CE loss, grad step, project.
    A_dict3, B_dict3 = init_lora_factors(upper, rank=4, device=device, dtype=torch.float32)
    z_h = torch.randn(16, 8, 4, 4, device=device)
    y_h = torch.randint(0, 5, (16,), device=device)
    eta_pgd = 0.01
    eps_pgd = 0.5
    for it in range(3):
        feat_h = forward_with_lora_factored(z_h, upper, A_dict3, B_dict3)
        omega_H = ridge_solve(feat_h, y_h, num_classes=5, gamma=1e-3)
        L_adv = F.cross_entropy(feat_h @ omega_H.T, y_h)
        grads = torch.autograd.grad(L_adv, list(A_dict3.values()) + list(B_dict3.values()))
        nA = len(A_dict3)
        with torch.no_grad():
            for i, name in enumerate(A_dict3):
                A_dict3[name].sub_(eta_pgd * grads[i])
                B_dict3[name].sub_(eta_pgd * grads[nA + i])
        project_rank_r_ball(A_dict3, B_dict3, eps=eps_pgd)
    # Final L_adv should be finite (no NaN)
    feat_final = forward_with_lora_factored(z_h, upper, A_dict3, B_dict3)
    omega_final = ridge_solve(feat_final, y_h, num_classes=5, gamma=1e-3)
    L_final = F.cross_entropy(feat_final @ omega_final.T, y_h)
    assert torch.isfinite(L_final).item(), "PGD inner loop produced NaN"
    print(f"  end-to-end PGD passed (k=3 inner steps, final L_adv={L_final.item():.4f})")

    print("\nrobust_v7.py self-test: ALL PASSED")


if __name__ == "__main__":
    _self_test()
