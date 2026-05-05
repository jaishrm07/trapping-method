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

- `trap_loss_lora_taylor_hvp(...)` — exact Taylor predictor in the expanded
  LoRA/head parameter space, using a Hessian-vector product to compute
  `Δφᵀ H_0 Δφ` without materializing `H_0`.

- `trap_loss_lora_bonly_ce_block(...)` — adapter-basis-aware loss. Freezes
  random LoRA A, trains only LoRA B plus the harmful head for k inner steps,
  then penalizes post-attack CE below random-guess CE.

- `trap_loss_multiop(...)` — randomly samples one operator per defender
  step (Plan C). The trap is shaped against a *mixture* of adversaries.
"""
from __future__ import annotations

import math
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


def knn_centroid_init_grad(features: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Gradient-aware version of `knn_centroid_init` for D2 ablation.

    Same per-class mean, but features are NOT detached — gradient flows back
    to θ_upper through the centroid init. Used to test whether paper's trap
    synergizes with r_ill via a non-detached centroid path.
    """
    if features.dim() != 2:
        raise ValueError(f"features must be [B, D_hid], got {tuple(features.shape)}")
    D = features.shape[1]
    rows = []
    global_mean = features.mean(dim=0)
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            rows.append(features[mask].mean(dim=0))
        else:
            rows.append(global_mean)
    return torch.stack(rows, dim=0)


def trap_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    k_inner: int = 3,
    eta_inner: float = 0.01,
    K_normalize_by_B: bool = True,
    detach_centroid_init: bool = True,
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

    # 1. KNN centroid init (no-grad by default; opt-in to gradient flow via
    #    `detach_centroid_init=False` for the D2 ablation that tests whether
    #    paper synergy with r_ill comes from a non-detached centroid path).
    if detach_centroid_init:
        omega_h_0 = knn_centroid_init(features.detach(), labels, num_classes).clone()
        omega_h_0.requires_grad_(True)
    else:
        omega_h_0 = knn_centroid_init_grad(features, labels, num_classes)
        # omega_h_0 is non-leaf; autograd.grad supports non-leaf inputs.

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
    # K_normalize_by_B controls whether K is the BATCH-AVERAGE covariance
    # (`X^T X / B`, our default) or the BATCH-SUM covariance (`X^T X`,
    # matching Zheng's K_H definition). Differs by a factor of B in the
    # quadratic term — meaningful at B=64.
    K = (features.T @ features) / (B if K_normalize_by_B else 1.0)
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
# v4 LoRA trap — FOMAML + Form (c) per-step linear-bound predictor
# -----------------------------------------------------------------------------

def trap_loss_lora_v2(
    upper: nn.Module,
    z_h: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    k_inner: int = 3,
    eta_inner: float = 0.1,
    lora_rank: int = 8,
    use_predictor: bool = True,
) -> torch.Tensor:
    """FOMAML LoRA trap with per-step linear-bound predictor.

    Two changes vs `trap_loss_lora` (v1), each independently motivated:

    1. **FOMAML inner update.** Inner-step parameter updates use detached
       gradients (no `create_graph=True` on the chain that produces φ^{t+1}).
       Tamirisa et al. 2024 (TAR, ICLR 2025) use FOMAML at k=64 stably;
       Nichol et al. 2018 (Reptile) prove first-order is sufficient. Our
       earlier second-order unrolls produced softplus values up to 360 from
       LoRA inner overshoot — symptom of through-step Hessian chain.

    2. **Predictor `Σ_t η · ‖g_t‖²` subtracted from ΔL_act.** Per-step
       Taylor first-order bound on the realized reduction (research
       thread 01, Form (c)). Penalizes only EXCESS reduction beyond what
       linear SGD progress predicts, not the trivial baseline. The
       per-step ‖g_t‖² is computed with `create_graph=True` so it stays
       differentiable w.r.t. θ_upper — costs one HVP per step.

    Cost: roughly the same as v1 (k forwards + k backwards through inner
    loop, with the create_graph=True only on the predictor branch).
    Stability is dramatically better — FOMAML eliminates the second-order
    gradient blow-up.
    """
    if z_h.dim() != 4:
        raise ValueError(f"z_h must be [B, C, H, W], got {tuple(z_h.shape)}")
    device = z_h.device
    dtype = z_h.dtype

    conv_specs = [(name, mod) for name, mod in upper.named_modules() if isinstance(mod, nn.Conv2d)]

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

    delta_L_exp = torch.zeros((), device=device, dtype=dtype)

    for _ in range(k_inner):
        L_H_t = forward_with_lora(cur_A, cur_B, cur_omega)
        param_list = list(cur_A.values()) + list(cur_B.values()) + [cur_omega]
        # create_graph=True only when we need the predictor differentiable
        # wrt θ_upper. v4 analysis showed this introduces a 2× cross-Hessian
        # HVP factor in the defender gradient, which causes catastrophic
        # cancellation against ΔL_act and NaN cascade ~step 2000. v4a sets
        # use_predictor=False → grads detached, no HVP, pure FOMAML.
        grads = torch.autograd.grad(L_H_t, param_list, create_graph=use_predictor)
        if use_predictor:
            step_predictor = sum((g * g).sum() for g in grads)
            delta_L_exp = delta_L_exp + eta_inner * step_predictor
            grads_for_update = [g.detach() for g in grads]
        else:
            grads_for_update = grads  # already detached (create_graph=False)
        new_A, new_B = {}, {}
        for i, (name, _) in enumerate(conv_specs):
            new_A[name] = cur_A[name] - eta_inner * grads_for_update[i]
            new_B[name] = cur_B[name] - eta_inner * grads_for_update[n_A + i]
        cur_A, cur_B = new_A, new_B
        cur_omega = cur_omega - eta_inner * grads_for_update[-1]

    L_H_k = forward_with_lora(cur_A, cur_B, cur_omega)
    delta_L_act = L_H_0 - L_H_k
    return F.softplus(delta_L_act - delta_L_exp)


# -----------------------------------------------------------------------------
# Exact LoRA Taylor trap — HVP form of Eq. 4 in φ = {LoRA_A, LoRA_B, ω_H}
# -----------------------------------------------------------------------------

def trap_loss_lora_taylor_hvp(
    upper: nn.Module,
    z_h: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    k_inner: int = 1,
    eta_inner: float = 0.05,
    lora_rank: int = 8,
    inner_create_graph: bool = False,
    detach_delta: bool = True,
    detach_phi_k_for_lk: bool = True,
) -> torch.Tensor:
    """LoRA-space analogue of the paper's Taylor trap, using one HVP.

    Computes Eq. 6 with the adversary parameter vector expanded from the
    harmful head alone to:

        φ = {LoRA_A_l, LoRA_B_l}_l ∪ {ω_H}

    The predicted reduction is exactly:

        ΔL_exp = -g_0ᵀΔφ - 1/2 ΔφᵀH_0Δφ

    where `ΔφᵀH_0Δφ` is computed as a Hessian-vector product with
    `torch.autograd.grad(g_0, φ_0, grad_outputs=Δφ)`. This avoids storing
    the dense Hessian over hundreds of thousands of LoRA/head parameters.

    The default is the conservative ablation:
    - FOMAML-style inner updates (`inner_create_graph=False`)
    - stop-gradient through `Δφ` in the Taylor predictor (`detach_delta=True`)
    - treat `φ_k` as fixed for `L_H(θ, φ_k)` (`detach_phi_k_for_lk=True`)

    These flags keep the Taylor value exact at the sampled trajectory while
    avoiding third-order graph paths through the inner LoRA optimization.
    """
    if z_h.dim() != 4:
        raise ValueError(f"z_h must be [B, C, H, W], got {tuple(z_h.shape)}")
    device = z_h.device
    dtype = z_h.dtype

    conv_specs = [(name, mod) for name, mod in upper.named_modules() if isinstance(mod, nn.Conv2d)]
    if not conv_specs:
        raise ValueError("upper has no Conv2d modules for LoRA Taylor trap")
    conv_names = [name for name, _ in conv_specs]

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

    def params_as_list(A_dict, B_dict, omega):
        return [A_dict[name] for name in conv_names] + [B_dict[name] for name in conv_names] + [omega]

    def forward_with_lora(A_dict, B_dict, omega):
        param_dict = dict(upper.named_parameters())
        for cname in conv_names:
            base_w = param_dict[cname + ".weight"]
            A = A_dict[cname]
            B = B_dict[cname]
            delta_w = (B.flatten(1) @ A.flatten(1)).view_as(base_w)
            param_dict[cname + ".weight"] = base_w + delta_w
        feat = torch.func.functional_call(upper, param_dict, z_h)
        logits = feat @ omega.T
        return F.cross_entropy(logits, labels)

    phi0 = params_as_list(lora_A, lora_B, omega_h_0)
    L_H_0 = forward_with_lora(lora_A, lora_B, omega_h_0)
    g_0 = torch.autograd.grad(L_H_0, phi0, create_graph=True, retain_graph=True)

    cur_A, cur_B, cur_omega = dict(lora_A), dict(lora_B), omega_h_0
    n_A = len(conv_names)
    for _ in range(k_inner):
        L_H_t = forward_with_lora(cur_A, cur_B, cur_omega)
        cur_phi = params_as_list(cur_A, cur_B, cur_omega)
        grads = torch.autograd.grad(L_H_t, cur_phi, create_graph=inner_create_graph)
        if not inner_create_graph:
            grads = tuple(g.detach() for g in grads)

        new_A, new_B = {}, {}
        for i, name in enumerate(conv_names):
            new_A[name] = cur_A[name] - eta_inner * grads[i]
            new_B[name] = cur_B[name] - eta_inner * grads[n_A + i]
        cur_A, cur_B = new_A, new_B
        cur_omega = cur_omega - eta_inner * grads[-1]

    phik = params_as_list(cur_A, cur_B, cur_omega)
    delta_phi = [pk - p0 for pk, p0 in zip(phik, phi0)]
    delta_for_taylor = [d.detach() for d in delta_phi] if detach_delta else delta_phi

    linear_term = sum((g * d).sum() for g, d in zip(g_0, delta_for_taylor))
    hvp = torch.autograd.grad(
        g_0,
        phi0,
        grad_outputs=delta_for_taylor,
        create_graph=True,
        retain_graph=True,
    )
    quadratic_term = sum((d * h).sum() for d, h in zip(delta_for_taylor, hvp))
    delta_L_exp = -(linear_term + 0.5 * quadratic_term)

    if detach_phi_k_for_lk:
        cur_A = {name: cur_A[name].detach() for name in conv_names}
        cur_B = {name: cur_B[name].detach() for name in conv_names}
        cur_omega = cur_omega.detach()
    L_H_k = forward_with_lora(cur_A, cur_B, cur_omega)
    delta_L_act = L_H_0 - L_H_k
    return F.softplus(delta_L_act - delta_L_exp)


# -----------------------------------------------------------------------------
# B-only LoRA block loss — Stage 8 adapter-basis-aware immunization
# -----------------------------------------------------------------------------

def trap_loss_lora_bonly_ce_block(
    upper: nn.Module,
    z_h: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    k_inner: int = 10,
    eta_inner: float = 0.01,
    lora_rank: int = 8,
    ce_threshold: float | None = None,
    inner_create_graph: bool = False,
    detach_inner_updates: bool = True,
) -> torch.Tensor:
    """Block harmful adaptation in the fixed-A LoRA adapter basis.

    Stage 7 showed that linear-probe immunization blocks frozen-feature
    readout but not a weaker LoRA attacker that freezes random A and trains
    only B plus the harmful head. This loss directly simulates that attacker.

    Inner attacker:
        - random LoRA A per conv, frozen
        - zero-initialized LoRA B, trainable
        - harmful linear head, trainable
        - k SGD steps minimizing harmful CE

    Outer loss:
        softplus(ce_threshold - CE_H(theta, B_k, head_k))

    By default `ce_threshold = log(num_classes)`, i.e. random-guess CE. If the
    adapted attacker gets CE below that threshold, the defender is penalized.
    This is deliberately not another Taylor trap; it targets the post-attack
    harmful performance of the adapter-basis operator isolated in Stage 7.
    """
    if z_h.dim() != 4:
        raise ValueError(f"z_h must be [B, C, H, W], got {tuple(z_h.shape)}")
    device = z_h.device
    dtype = z_h.dtype
    threshold = math.log(num_classes) if ce_threshold is None else float(ce_threshold)

    conv_specs = [(name, mod) for name, mod in upper.named_modules() if isinstance(mod, nn.Conv2d)]
    if not conv_specs:
        raise ValueError("upper has no Conv2d modules for B-only LoRA block loss")
    conv_names = [name for name, _ in conv_specs]

    lora_A, lora_B = {}, {}
    for name, conv in conv_specs:
        Cout, Cin = conv.out_channels, conv.in_channels
        K1, K2 = conv.kernel_size
        A = torch.empty(lora_rank, Cin, K1, K2, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(A, a=5 ** 0.5)
        B = torch.zeros(Cout, lora_rank, 1, 1, device=device, dtype=dtype)
        lora_A[name] = A  # fixed random adapter basis
        lora_B[name] = B.requires_grad_(True)

    with torch.no_grad():
        feat_init = upper(z_h)
        omega_h_init = knn_centroid_init(feat_init, labels, num_classes)
    omega_h = omega_h_init.detach().clone().requires_grad_(True)

    def params_as_list(B_dict, omega):
        return [B_dict[name] for name in conv_names] + [omega]

    def forward_with_bonly(B_dict, omega):
        param_dict = dict(upper.named_parameters())
        for cname in conv_names:
            base_w = param_dict[cname + ".weight"]
            A = lora_A[cname]
            B = B_dict[cname]
            delta_w = (B.flatten(1) @ A.flatten(1)).view_as(base_w)
            param_dict[cname + ".weight"] = base_w + delta_w
        feat = torch.func.functional_call(upper, param_dict, z_h)
        logits = feat @ omega.T
        return F.cross_entropy(logits, labels)

    cur_B = dict(lora_B)
    cur_omega = omega_h
    for _ in range(k_inner):
        L_H_t = forward_with_bonly(cur_B, cur_omega)
        cur_phi = params_as_list(cur_B, cur_omega)
        grads = torch.autograd.grad(L_H_t, cur_phi, create_graph=inner_create_graph)
        if detach_inner_updates:
            grads = tuple(g.detach() for g in grads)

        new_B = {}
        for i, name in enumerate(conv_names):
            next_B = cur_B[name] - eta_inner * grads[i]
            if detach_inner_updates:
                next_B = next_B.detach().requires_grad_(True)
            new_B[name] = next_B
        cur_B = new_B

        cur_omega = cur_omega - eta_inner * grads[-1]
        if detach_inner_updates:
            cur_omega = cur_omega.detach().requires_grad_(True)

    L_H_k = forward_with_bonly(cur_B, cur_omega)
    return F.softplus(torch.as_tensor(threshold, device=device, dtype=dtype) - L_H_k)


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
    max_value: float = 10.0,
    lora_variant: str = "v1",
    use_predictor: bool = True,
    taylor_inner_create_graph: bool = False,
    taylor_detach_delta: bool = True,
    taylor_detach_phi_k_for_lk: bool = True,
    bonly_ce_threshold: float | None = None,
    bonly_inner_create_graph: bool = False,
    bonly_detach_inner_updates: bool = True,
    op_weights: dict | None = None,
    dro_decay: float = 0.95,
) -> torch.Tensor:
    """Plan C: per-step operator selection (uniform random, or DRO-weighted).

    Samples one operator from `operators`, runs the corresponding inner-loop
    simulation, returns its trap loss. Over many defender steps the trap is
    shaped against the operator mixture.

    Currently supports:
        "linear_probe" → trap_loss(feat_h, labels, ...)
        "lora_bonly_r4"/"lora_bonly_r8"/... → fixed-A LoRA B-only
        post-attack CE block loss.
        "lora_r4"/"lora_r8"/"lora_r16"/"lora_r32" → trap_loss_lora(...) at
        the corresponding rank. Add new ranks via `lora_rank_for`.

    `lora_variant`:
        "v1" → original `trap_loss_lora` (second-order MAML, no predictor).
        "v2" → `trap_loss_lora_v2` (FOMAML + optional predictor).
        "taylor_hvp" → exact LoRA/head Taylor predictor via HVP.

    `op_weights` (DRO mode, thread 03): if provided, a mutable dict
    `{op_name: float}` of running-mean trap values. Sampling is proportional
    to these (clamped to [0.1, 10] for stability). After computing the trap,
    the entry is updated in place via:
        op_weights[op] = dro_decay * old + (1 - dro_decay) * trap_value
    Operators where the defender is *currently failing* (high trap value)
    get sampled more often. If None, uniform random sampling (Plan C).

    The result is clamped at `max_value` to neutralize occasional runaway
    LoRA inner-loop steps (η=0.1 + B-init-zero can produce softplus values
    in the hundreds at step 0; without clamping these dominate the defender
    gradient and collapse primary features within ~1500 steps).
    """
    if not operators:
        raise ValueError("operators list must be non-empty")
    if lora_rank_for is None:
        lora_rank_for = {
            "lora_r4": 4, "lora_r8": 8, "lora_r16": 16, "lora_r32": 32,
            "lora_bonly_r4": 4, "lora_bonly_r8": 8, "lora_bonly_r16": 16, "lora_bonly_r32": 32,
        }

    if op_weights is None:
        op = random.choice(operators)
    else:
        # DRO sampling: proportional to running-mean trap values, clamped.
        weights = [max(0.1, min(10.0, op_weights.get(o, 1.0))) for o in operators]
        total = sum(weights)
        r = random.uniform(0, total)
        cum = 0.0
        op = operators[-1]
        for o, w in zip(operators, weights):
            cum += w
            if r <= cum:
                op = o
                break

    if op == "linear_probe":
        raw = trap_loss(feat_h, labels, num_classes=num_classes,
                        k_inner=k_inner, eta_inner=eta_inner)
    elif op.startswith("lora_bonly_r"):
        try:
            rank = int(op.split("_r", 1)[1])
        except ValueError:
            rank = lora_rank_for.get(op)
        if rank is None:
            raise ValueError(f"Cannot parse rank from {op}; expected lora_bonly_r<int>")
        raw = trap_loss_lora_bonly_ce_block(
            upper, z_h, labels, num_classes=num_classes,
            k_inner=k_inner, eta_inner=eta_inner,
            lora_rank=rank,
            ce_threshold=bonly_ce_threshold,
            inner_create_graph=bonly_inner_create_graph,
            detach_inner_updates=bonly_detach_inner_updates,
        )
    elif op in lora_rank_for:
        if lora_variant == "taylor_hvp":
            raw = trap_loss_lora_taylor_hvp(
                upper, z_h, labels, num_classes=num_classes,
                k_inner=k_inner, eta_inner=eta_inner,
                lora_rank=lora_rank_for[op],
                inner_create_graph=taylor_inner_create_graph,
                detach_delta=taylor_detach_delta,
                detach_phi_k_for_lk=taylor_detach_phi_k_for_lk,
            )
        elif lora_variant == "v2":
            raw = trap_loss_lora_v2(upper, z_h, labels, num_classes=num_classes,
                                    k_inner=k_inner, eta_inner=eta_inner,
                                    lora_rank=lora_rank_for[op],
                                    use_predictor=use_predictor)
        else:
            raw = trap_loss_lora(upper, z_h, labels, num_classes=num_classes,
                                 k_inner=k_inner, eta_inner=eta_inner,
                                 lora_rank=lora_rank_for[op])
    else:
        raise ValueError(f"Unknown operator: {op}. Supported: linear_probe, "
                         f"lora_bonly_r<int>, and any key in lora_rank_for ({list(lora_rank_for)})")

    clamped = raw.clamp(max=max_value)

    # DRO update: running mean of trap value per operator (after clamp).
    if op_weights is not None:
        old = op_weights.get(op, 1.0)
        op_weights[op] = dro_decay * old + (1.0 - dro_decay) * float(clamped.detach().item())

    return clamped


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

    loss_taylor = trap_loss_lora_taylor_hvp(
        upper, z_h, labels2, num_classes=C,
        k_inner=1, eta_inner=0.05, lora_rank=2,
    )
    assert loss_taylor.requires_grad and loss_taylor.numel() == 1 and loss_taylor.item() >= 0
    grad_taylor = torch.autograd.grad(loss_taylor, list(upper.parameters()), allow_unused=False)
    grad_taylor_norms = [g.norm().item() if g is not None else 0.0 for g in grad_taylor]
    assert all(torch.isfinite(g).all() for g in grad_taylor if g is not None)
    print(f"trap_loss_taylor passed (loss={loss_taylor.item():.6f}, max||grad||={max(grad_taylor_norms):.4f})")

    loss_bonly = trap_loss_lora_bonly_ce_block(
        upper, z_h, labels2, num_classes=C,
        k_inner=2, eta_inner=0.01, lora_rank=2,
    )
    assert loss_bonly.requires_grad and loss_bonly.numel() == 1 and loss_bonly.item() >= 0
    grad_bonly = torch.autograd.grad(loss_bonly, list(upper.parameters()), allow_unused=False)
    grad_bonly_norms = [g.norm().item() if g is not None else 0.0 for g in grad_bonly]
    assert all(torch.isfinite(g).all() for g in grad_bonly if g is not None)
    print(f"trap_loss_bonly  passed (loss={loss_bonly.item():.6f}, max||grad||={max(grad_bonly_norms):.4f})")

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
    l_bonly = trap_loss_multiop(upper, feat_h, z_h, labels2,
                                num_classes=C, operators=["lora_bonly_r2"],
                                k_inner=2, eta_inner=0.01)
    assert l_bonly.requires_grad and l_bonly.numel() == 1 and l_bonly.item() >= 0
    print(f"trap_loss_multiop passed (5 samples, range=[{min(losses):.4f}, {max(losses):.4f}], "
          f"bonly={l_bonly.item():.4f})")


if __name__ == "__main__":
    _self_test()
