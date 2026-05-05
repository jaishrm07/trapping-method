"""Stage 7 — robust optimization in LoRA-r weight ball.

Replaces the trap mechanism (Stage 4–5) with adversarial training:
at each defender step, an inner PGD finds the rank-r LoRA factors
(A, B) maximizing harmful classifier success at θ + B@A. The defender
optimizes θ to make this worst-case post-perturbation harmful CE high.

Defender objective:
    L_total(θ) = L_primary(θ) − λ_robust · L_harm(θ + Δ*, ω_H*(θ + Δ*))

where:
    Δ* = argmin_{rank-r, ||·|| ≤ ε} L_harm(θ + Δ, ω_H*(θ + Δ))
    ω_H* = closed-form ridge solve given features at θ + Δ

See `chris-thomas/research/threads/07_v7_design.md` for the full design.

Run:
    python experiments/run_immunization_v7.py --config configs/immunize_v7.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_dataset_by_name
from src.metrics import relative_immunization_ratio
from src.models import (
    get_resnet18_full_extractor_from_split,
    get_resnet18_split,
    get_resnet18_extractor,
)
from src.provenance import capture_provenance
from src.robust_v7 import (
    forward_with_lora_factored,
    init_lora_factors,
    project_rank_r_ball,
    ridge_solve,
)
from src.utils import get_device, set_seed


def maybe_subset(dataset, max_n: int | None):
    if max_n is None or max_n <= 0 or max_n >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_n)))


def evaluate_imagenet_top1(extractor, head, loader, device):
    extractor.eval()
    head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = head(extractor(x))
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def inner_pgd_worst_case(
    upper: nn.Module,
    z_h_train: torch.Tensor,
    y_h_train: torch.Tensor,
    z_h_eval: torch.Tensor,
    y_h_eval: torch.Tensor,
    *,
    num_classes_h: int,
    rank: int,
    eps: float,
    eta_pgd: float,
    k_pgd: int,
    gamma_ridge: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict, dict, float]:
    """Inner PGD: find rank-r LoRA factors (A, B) maximizing the adversary's
    *generalization* on a held-out harmful eval batch.

    At each PGD iteration:
      1. Forward x_h_train and x_h_eval through θ + B@A
      2. Fit ω_H via closed-form ridge on (feat_train, y_train) — frozen
      3. Adversary's loss = CE(feat_eval @ ω_Hᵀ, y_eval)  ← held-out
      4. PGD on (A, B) descends this loss

    Held-out eval is necessary because Cars has 196 classes and our 64-sample
    batch has only ~64 distinct labels present; ridge on a single batch
    trivially memorizes its own labels regardless of Δ. Using a *different*
    batch for evaluation forces the adversary to find Δ that produces
    *generalizable* harmful-discriminative features — closer to what a real
    LoRA fine-tune attacker is doing.

    Returns:
        A_dict, B_dict — leaf tensors with the post-PGD factors
        adv_acc_final — adversary's final accuracy on the eval batch
    """
    A_dict, B_dict = init_lora_factors(upper, rank=rank, device=device, dtype=dtype)

    for _ in range(k_pgd):
        feat_train = forward_with_lora_factored(z_h_train, upper, A_dict, B_dict)
        feat_eval = forward_with_lora_factored(z_h_eval, upper, A_dict, B_dict)
        # Solve ω_H on train batch — detached so adversary treats it as fixed
        # per step. Alternating opt: ω_H re-fits each PGD iteration.
        with torch.no_grad():
            omega_H = ridge_solve(feat_train, y_h_train, num_classes=num_classes_h, gamma=gamma_ridge)
        # Adversary's loss = CE on EVAL batch (held-out generalization)
        L_adv = F.cross_entropy(feat_eval @ omega_H.T, y_h_eval)
        grads = torch.autograd.grad(L_adv, list(A_dict.values()) + list(B_dict.values()))
        n_A = len(A_dict)
        with torch.no_grad():
            for i, name in enumerate(A_dict):
                A_dict[name].sub_(eta_pgd * grads[i])
                B_dict[name].sub_(eta_pgd * grads[n_A + i])
        project_rank_r_ball(A_dict, B_dict, eps=eps)

    # Final adv accuracy: ω_H freshly fit on train, evaluated on eval
    with torch.no_grad():
        feat_train_final = forward_with_lora_factored(z_h_train, upper, A_dict, B_dict)
        feat_eval_final = forward_with_lora_factored(z_h_eval, upper, A_dict, B_dict)
        omega_final = ridge_solve(feat_train_final, y_h_train,
                                  num_classes=num_classes_h, gamma=gamma_ridge)
        adv_acc = (feat_eval_final @ omega_final.T).argmax(dim=-1).eq(y_h_eval).float().mean().item()

    return A_dict, B_dict, adv_acc


def train_v7_immunization(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    device = get_device()
    print(f"Device: {device}")

    # --- data ---------------------------------------------------------------
    splits_P = load_dataset_by_name(cfg["primary"]["dataset"], root=cfg["primary"]["root"], image_size=cfg["image_size"])
    splits_H = load_dataset_by_name(cfg["harmful"]["dataset"], root=cfg["harmful"]["root"], image_size=cfg["image_size"])
    train_P = maybe_subset(splits_P.train, cfg["primary"].get("max_train"))
    train_H = maybe_subset(splits_H.train, cfg["harmful"].get("max_train"))
    nw = cfg["num_workers"]
    persist = nw > 0
    loader_P = DataLoader(train_P, batch_size=cfg["batch_size"], shuffle=True, num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=persist)
    loader_H = DataLoader(train_H, batch_size=cfg["batch_size"], shuffle=True, num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=persist)
    test_P = DataLoader(splits_P.test, batch_size=cfg["batch_size"], shuffle=False, num_workers=nw, pin_memory=True, persistent_workers=persist)

    print(f"Primary {cfg['primary']['dataset']}: {len(train_P)} train")
    print(f"Harmful {cfg['harmful']['dataset']}: {len(train_H)} train, {splits_H.num_classes} classes")

    # --- model --------------------------------------------------------------
    lower, upper, primary_head = get_resnet18_split()
    lower = lower.to(device)
    upper = upper.to(device)
    primary_head = primary_head.to(device)

    baseline_extractor = get_resnet18_extractor().to(device).eval()

    # --- optimizer ----------------------------------------------------------
    params = list(upper.parameters()) + list(primary_head.parameters())
    optim_outer = torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])

    # --- v7 hyperparameters -------------------------------------------------
    lambda_robust = float(cfg["lambda_robust"])
    rank = int(cfg["rank"])
    eps = float(cfg["eps"])
    k_pgd = int(cfg["k_pgd"])
    eta_pgd = float(cfg.get("eta_pgd", eps / 4.0))
    gamma_ridge = float(cfg.get("gamma_ridge", 1e-3))
    print(f"[v7] rank={rank}, eps={eps}, k_pgd={k_pgd}, eta_pgd={eta_pgd}, "
          f"gamma_ridge={gamma_ridge}, lambda_robust={lambda_robust}")

    iters = cfg["iterations"]
    log_every = cfg["log_every"]
    eval_every = cfg["eval_every"]

    # Two cycle iterators on the same loader → train/eval split per step
    iter_H_train = cycle(loader_H)
    iter_H_eval = cycle(loader_H)
    history = []
    pbar = tqdm(total=iters, desc="immunize_v7")

    step = 0
    for x_P, y_P in cycle(loader_P):
        if step >= iters:
            break
        x_H_train, y_H_train = next(iter_H_train)
        x_H_eval, y_H_eval = next(iter_H_eval)
        x_P = x_P.to(device, non_blocking=True)
        y_P = y_P.to(device, non_blocking=True)
        x_H_train = x_H_train.to(device, non_blocking=True)
        y_H_train = y_H_train.to(device, non_blocking=True)
        x_H_eval = x_H_eval.to(device, non_blocking=True)
        y_H_eval = y_H_eval.to(device, non_blocking=True)

        # Frozen lower features
        with torch.no_grad():
            z_P = lower(x_P)
            z_H_train = lower(x_H_train)
            z_H_eval = lower(x_H_eval)

        # --- primary CE on θ_upper + primary_head ---------------------------
        feat_P = upper(z_P)
        L_primary = F.cross_entropy(primary_head(feat_P), y_P)

        # --- inner PGD: worst-case rank-r LoRA Δ (held-out adv eval) -------
        A_dict, B_dict, adv_acc = inner_pgd_worst_case(
            upper, z_H_train, y_H_train, z_H_eval, y_H_eval,
            num_classes_h=splits_H.num_classes,
            rank=rank, eps=eps, eta_pgd=eta_pgd, k_pgd=k_pgd,
            gamma_ridge=gamma_ridge, device=device, dtype=torch.float32,
        )

        # Detach LoRA factors → FOMAML on outer (no second-order through PGD)
        A_det = {k: v.detach() for k, v in A_dict.items()}
        B_det = {k: v.detach() for k, v in B_dict.items()}

        # Defender's loss: post-PGD adversary's success on EVAL batch
        # (ω_H freshly fitted on train batch's perturbed features, detached)
        feat_train_adv = forward_with_lora_factored(z_H_train, upper, A_det, B_det)
        feat_eval_adv = forward_with_lora_factored(z_H_eval, upper, A_det, B_det)
        omega_H_adv = ridge_solve(feat_train_adv.detach(), y_H_train,
                                  num_classes=splits_H.num_classes, gamma=gamma_ridge)
        L_post_pgd = F.cross_entropy(feat_eval_adv @ omega_H_adv.T, y_H_eval)

        # Total: minimize primary CE, MAXIMIZE harmful CE at worst-case Δ
        loss = L_primary - lambda_robust * L_post_pgd

        optim_outer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.get("grad_clip", 1.0))
        optim_outer.step()

        if step % log_every == 0:
            postfix = {
                "primary": f"{L_primary.item():.3f}",
                "harm@Δ*": f"{L_post_pgd.item():.3f}",
                "adv_acc": f"{adv_acc:.3f}",
            }
            pbar.set_postfix(**postfix)
            history.append({
                "step": step,
                "loss_primary": float(L_primary.item()),
                "loss_post_pgd_harm": float(L_post_pgd.item()),
                "adv_acc_final": float(adv_acc),
                "loss_total": float(loss.item()),
            })

        if eval_every > 0 and step > 0 and step % eval_every == 0:
            print(f"\n[step {step}] running RIR eval (sampling {cfg['rir_num_groups']} × {cfg['rir_group_size']})")
            immu_extractor = get_resnet18_full_extractor_from_split(lower, upper)
            rir = relative_immunization_ratio(
                extractor_immunized=immu_extractor,
                extractor_baseline=baseline_extractor,
                dataset_harmful=splits_H.train,
                dataset_primary=splits_P.train,
                num_groups=cfg["rir_num_groups"],
                group_size=cfg["rir_group_size"],
                device=device,
                seed=cfg["seed"],
            )
            print(f"  RIR={rir['rir']:.3f} | κ_H_immu={rir['kappa_H_immunized']:.3f} | κ_P_immu={rir['kappa_P_immunized']:.3f}")
            history.append({"step": step, "rir": rir})

        step += 1
        pbar.update(1)
    pbar.close()

    # --- final RIR + primary accuracy ---------------------------------------
    print("\nFinal RIR eval...")
    immu_extractor = get_resnet18_full_extractor_from_split(lower, upper)
    final_rir = relative_immunization_ratio(
        extractor_immunized=immu_extractor,
        extractor_baseline=baseline_extractor,
        dataset_harmful=splits_H.train,
        dataset_primary=splits_P.train,
        num_groups=cfg["rir_num_groups"],
        group_size=cfg["rir_group_size"],
        device=device,
        seed=cfg["seed"],
    )

    print("Primary task (ImageNet val) top-1 of immunized model...")
    final_primary_acc = evaluate_imagenet_top1(immu_extractor, primary_head, test_P, device)

    return {
        "config": cfg,
        "provenance": capture_provenance(),
        "final_rir": final_rir,
        "final_primary_acc": final_primary_acc,
        "history": history,
        "_lower_state": lower.state_dict(),
        "_upper_state": upper.state_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/immunize_v7.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(cfg["results_dir"]) / cfg["run_name"]
    results_dir.mkdir(parents=True, exist_ok=True)

    out = train_v7_immunization(cfg)

    extractor_path = results_dir / "extractor.pt"
    torch.save({"lower": out.pop("_lower_state"), "upper": out.pop("_upper_state")}, extractor_path)
    print(f"Saved immunized extractor → {extractor_path}")

    out_path = results_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nFinal RIR: {out['final_rir']['rir']:.3f}")
    print(f"Final primary acc: {out['final_primary_acc'] * 100:.2f}%")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
