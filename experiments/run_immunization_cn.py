"""Stage 2 — condition-number immunization (the "CN" baseline of the trapping paper).

Implements Algorithm 1 of Zheng et al. (ICML 2025) for ResNet18:
- Freeze conv1..layer2.
- Train layer3+layer4 with: primary CE (on ImageNet val) + λ_well·R_well(H_P) + λ_ill·R_ill(H_H).
- Evaluate RIR (Eq. 17) periodically and at the end.

Stage-2 known simplification (see src/losses.py docstring): the K^{-1}
preconditioner of Algorithm 1 line 6 is **omitted** — vanilla autograd is used
instead. If RIR comes in much below ~3.5 on Cars/ResNet18, add a dummy-layer
preconditioner per Zheng §4.4.

Run:
    python experiments/run_immunization_cn.py --config configs/immunize_cn.yaml
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

from src.data import load_dataset_by_name, make_loaders
from src.hessian import condition_number
from src.k_inv_layer import k_inv_dummy_layer
from src.losses import r_ill, r_well
from src.metrics import relative_immunization_ratio
from src.models import (
    get_resnet18_full_extractor_from_split,
    get_resnet18_split,
    get_resnet18_extractor,
)
from src.trap_loss import trap_loss, trap_loss_multiop
from src.utils import get_device, set_seed


def maybe_subset(dataset, max_n: int | None):
    if max_n is None or max_n <= 0 or max_n >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_n)))


def feature_hessian(features: torch.Tensor) -> torch.Tensor:
    """H(θ) ≈ (1/B) X̃^T X̃ on a minibatch of features."""
    return features.T @ features / features.size(0)


def evaluate_imagenet_top1(extractor: nn.Module, head: nn.Module, loader: DataLoader, device: torch.device) -> float:
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


def train_cn_immunization(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    device = get_device()
    print(f"Device: {device}")

    # --- data ----------------------------------------------------------------
    splits_P = load_dataset_by_name(cfg["primary"]["dataset"], root=cfg["primary"]["root"], image_size=cfg["image_size"])
    splits_H = load_dataset_by_name(cfg["harmful"]["dataset"], root=cfg["harmful"]["root"], image_size=cfg["image_size"])

    train_P = maybe_subset(splits_P.train, cfg["primary"].get("max_train"))
    train_H = maybe_subset(splits_H.train, cfg["harmful"].get("max_train"))

    loader_P = DataLoader(train_P, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=True, drop_last=True)
    loader_H = DataLoader(train_H, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=True, drop_last=True)
    test_P = DataLoader(splits_P.test, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)

    print(f"Primary {cfg['primary']['dataset']}: {len(train_P)} train (max-capped)")
    print(f"Harmful {cfg['harmful']['dataset']}: {len(train_H)} train (max-capped)")

    # --- model ---------------------------------------------------------------
    lower, upper, primary_head = get_resnet18_split()
    lower = lower.to(device)
    upper = upper.to(device)
    primary_head = primary_head.to(device)

    # Baseline (pre-immunization) extractor for RIR comparison — frozen copy.
    baseline_extractor = get_resnet18_extractor().to(device).eval()

    # --- optimizer -----------------------------------------------------------
    params = list(upper.parameters()) + list(primary_head.parameters())
    optim = torch.optim.SGD(
        params,
        lr=cfg["lr"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
    )

    iters = cfg["iterations"]
    log_every = cfg["log_every"]
    eval_every = cfg["eval_every"]

    # Cycle harmful loader so each iteration always has a harmful batch.
    iter_H = cycle(loader_H)

    # Trap loss is opt-in (Stage 4). lambda_trap > 0 enables.
    lambda_trap = float(cfg.get("lambda_trap", 0.0))
    k_inner_trap = int(cfg.get("trap_k_inner", 3))
    eta_inner_trap = float(cfg.get("trap_eta_inner", 0.01))
    use_trap = lambda_trap > 0
    # Plan C: optional multi-operator trap. Empty list / None → use the
    # original single-operator linear-probing trap. Non-empty list → randomize
    # per defender step over the listed operators.
    trap_operators = cfg.get("trap_operators") or []
    if use_trap:
        if trap_operators:
            print(f"[trap] enabled (multi-op): operators={trap_operators}, "
                  f"λ_trap={lambda_trap}, k_inner={k_inner_trap}, η_inner={eta_inner_trap}")
        else:
            print(f"[trap] enabled (single-op LP): λ_trap={lambda_trap}, "
                  f"k_inner={k_inner_trap}, η_inner={eta_inner_trap}")

    # K^-1 dummy-layer preconditioner (Zheng §4.4). Opt-in via config.
    use_k_inv = bool(cfg.get("use_k_inv_preconditioner", False))
    k_inv_ridge = float(cfg.get("k_inv_ridge", 1e-3))
    if use_k_inv:
        print(f"[k_inv] dummy-layer preconditioner enabled, ridge={k_inv_ridge}")

    history = []
    pbar = tqdm(total=iters, desc="immunize")
    step = 0
    for x_P, y_P in cycle(loader_P):
        if step >= iters:
            break
        x_H, y_H = next(iter_H)
        x_P = x_P.to(device, non_blocking=True)
        y_P = y_P.to(device, non_blocking=True)
        x_H = x_H.to(device, non_blocking=True)
        y_H = y_H.to(device, non_blocking=True)

        # Forward
        with torch.no_grad():
            z_P = lower(x_P)
            z_H = lower(x_H)
        feat_P = upper(z_P)            # [B, 512]
        feat_H = upper(z_H)

        # Primary CE — uses raw features (no K^-1 preconditioning).
        L_primary = F.cross_entropy(primary_head(feat_P), y_P)

        # Regularizer Hessians — wrap features through dummy K^-1 layer if
        # enabled, so backward gradients to θ get K^-1-preconditioned (Zheng §4.4).
        feat_P_for_reg = k_inv_dummy_layer(feat_P, ridge=k_inv_ridge) if use_k_inv else feat_P
        feat_H_for_reg = k_inv_dummy_layer(feat_H, ridge=k_inv_ridge) if use_k_inv else feat_H
        H_P = feature_hessian(feat_P_for_reg)
        H_H = feature_hessian(feat_H_for_reg)

        L_well = r_well(H_P)
        L_ill = r_ill(H_H)
        loss = L_primary + cfg["lambda_well"] * L_well + cfg["lambda_ill"] * L_ill

        L_trap = None
        if use_trap:
            if trap_operators:
                L_trap = trap_loss_multiop(
                    upper, feat_H, z_H, y_H,
                    num_classes=splits_H.num_classes,
                    operators=trap_operators,
                    k_inner=k_inner_trap,
                    eta_inner=eta_inner_trap,
                )
            else:
                # Original LP-only trap (un-preconditioned features).
                L_trap = trap_loss(
                    feat_H, y_H,
                    num_classes=splits_H.num_classes,
                    k_inner=k_inner_trap,
                    eta_inner=eta_inner_trap,
                )
            loss = loss + lambda_trap * L_trap

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.get("grad_clip", 5.0))
        optim.step()

        if step % log_every == 0:
            postfix = {
                "primary": f"{L_primary.item():.3f}",
                "well": f"{L_well.item():.4f}",
                "ill": f"{L_ill.item():.4f}",
            }
            if use_trap:
                postfix["trap"] = f"{L_trap.item():.4f}"
            pbar.set_postfix(**postfix)
            entry = {
                "step": step,
                "loss_primary": float(L_primary.item()),
                "loss_well": float(L_well.item()),
                "loss_ill": float(L_ill.item()),
                "kappa_H_batch": condition_number(H_H.detach()),
                "kappa_P_batch": condition_number(H_P.detach()),
            }
            if use_trap:
                entry["loss_trap"] = float(L_trap.item())
            history.append(entry)

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

    # --- final RIR + primary accuracy ----------------------------------------
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
        "final_rir": final_rir,
        "final_primary_acc": final_primary_acc,
        "history": history,
        # Cached for caller: run_main saves these to extractor.pt
        "_lower_state": lower.state_dict(),
        "_upper_state": upper.state_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/immunize_cn.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(cfg["results_dir"]) / cfg["run_name"]
    results_dir.mkdir(parents=True, exist_ok=True)

    out = train_cn_immunization(cfg)

    # Save the immunized backbone state separately so run_baseline_probe.py can load it.
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
