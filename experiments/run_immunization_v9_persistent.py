"""Stage 9 — persistent B-only LoRA attacker immunization.

v8a showed that a fresh k=10 B-only inner loop does not move the 50-epoch
B-only LoRA probe. v9 changes the time scale: the attacker state persists
across defender steps and keeps training, so the defender sees an actually
trained adapter rather than a freshly initialized local probe.

Threat operator:
    A_l: fixed random LoRA basis per conv
    B_l: persistent trainable LoRA B factors
    omega, bias: persistent harmful classifier head

Alternating update:
    1. Attacker minimizes harmful CE over B/head for m minibatch steps.
    2. Defender minimizes primary CE + CN regularizers +
       lambda_block * softplus(log(C_H) - harmful_CE(current attacker)).

Run:
    python experiments/run_immunization_v9_persistent.py \
        --config configs/immunize_v9a_persistent_bonly.yaml
"""
from __future__ import annotations

import argparse
import json
import math
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
from src.hessian import condition_number
from src.k_inv_layer import k_inv_dummy_layer
from src.losses import r_ill, r_well
from src.metrics import relative_immunization_ratio
from src.models import (
    get_resnet18_extractor,
    get_resnet18_full_extractor_from_split,
    get_resnet18_split,
)
from src.provenance import capture_provenance
from src.robust_v7 import forward_with_lora_factored, init_lora_factors
from src.utils import get_device, set_seed


def maybe_subset(dataset, max_n: int | None):
    if max_n is None or max_n <= 0 or max_n >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_n)))


def feature_hessian(features: torch.Tensor) -> torch.Tensor:
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


def init_persistent_bonly_attacker(
    upper: nn.Module,
    *,
    rank: int,
    num_classes: int,
    feature_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    A_dict, B_dict = init_lora_factors(upper, rank=rank, device=device, dtype=dtype)
    for A in A_dict.values():
        A.requires_grad_(False)

    omega = torch.empty(num_classes, feature_dim, device=device, dtype=dtype)
    nn.init.kaiming_normal_(omega, mode="fan_out", nonlinearity="linear")
    omega.requires_grad_(True)
    bias = torch.zeros(num_classes, device=device, dtype=dtype, requires_grad=True)
    return A_dict, B_dict, omega, bias


def attacker_params(B_dict: dict[str, torch.Tensor], omega: torch.Tensor, bias: torch.Tensor) -> list[torch.Tensor]:
    return list(B_dict.values()) + [omega, bias]


def harmful_logits_bonly(
    upper: nn.Module,
    z_h: torch.Tensor,
    A_dict: dict[str, torch.Tensor],
    B_dict: dict[str, torch.Tensor],
    omega: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    feat = forward_with_lora_factored(z_h, upper, A_dict, B_dict)
    return feat @ omega.T + bias


def make_attacker_optimizer(params: list[torch.Tensor], cfg: dict) -> torch.optim.Optimizer:
    opt_name = str(cfg.get("attacker_optimizer", "adamw")).lower()
    lr = float(cfg.get("attacker_lr", 1e-3))
    wd = float(cfg.get("attacker_weight_decay", 0.0))
    if opt_name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(cfg.get("attacker_momentum", 0.9)),
            weight_decay=wd,
        )
    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown attacker_optimizer={opt_name}; expected adamw or sgd")


def attacker_minibatch_step(
    lower: nn.Module,
    upper: nn.Module,
    x_h: torch.Tensor,
    y_h: torch.Tensor,
    A_dict: dict[str, torch.Tensor],
    B_dict: dict[str, torch.Tensor],
    omega: torch.Tensor,
    bias: torch.Tensor,
    optim_attacker: torch.optim.Optimizer,
    *,
    grad_clip: float,
) -> tuple[float, float]:
    with torch.no_grad():
        z_h = lower(x_h)

    logits = harmful_logits_bonly(upper, z_h, A_dict, B_dict, omega, bias)
    loss = F.cross_entropy(logits, y_h)
    params = attacker_params(B_dict, omega, bias)
    grads = torch.autograd.grad(loss, params, create_graph=False)

    optim_attacker.zero_grad(set_to_none=True)
    for p, g in zip(params, grads):
        p.grad = g.detach()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
    optim_attacker.step()

    acc = logits.detach().argmax(dim=-1).eq(y_h).float().mean().item()
    return float(loss.detach().item()), float(acc)


def train_v9_persistent(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    device = get_device()
    print(f"Device: {device}")

    splits_P = load_dataset_by_name(cfg["primary"]["dataset"], root=cfg["primary"]["root"], image_size=cfg["image_size"])
    splits_H = load_dataset_by_name(cfg["harmful"]["dataset"], root=cfg["harmful"]["root"], image_size=cfg["image_size"])
    train_P = maybe_subset(splits_P.train, cfg["primary"].get("max_train"))
    train_H = maybe_subset(splits_H.train, cfg["harmful"].get("max_train"))

    nw = cfg["num_workers"]
    persist = nw > 0
    loader_P = DataLoader(train_P, batch_size=cfg["batch_size"], shuffle=True, num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=persist)
    loader_H = DataLoader(train_H, batch_size=cfg["batch_size"], shuffle=True, num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=persist)
    test_P = DataLoader(splits_P.test, batch_size=cfg["batch_size"], shuffle=False, num_workers=nw, pin_memory=True, persistent_workers=persist)

    print(f"Primary {cfg['primary']['dataset']}: {len(train_P)} train (max-capped)")
    print(f"Harmful {cfg['harmful']['dataset']}: {len(train_H)} train (max-capped), {splits_H.num_classes} classes")

    lower, upper, primary_head = get_resnet18_split()
    lower = lower.to(device)
    upper = upper.to(device)
    primary_head = primary_head.to(device)
    baseline_extractor = get_resnet18_extractor().to(device).eval()

    rank = int(cfg.get("rank", 8))
    A_dict, B_dict, omega_h, bias_h = init_persistent_bonly_attacker(
        upper,
        rank=rank,
        num_classes=splits_H.num_classes,
        feature_dim=512,
        device=device,
        dtype=torch.float32,
    )

    defender_params = list(upper.parameters()) + list(primary_head.parameters())
    optim_defender = torch.optim.SGD(
        defender_params,
        lr=cfg["lr"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
    )
    att_params = attacker_params(B_dict, omega_h, bias_h)
    optim_attacker = make_attacker_optimizer(att_params, cfg)

    lambda_block = float(cfg.get("lambda_block", 0.3))
    lambda_well = float(cfg.get("lambda_well", 1.0))
    lambda_ill = float(cfg.get("lambda_ill", 1.0))
    ce_threshold_cfg = cfg.get("ce_threshold")
    ce_threshold = math.log(splits_H.num_classes) if ce_threshold_cfg is None else float(ce_threshold_cfg)
    attacker_steps = int(cfg.get("attacker_steps_per_outer", 3))
    attacker_grad_clip = float(cfg.get("attacker_grad_clip", 5.0))
    defender_grad_clip = float(cfg.get("grad_clip", 1.0))
    use_k_inv = bool(cfg.get("use_k_inv_preconditioner", True))
    k_inv_ridge = float(cfg.get("k_inv_ridge", 1e-2))

    print(f"[v9] persistent B-only LoRA: rank={rank}, attacker_steps={attacker_steps}, "
          f"attacker_opt={cfg.get('attacker_optimizer', 'adamw')}, attacker_lr={cfg.get('attacker_lr', 1e-3)}, "
          f"lambda_block={lambda_block}, ce_threshold={ce_threshold:.4f}")
    if use_k_inv:
        print(f"[k_inv] dummy-layer preconditioner enabled, ridge={k_inv_ridge}")

    iter_H_att = cycle(loader_H)
    iter_H_def = cycle(loader_H)
    history = []
    iters = int(cfg["iterations"])
    log_every = int(cfg["log_every"])
    eval_every = int(cfg["eval_every"])
    pbar = tqdm(total=iters, desc="immunize_v9")

    step = 0
    for x_P, y_P in cycle(loader_P):
        if step >= iters:
            break

        # 1. Persistent attacker keeps training on harmful minibatches.
        att_loss_last = 0.0
        att_acc_last = 0.0
        for _ in range(attacker_steps):
            x_A, y_A = next(iter_H_att)
            x_A = x_A.to(device, non_blocking=True)
            y_A = y_A.to(device, non_blocking=True)
            att_loss_last, att_acc_last = attacker_minibatch_step(
                lower, upper, x_A, y_A,
                A_dict, B_dict, omega_h, bias_h,
                optim_attacker,
                grad_clip=attacker_grad_clip,
            )

        # 2. Defender update against current persistent attacker.
        x_H, y_H = next(iter_H_def)
        x_P = x_P.to(device, non_blocking=True)
        y_P = y_P.to(device, non_blocking=True)
        x_H = x_H.to(device, non_blocking=True)
        y_H = y_H.to(device, non_blocking=True)

        with torch.no_grad():
            z_P = lower(x_P)
            z_H = lower(x_H)

        feat_P = upper(z_P)
        feat_H = upper(z_H)
        L_primary = F.cross_entropy(primary_head(feat_P), y_P)

        feat_P_for_reg = k_inv_dummy_layer(feat_P, ridge=k_inv_ridge) if use_k_inv else feat_P
        feat_H_for_reg = k_inv_dummy_layer(feat_H, ridge=k_inv_ridge) if use_k_inv else feat_H
        H_P = feature_hessian(feat_P_for_reg)
        H_H = feature_hessian(feat_H_for_reg)
        L_well = r_well(H_P)
        L_ill = r_ill(H_H)

        A_det = {name: A.detach() for name, A in A_dict.items()}
        B_det = {name: B.detach() for name, B in B_dict.items()}
        logits_def = harmful_logits_bonly(
            upper, z_H, A_det, B_det, omega_h.detach(), bias_h.detach()
        )
        L_harm_current = F.cross_entropy(logits_def, y_H)
        L_block = F.softplus(torch.as_tensor(ce_threshold, device=device, dtype=feat_P.dtype) - L_harm_current)
        attacker_acc_def = logits_def.detach().argmax(dim=-1).eq(y_H).float().mean().item()

        loss = L_primary + lambda_well * L_well + lambda_ill * L_ill + lambda_block * L_block

        optim_defender.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(defender_params, defender_grad_clip)
        optim_defender.step()

        if step % log_every == 0:
            postfix = {
                "primary": f"{L_primary.item():.3f}",
                "block": f"{L_block.item():.3f}",
                "harm": f"{L_harm_current.item():.3f}",
                "att_acc": f"{attacker_acc_def:.3f}",
            }
            pbar.set_postfix(**postfix)
            history.append({
                "step": step,
                "loss_primary": float(L_primary.item()),
                "loss_well": float(L_well.item()),
                "loss_ill": float(L_ill.item()),
                "loss_block": float(L_block.item()),
                "loss_harm_current": float(L_harm_current.item()),
                "attacker_loss_last": att_loss_last,
                "attacker_acc_last": att_acc_last,
                "attacker_acc_defender_batch": float(attacker_acc_def),
                "kappa_H_batch": condition_number(H_H.detach()),
                "kappa_P_batch": condition_number(H_P.detach()),
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
            print(f"  RIR={rir['rir']:.3f} | kH={rir['kappa_H_immunized']:.3f} | kP={rir['kappa_P_immunized']:.3f}")
            history.append({"step": step, "rir": rir})

        step += 1
        pbar.update(1)

    pbar.close()

    immu_extractor = get_resnet18_full_extractor_from_split(lower, upper)
    if bool(cfg.get("skip_final_eval", False)):
        print("\nSkipping final RIR/primary eval (smoke mode).")
        final_rir = None
        final_primary_acc = None
    else:
        print("\nFinal RIR eval...")
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
        "_attacker_state": {
            "A": {k: v.detach().cpu() for k, v in A_dict.items()},
            "B": {k: v.detach().cpu() for k, v in B_dict.items()},
            "omega": omega_h.detach().cpu(),
            "bias": bias_h.detach().cpu(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/immunize_v9a_persistent_bonly.yaml")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--skip-final-eval", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.iterations is not None:
        cfg["iterations"] = args.iterations
    if args.run_name is not None:
        cfg["run_name"] = args.run_name
    if args.skip_final_eval:
        cfg["skip_final_eval"] = True

    results_dir = Path(cfg["results_dir"]) / cfg["run_name"]
    results_dir.mkdir(parents=True, exist_ok=True)

    out = train_v9_persistent(cfg)

    extractor_path = results_dir / "extractor.pt"
    torch.save({"lower": out.pop("_lower_state"), "upper": out.pop("_upper_state")}, extractor_path)
    print(f"Saved immunized extractor -> {extractor_path}")

    attacker_path = results_dir / "persistent_attacker.pt"
    torch.save(out.pop("_attacker_state"), attacker_path)
    print(f"Saved persistent attacker -> {attacker_path}")

    out_path = results_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    if out["final_rir"] is not None:
        print(f"\nFinal RIR: {out['final_rir']['rir']:.3f}")
    else:
        print("\nFinal RIR: skipped")
    if out["final_primary_acc"] is not None:
        print(f"Final primary acc: {out['final_primary_acc'] * 100:.2f}%")
    else:
        print("Final primary acc: skipped")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
