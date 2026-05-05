"""Stage 12 - LoRA-tangent label-removal immunization.

Stages 9-10 tried to make persistent B-only LoRA attackers fail at the adapted
endpoint. Stage 11 showed the underlying issue more directly: Cars labels remain
readable from learned LoRA tangent features.

This script targets that signal. It maintains a small population of attackers.
Each attacker learns several B-only LoRA directions and a linear classifier on
the finite-difference tangent features:

    T_i(x) = (f_{theta + eps * B_i A_i}(x) - f_theta(x)) / eps

The defender then minimizes primary loss and condition-number regularizers while
maximizing the current strongest attacker's tangent CE via:

    L_tangent = softplus(log(C_H) - min_j CE(q_j([T_1..T_m]), y_H))

Run:
    python experiments/run_immunization_v12_tangent_removal.py \
        --config configs/immunize_v12a_tangent_removal.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_immunization_v9_persistent import (
    evaluate_imagenet_top1,
    feature_hessian,
    maybe_subset,
)
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


@dataclass
class TangentAttacker:
    index: int
    A_list: list[dict[str, torch.Tensor]]
    B_list: list[dict[str, torch.Tensor]]
    omega: torch.Tensor
    bias: torch.Tensor
    optimizer: torch.optim.Optimizer
    train_iter: object
    last_loss: float = 0.0
    last_acc: float = 0.0
    updates: int = 0


def init_head(num_classes: int, feature_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    omega = torch.empty(num_classes, feature_dim, device=device)
    nn.init.kaiming_normal_(omega, mode="fan_out", nonlinearity="linear")
    omega.requires_grad_(True)
    bias = torch.zeros(num_classes, device=device, requires_grad=True)
    return omega, bias


def global_b_norm(B: dict[str, torch.Tensor]) -> torch.Tensor:
    norms = [value.pow(2).sum() for value in B.values()]
    if not norms:
        raise ValueError("empty B dictionary")
    return torch.stack(norms).sum().sqrt()


def scaled_b_direction(
    B: dict[str, torch.Tensor],
    epsilon: float,
    *,
    detach: bool,
) -> dict[str, torch.Tensor]:
    values = {name: value.detach() if detach else value for name, value in B.items()}
    norm = torch.stack([value.pow(2).sum() for value in values.values()]).sum().sqrt().clamp_min(1e-12)
    return {name: value / norm * epsilon for name, value in values.items()}


def clone_detached_tree(tree: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach() for name, value in tree.items()}


def tangent_features(
    upper: nn.Module,
    z_h: torch.Tensor,
    A_list: list[dict[str, torch.Tensor]],
    B_list: list[dict[str, torch.Tensor]],
    *,
    epsilon: float,
    detach_adapter: bool,
) -> torch.Tensor:
    base = upper(z_h)
    pieces = []
    for A, B in zip(A_list, B_list):
        A_use = clone_detached_tree(A) if detach_adapter else A
        B_use = scaled_b_direction(B, epsilon, detach=detach_adapter)
        feat_eps = forward_with_lora_factored(z_h, upper, A_use, B_use)
        pieces.append((feat_eps - base) / epsilon)
    return torch.cat(pieces, dim=1)


def tangent_attacker_params(attacker: TangentAttacker) -> list[torch.Tensor]:
    params: list[torch.Tensor] = []
    for B in attacker.B_list:
        params.extend(B.values())
    params.extend([attacker.omega, attacker.bias])
    return params


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


def init_tangent_attacker(
    upper: nn.Module,
    loader_H: DataLoader,
    *,
    index: int,
    seed: int,
    rank: int,
    directions_per_attacker: int,
    num_classes: int,
    device: torch.device,
    cfg: dict,
) -> TangentAttacker:
    torch.manual_seed(seed)
    A_list: list[dict[str, torch.Tensor]] = []
    B_list: list[dict[str, torch.Tensor]] = []
    b_init_std = float(cfg.get("b_init_std", 1e-3))

    for _ in range(directions_per_attacker):
        A, B = init_lora_factors(upper, rank=rank, device=device, dtype=torch.float32)
        for value in A.values():
            value.requires_grad_(False)
        with torch.no_grad():
            for value in B.values():
                value.normal_(mean=0.0, std=b_init_std)
        A_list.append(A)
        B_list.append(B)

    omega, bias = init_head(num_classes, 512 * directions_per_attacker, device)
    placeholder = TangentAttacker(
        index=index,
        A_list=A_list,
        B_list=B_list,
        omega=omega,
        bias=bias,
        optimizer=None,  # type: ignore[arg-type]
        train_iter=cycle(loader_H),
    )
    placeholder.optimizer = make_attacker_optimizer(tangent_attacker_params(placeholder), cfg)
    return placeholder


def init_population(
    upper: nn.Module,
    loader_H: DataLoader,
    *,
    num_attackers: int,
    seed: int,
    rank: int,
    directions_per_attacker: int,
    num_classes: int,
    device: torch.device,
    cfg: dict,
) -> list[TangentAttacker]:
    return [
        init_tangent_attacker(
            upper,
            loader_H,
            index=idx,
            seed=seed + 1009 * idx,
            rank=rank,
            directions_per_attacker=directions_per_attacker,
            num_classes=num_classes,
            device=device,
            cfg=cfg,
        )
        for idx in range(num_attackers)
    ]


def update_attacker(
    attacker: TangentAttacker,
    lower: nn.Module,
    upper: nn.Module,
    device: torch.device,
    *,
    epsilon: float,
    steps: int,
    grad_clip: float,
) -> None:
    params = tangent_attacker_params(attacker)
    for _ in range(steps):
        x_h, y_h = next(attacker.train_iter)
        x_h = x_h.to(device, non_blocking=True)
        y_h = y_h.to(device, non_blocking=True)
        with torch.no_grad():
            z_h = lower(x_h)

        feat = tangent_features(
            upper,
            z_h,
            attacker.A_list,
            attacker.B_list,
            epsilon=epsilon,
            detach_adapter=False,
        )
        logits = feat @ attacker.omega.T + attacker.bias
        loss = F.cross_entropy(logits, y_h)
        grads = torch.autograd.grad(loss, params, create_graph=False)

        attacker.optimizer.zero_grad(set_to_none=True)
        for param, grad in zip(params, grads):
            param.grad = grad.detach()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        attacker.optimizer.step()

        attacker.last_loss = float(loss.detach().item())
        attacker.last_acc = float(logits.detach().argmax(dim=-1).eq(y_h).float().mean().item())
        attacker.updates += 1


def defender_tangent_ce(
    attacker: TangentAttacker,
    upper: nn.Module,
    z_h: torch.Tensor,
    y_h: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, float]:
    feat = tangent_features(
        upper,
        z_h,
        attacker.A_list,
        attacker.B_list,
        epsilon=epsilon,
        detach_adapter=True,
    )
    omega = attacker.omega.detach()
    bias = attacker.bias.detach()
    logits = feat @ omega.T + bias
    ce = F.cross_entropy(logits, y_h)
    acc = logits.detach().argmax(dim=-1).eq(y_h).float().mean().item()
    return ce, float(acc)


def train_v12_tangent_removal(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    device = get_device()
    print(f"Device: {device}")

    splits_P = load_dataset_by_name(cfg["primary"]["dataset"], root=cfg["primary"]["root"], image_size=cfg["image_size"])
    splits_H = load_dataset_by_name(cfg["harmful"]["dataset"], root=cfg["harmful"]["root"], image_size=cfg["image_size"])
    train_P = maybe_subset(splits_P.train, cfg["primary"].get("max_train"))
    train_H = maybe_subset(splits_H.train, cfg["harmful"].get("max_train"))

    nw = int(cfg["num_workers"])
    persist = nw > 0
    loader_P = DataLoader(train_P, batch_size=cfg["batch_size"], shuffle=True, num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=persist)
    loader_H_def = DataLoader(train_H, batch_size=cfg["batch_size"], shuffle=True, num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=persist)
    loader_H_att = DataLoader(train_H, batch_size=cfg["batch_size"], shuffle=True, num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=persist)
    test_P = DataLoader(splits_P.test, batch_size=cfg["batch_size"], shuffle=False, num_workers=nw, pin_memory=True, persistent_workers=persist)

    print(f"Primary {cfg['primary']['dataset']}: {len(train_P)} train (max-capped)")
    print(f"Harmful {cfg['harmful']['dataset']}: {len(train_H)} train (max-capped), {splits_H.num_classes} classes")

    lower, upper, primary_head = get_resnet18_split()
    lower = lower.to(device).eval()
    upper = upper.to(device).eval()
    primary_head = primary_head.to(device)
    baseline_extractor = get_resnet18_extractor().to(device).eval()

    defender_params = list(upper.parameters()) + list(primary_head.parameters())
    optim_defender = torch.optim.SGD(
        defender_params,
        lr=float(cfg["lr"]),
        momentum=float(cfg["momentum"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    num_attackers = int(cfg.get("num_attackers", 2))
    rank = int(cfg.get("rank", 8))
    directions_per_attacker = int(cfg.get("directions_per_attacker", 3))
    epsilon = float(cfg.get("epsilon", 1e-2))
    attackers = init_population(
        upper,
        loader_H_att,
        num_attackers=num_attackers,
        seed=int(cfg["seed"]),
        rank=rank,
        directions_per_attacker=directions_per_attacker,
        num_classes=splits_H.num_classes,
        device=device,
        cfg=cfg,
    )

    lambda_tangent = float(cfg.get("lambda_tangent", 0.2))
    lambda_well = float(cfg.get("lambda_well", 1.0))
    lambda_ill = float(cfg.get("lambda_ill", 1.0))
    ce_threshold_cfg = cfg.get("ce_threshold")
    ce_threshold = math.log(splits_H.num_classes) if ce_threshold_cfg is None else float(ce_threshold_cfg)
    attackers_per_outer = int(cfg.get("attackers_per_outer", 1))
    attacker_steps = int(cfg.get("attacker_steps_per_selected", 1))
    attacker_grad_clip = float(cfg.get("attacker_grad_clip", 5.0))
    defender_grad_clip = float(cfg.get("grad_clip", 1.0))
    use_k_inv = bool(cfg.get("use_k_inv_preconditioner", True))
    k_inv_ridge = float(cfg.get("k_inv_ridge", 1e-2))

    print(f"[v12] tangent label removal: num_attackers={num_attackers}, "
          f"directions_per_attacker={directions_per_attacker}, rank={rank}, epsilon={epsilon}, "
          f"attackers_per_outer={attackers_per_outer}, attacker_steps={attacker_steps}, "
          f"lambda_tangent={lambda_tangent}, ce_threshold={ce_threshold:.4f}")
    if use_k_inv:
        print(f"[k_inv] dummy-layer preconditioner enabled, ridge={k_inv_ridge}")

    iter_H_def = cycle(loader_H_def)
    history = []
    iters = int(cfg["iterations"])
    log_every = int(cfg["log_every"])
    eval_every = int(cfg.get("eval_every", 0))
    pbar = tqdm(total=iters, desc="immunize_v12")

    step = 0
    for x_P, y_P in cycle(loader_P):
        if step >= iters:
            break

        upper.eval()
        for offset in range(attackers_per_outer):
            idx = (step * attackers_per_outer + offset) % num_attackers
            update_attacker(
                attackers[idx],
                lower,
                upper,
                device,
                epsilon=epsilon,
                steps=attacker_steps,
                grad_clip=attacker_grad_clip,
            )

        x_H, y_H = next(iter_H_def)
        x_P = x_P.to(device, non_blocking=True)
        y_P = y_P.to(device, non_blocking=True)
        x_H = x_H.to(device, non_blocking=True)
        y_H = y_H.to(device, non_blocking=True)

        with torch.no_grad():
            z_P = lower(x_P)
            z_H = lower(x_H)

        upper.eval()
        feat_P = upper(z_P)
        feat_H = upper(z_H)
        L_primary = F.cross_entropy(primary_head(feat_P), y_P)

        feat_P_for_reg = k_inv_dummy_layer(feat_P, ridge=k_inv_ridge) if use_k_inv else feat_P
        feat_H_for_reg = k_inv_dummy_layer(feat_H, ridge=k_inv_ridge) if use_k_inv else feat_H
        H_P = feature_hessian(feat_P_for_reg)
        H_H = feature_hessian(feat_H_for_reg)
        L_well = r_well(H_P)
        L_ill = r_ill(H_H)

        tangent_losses = []
        tangent_accs = []
        for attacker in attackers:
            ce, acc = defender_tangent_ce(attacker, upper, z_H, y_H, epsilon=epsilon)
            tangent_losses.append(ce)
            tangent_accs.append(acc)

        loss_stack = torch.stack(tangent_losses)
        min_tangent_loss, min_idx_t = torch.min(loss_stack, dim=0)
        min_idx = int(min_idx_t.detach().item())
        L_tangent = F.softplus(torch.as_tensor(ce_threshold, device=device, dtype=feat_P.dtype) - min_tangent_loss)

        loss = L_primary + lambda_well * L_well + lambda_ill * L_ill + lambda_tangent * L_tangent

        optim_defender.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(defender_params, defender_grad_clip)
        optim_defender.step()

        if step % log_every == 0:
            tangent_ce_all = [float(value.detach().item()) for value in tangent_losses]
            tangent_norms = [
                [float(global_b_norm(B).detach().item()) for B in attacker.B_list]
                for attacker in attackers
            ]
            postfix = {
                "primary": f"{L_primary.item():.3f}",
                "tangent": f"{L_tangent.item():.3f}",
                "min_ce": f"{min_tangent_loss.item():.3f}",
                "min_j": min_idx,
                "acc": f"{tangent_accs[min_idx]:.3f}",
            }
            pbar.set_postfix(**postfix)
            history.append({
                "step": step,
                "loss_primary": float(L_primary.item()),
                "loss_well": float(L_well.item()),
                "loss_ill": float(L_ill.item()),
                "loss_tangent": float(L_tangent.item()),
                "loss_tangent_min": float(min_tangent_loss.item()),
                "loss_tangent_all": tangent_ce_all,
                "tangent_acc_min_idx": float(tangent_accs[min_idx]),
                "tangent_acc_all": [float(value) for value in tangent_accs],
                "min_attacker_index": min_idx,
                "attacker_updates": [int(attacker.updates) for attacker in attackers],
                "attacker_last_loss": [float(attacker.last_loss) for attacker in attackers],
                "attacker_last_acc": [float(attacker.last_acc) for attacker in attackers],
                "attacker_b_norms": tangent_norms,
                "kappa_H_batch": condition_number(H_H.detach()),
                "kappa_P_batch": condition_number(H_P.detach()),
            })

        if eval_every > 0 and step > 0 and step % eval_every == 0:
            print(f"\n[step {step}] running RIR eval (sampling {cfg['rir_num_groups']} x {cfg['rir_group_size']})")
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
        "_attacker_population_state": [
            {
                "index": attacker.index,
                "A_list": [
                    {name: value.detach().cpu() for name, value in A.items()}
                    for A in attacker.A_list
                ],
                "B_list": [
                    {name: value.detach().cpu() for name, value in B.items()}
                    for B in attacker.B_list
                ],
                "omega": attacker.omega.detach().cpu(),
                "bias": attacker.bias.detach().cpu(),
                "updates": attacker.updates,
            }
            for attacker in attackers
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/immunize_v12a_tangent_removal.yaml")
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

    out = train_v12_tangent_removal(cfg)

    extractor_path = results_dir / "extractor.pt"
    torch.save({"lower": out.pop("_lower_state"), "upper": out.pop("_upper_state")}, extractor_path)
    print(f"Saved immunized extractor -> {extractor_path}")

    attackers_path = results_dir / "tangent_attackers.pt"
    torch.save(out.pop("_attacker_population_state"), attackers_path)
    print(f"Saved tangent attacker population -> {attackers_path}")

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
