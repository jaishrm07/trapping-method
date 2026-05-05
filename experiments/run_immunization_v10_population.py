"""Stage 10 — population persistent B-only LoRA immunization.

v9a kept one B-only LoRA attacker alive across defender steps. It made that
single live attacker's CE high, but fresh post-hoc attackers still adapted.

v10 tests the next hypothesis: the defender overfit one attacker state. Keep a
small population of persistent B-only attackers with independent random A bases,
heads, and minibatch histories. Update a subset each outer step, then block the
strongest current attacker:

    L_block = softplus(log(C_H) - min_j CE_H(theta, attacker_j))

Run:
    python experiments/run_immunization_v10_population.py \
        --config configs/immunize_v10a_population_bonly.yaml
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
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_immunization_v9_persistent import (
    attacker_minibatch_step,
    attacker_params,
    evaluate_imagenet_top1,
    feature_hessian,
    harmful_logits_bonly,
    init_persistent_bonly_attacker,
    make_attacker_optimizer,
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
from src.utils import get_device, set_seed


@dataclass
class PersistentBOnlyAttacker:
    index: int
    A: dict[str, torch.Tensor]
    B: dict[str, torch.Tensor]
    omega: torch.Tensor
    bias: torch.Tensor
    optimizer: torch.optim.Optimizer
    train_iter: object
    last_loss: float = 0.0
    last_acc: float = 0.0
    updates: int = 0


def init_population(
    upper: torch.nn.Module,
    loader_H: DataLoader,
    *,
    num_attackers: int,
    seed: int,
    rank: int,
    num_classes: int,
    feature_dim: int,
    device: torch.device,
    cfg: dict,
) -> list[PersistentBOnlyAttacker]:
    attackers: list[PersistentBOnlyAttacker] = []
    for idx in range(num_attackers):
        torch.manual_seed(seed + 1009 * idx)
        A, B, omega, bias = init_persistent_bonly_attacker(
            upper,
            rank=rank,
            num_classes=num_classes,
            feature_dim=feature_dim,
            device=device,
            dtype=torch.float32,
        )
        params = attacker_params(B, omega, bias)
        opt = make_attacker_optimizer(params, cfg)
        attackers.append(PersistentBOnlyAttacker(
            index=idx,
            A=A,
            B=B,
            omega=omega,
            bias=bias,
            optimizer=opt,
            train_iter=cycle(loader_H),
        ))
    return attackers


def update_attacker(
    attacker: PersistentBOnlyAttacker,
    lower: torch.nn.Module,
    upper: torch.nn.Module,
    device: torch.device,
    *,
    steps: int,
    grad_clip: float,
) -> None:
    for _ in range(steps):
        x_h, y_h = next(attacker.train_iter)
        x_h = x_h.to(device, non_blocking=True)
        y_h = y_h.to(device, non_blocking=True)
        loss, acc = attacker_minibatch_step(
            lower,
            upper,
            x_h,
            y_h,
            attacker.A,
            attacker.B,
            attacker.omega,
            attacker.bias,
            attacker.optimizer,
            grad_clip=grad_clip,
        )
        attacker.last_loss = loss
        attacker.last_acc = acc
        attacker.updates += 1


def detached_attacker_view(attacker: PersistentBOnlyAttacker) -> tuple[dict, dict, torch.Tensor, torch.Tensor]:
    A = {name: value.detach() for name, value in attacker.A.items()}
    B = {name: value.detach() for name, value in attacker.B.items()}
    return A, B, attacker.omega.detach(), attacker.bias.detach()


def train_v10_population(cfg: dict) -> dict:
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
    loader_H_def = DataLoader(train_H, batch_size=cfg["batch_size"], shuffle=True, num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=persist)
    loader_H_att = DataLoader(train_H, batch_size=cfg["batch_size"], shuffle=True, num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=persist)
    test_P = DataLoader(splits_P.test, batch_size=cfg["batch_size"], shuffle=False, num_workers=nw, pin_memory=True, persistent_workers=persist)

    print(f"Primary {cfg['primary']['dataset']}: {len(train_P)} train (max-capped)")
    print(f"Harmful {cfg['harmful']['dataset']}: {len(train_H)} train (max-capped), {splits_H.num_classes} classes")

    lower, upper, primary_head = get_resnet18_split()
    lower = lower.to(device)
    upper = upper.to(device)
    primary_head = primary_head.to(device)
    baseline_extractor = get_resnet18_extractor().to(device).eval()

    defender_params = list(upper.parameters()) + list(primary_head.parameters())
    optim_defender = torch.optim.SGD(
        defender_params,
        lr=cfg["lr"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
    )

    num_attackers = int(cfg.get("num_attackers", 4))
    rank = int(cfg.get("rank", 8))
    attackers = init_population(
        upper,
        loader_H_att,
        num_attackers=num_attackers,
        seed=int(cfg["seed"]),
        rank=rank,
        num_classes=splits_H.num_classes,
        feature_dim=512,
        device=device,
        cfg=cfg,
    )

    lambda_block = float(cfg.get("lambda_block", 0.3))
    lambda_well = float(cfg.get("lambda_well", 1.0))
    lambda_ill = float(cfg.get("lambda_ill", 1.0))
    ce_threshold_cfg = cfg.get("ce_threshold")
    ce_threshold = math.log(splits_H.num_classes) if ce_threshold_cfg is None else float(ce_threshold_cfg)
    attackers_per_outer = int(cfg.get("attackers_per_outer", num_attackers))
    attacker_steps = int(cfg.get("attacker_steps_per_selected", 1))
    attacker_grad_clip = float(cfg.get("attacker_grad_clip", 5.0))
    defender_grad_clip = float(cfg.get("grad_clip", 1.0))
    use_k_inv = bool(cfg.get("use_k_inv_preconditioner", True))
    k_inv_ridge = float(cfg.get("k_inv_ridge", 1e-2))

    print(f"[v10] population B-only LoRA: num_attackers={num_attackers}, rank={rank}, "
          f"attackers_per_outer={attackers_per_outer}, attacker_steps={attacker_steps}, "
          f"attacker_opt={cfg.get('attacker_optimizer', 'adamw')}, attacker_lr={cfg.get('attacker_lr', 1e-3)}, "
          f"lambda_block={lambda_block}, ce_threshold={ce_threshold:.4f}")
    if use_k_inv:
        print(f"[k_inv] dummy-layer preconditioner enabled, ridge={k_inv_ridge}")

    iter_H_def = cycle(loader_H_def)
    history = []
    iters = int(cfg["iterations"])
    log_every = int(cfg["log_every"])
    eval_every = int(cfg.get("eval_every", 0))
    pbar = tqdm(total=iters, desc="immunize_v10")

    step = 0
    for x_P, y_P in cycle(loader_P):
        if step >= iters:
            break

        # Round-robin population updates. Each attacker has its own loader
        # cycle, optimizer state, A basis, B, and head.
        for offset in range(attackers_per_outer):
            idx = (step * attackers_per_outer + offset) % num_attackers
            update_attacker(
                attackers[idx],
                lower,
                upper,
                device,
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

        feat_P = upper(z_P)
        feat_H = upper(z_H)
        L_primary = F.cross_entropy(primary_head(feat_P), y_P)

        feat_P_for_reg = k_inv_dummy_layer(feat_P, ridge=k_inv_ridge) if use_k_inv else feat_P
        feat_H_for_reg = k_inv_dummy_layer(feat_H, ridge=k_inv_ridge) if use_k_inv else feat_H
        H_P = feature_hessian(feat_P_for_reg)
        H_H = feature_hessian(feat_H_for_reg)
        L_well = r_well(H_P)
        L_ill = r_ill(H_H)

        harmful_losses = []
        harmful_accs = []
        for attacker in attackers:
            A_det, B_det, omega_det, bias_det = detached_attacker_view(attacker)
            logits = harmful_logits_bonly(upper, z_H, A_det, B_det, omega_det, bias_det)
            ce = F.cross_entropy(logits, y_H)
            harmful_losses.append(ce)
            harmful_accs.append(logits.detach().argmax(dim=-1).eq(y_H).float().mean().item())

        loss_stack = torch.stack(harmful_losses)
        min_harm_loss, min_idx_t = torch.min(loss_stack, dim=0)
        min_idx = int(min_idx_t.detach().item())
        L_block = F.softplus(torch.as_tensor(ce_threshold, device=device, dtype=feat_P.dtype) - min_harm_loss)

        loss = L_primary + lambda_well * L_well + lambda_ill * L_ill + lambda_block * L_block

        optim_defender.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(defender_params, defender_grad_clip)
        optim_defender.step()

        if step % log_every == 0:
            all_ce = [float(v.detach().item()) for v in harmful_losses]
            postfix = {
                "primary": f"{L_primary.item():.3f}",
                "block": f"{L_block.item():.3f}",
                "min_harm": f"{min_harm_loss.item():.3f}",
                "min_j": min_idx,
                "acc": f"{harmful_accs[min_idx]:.3f}",
            }
            pbar.set_postfix(**postfix)
            history.append({
                "step": step,
                "loss_primary": float(L_primary.item()),
                "loss_well": float(L_well.item()),
                "loss_ill": float(L_ill.item()),
                "loss_block": float(L_block.item()),
                "loss_harm_min": float(min_harm_loss.item()),
                "loss_harm_all": all_ce,
                "attacker_acc_min_idx": float(harmful_accs[min_idx]),
                "attacker_acc_all": [float(v) for v in harmful_accs],
                "min_attacker_index": min_idx,
                "attacker_updates": [int(a.updates) for a in attackers],
                "attacker_last_loss": [float(a.last_loss) for a in attackers],
                "attacker_last_acc": [float(a.last_acc) for a in attackers],
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
                "A": {k: v.detach().cpu() for k, v in attacker.A.items()},
                "B": {k: v.detach().cpu() for k, v in attacker.B.items()},
                "omega": attacker.omega.detach().cpu(),
                "bias": attacker.bias.detach().cpu(),
                "updates": attacker.updates,
            }
            for attacker in attackers
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/immunize_v10a_population_bonly.yaml")
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

    out = train_v10_population(cfg)

    extractor_path = results_dir / "extractor.pt"
    torch.save({"lower": out.pop("_lower_state"), "upper": out.pop("_upper_state")}, extractor_path)
    print(f"Saved immunized extractor -> {extractor_path}")

    attackers_path = results_dir / "population_attackers.pt"
    torch.save(out.pop("_attacker_population_state"), attackers_path)
    print(f"Saved attacker population -> {attackers_path}")

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
