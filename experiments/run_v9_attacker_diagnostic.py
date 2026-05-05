"""Diagnose Stage 9 persistent-attacker transfer.

v9a trained against one persistent B-only LoRA attacker state. The post-hoc
probe matrix showed fresh attackers still adapt normally. This diagnostic
checks whether the saved live attacker itself was weak, and whether fresh
B/head restarts recover under either the saved A basis or new random A bases.

Run:
    python experiments/run_v9_attacker_diagnostic.py \
      --extractor-checkpoint results/trap_v9a_persistent_bonly_resnet18_cars/extractor.pt \
      --attacker-checkpoint results/trap_v9a_persistent_bonly_resnet18_cars/persistent_attacker.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_dataset_by_name
from src.models import get_resnet18_split
from src.provenance import capture_provenance
from src.robust_v7 import forward_with_lora_factored, init_lora_factors
from src.utils import get_device, set_seed


def to_device_tree(tree: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone().to(device=device, dtype=torch.float32) for k, v in tree.items()}


def load_split_extractor(path: str, device: torch.device) -> tuple[nn.Module, nn.Module]:
    lower, upper, _ = get_resnet18_split()
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    lower.load_state_dict(ckpt["lower"])
    upper.load_state_dict(ckpt["upper"])
    lower.to(device)
    upper.to(device)
    return lower, upper


def load_attacker_state(path: str, device: torch.device) -> tuple[dict, dict, torch.Tensor, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    A = to_device_tree(ckpt["A"], device)
    B = to_device_tree(ckpt["B"], device)
    omega = ckpt["omega"].detach().clone().to(device=device, dtype=torch.float32)
    bias = ckpt["bias"].detach().clone().to(device=device, dtype=torch.float32)
    return A, B, omega, bias


def init_head(num_classes: int, feature_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    omega = torch.empty(num_classes, feature_dim, device=device)
    nn.init.kaiming_normal_(omega, mode="fan_out", nonlinearity="linear")
    omega.requires_grad_(True)
    bias = torch.zeros(num_classes, device=device, requires_grad=True)
    return omega, bias


def evaluate_bonly(
    lower: nn.Module,
    upper: nn.Module,
    A: dict[str, torch.Tensor],
    B: dict[str, torch.Tensor],
    omega: torch.Tensor,
    bias: torch.Tensor,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    lower.eval()
    upper.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z = lower(x)
            feat = forward_with_lora_factored(z, upper, A, B)
            logits = feat @ omega.T + bias
            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
            correct += logits.argmax(dim=-1).eq(y).sum().item()
            total += y.numel()
    return {
        "acc": correct / max(total, 1),
        "ce": loss_sum / max(total, 1),
        "n": total,
    }


def make_trainable_b_like(B_template: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros_like(B, requires_grad=True)
        for name, B in B_template.items()
    }


def clone_fixed_a(A_template: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {name: A.detach().clone() for name, A in A_template.items()}
    for A in out.values():
        A.requires_grad_(False)
    return out


def train_restart(
    *,
    variant: str,
    seed: int,
    extractor_checkpoint: str,
    saved_A: dict[str, torch.Tensor],
    saved_B: dict[str, torch.Tensor],
    num_classes: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    rank: int,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    grad_clip: float,
) -> dict:
    set_seed(seed)
    lower, upper = load_split_extractor(extractor_checkpoint, device)
    for p in lower.parameters():
        p.requires_grad = False
    for p in upper.parameters():
        p.requires_grad = False

    if variant == "saved_A_reset_B_head":
        A = clone_fixed_a(saved_A)
        B = make_trainable_b_like(saved_B)
    elif variant == "fresh_A_reset_B_head":
        A, B = init_lora_factors(upper, rank=rank, device=device, dtype=torch.float32)
        for A_t in A.values():
            A_t.requires_grad_(False)
    else:
        raise ValueError(f"unknown variant: {variant}")

    omega, bias = init_head(num_classes, 512, device)
    params = list(B.values()) + [omega, bias]
    optim = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    epoch_accs: list[float] = []
    epoch_ces: list[float] = []
    for epoch in range(1, epochs + 1):
        lower.train()
        upper.train()
        running = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"{variant} seed={seed} epoch {epoch:02d}", leave=False)
        for x, y in progress:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                z = lower(x)
            feat = forward_with_lora_factored(z, upper, A, B)
            logits = feat @ omega.T + bias
            loss = F.cross_entropy(logits, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optim.step()

            running += loss.item() * x.size(0)
            seen += x.size(0)

        metrics = evaluate_bonly(lower, upper, A, B, omega, bias, test_loader, device)
        epoch_accs.append(metrics["acc"])
        epoch_ces.append(metrics["ce"])
        print(
            f"{variant} seed={seed} epoch {epoch:02d} | "
            f"train_loss={running / max(seen, 1):.4f} | "
            f"test_acc={metrics['acc'] * 100:.2f}% | test_ce={metrics['ce']:.4f}"
        )

    return {
        "variant": variant,
        "seed": seed,
        "epoch_accs": epoch_accs,
        "epoch_ces": epoch_ces,
        "final_acc": epoch_accs[-1],
        "best_acc": max(epoch_accs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor-checkpoint", default="results/trap_v9a_persistent_bonly_resnet18_cars/extractor.pt")
    parser.add_argument("--attacker-checkpoint", default="results/trap_v9a_persistent_bonly_resnet18_cars/persistent_attacker.pt")
    parser.add_argument("--dataset", default="cars")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--restart-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--run-name", default="diagnose_v9a_persistent_attacker")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    splits = load_dataset_by_name(args.dataset, root=args.data_root, image_size=args.image_size)
    persist = args.num_workers > 0
    train_loader = DataLoader(
        splits.train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persist,
    )
    eval_train_loader = DataLoader(
        splits.train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persist,
    )
    test_loader = DataLoader(
        splits.test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persist,
    )
    print(f"Loaded {args.dataset}: {len(splits.train)} train, {len(splits.test)} test, {splits.num_classes} classes")

    lower, upper = load_split_extractor(args.extractor_checkpoint, device)
    saved_A, saved_B, saved_omega, saved_bias = load_attacker_state(args.attacker_checkpoint, device)

    saved_train = evaluate_bonly(lower, upper, saved_A, saved_B, saved_omega, saved_bias, eval_train_loader, device)
    saved_test = evaluate_bonly(lower, upper, saved_A, saved_B, saved_omega, saved_bias, test_loader, device)
    print(
        f"saved persistent attacker | train_acc={saved_train['acc'] * 100:.2f}% "
        f"train_ce={saved_train['ce']:.4f} | test_acc={saved_test['acc'] * 100:.2f}% "
        f"test_ce={saved_test['ce']:.4f}"
    )

    restarts = []
    for variant in ["saved_A_reset_B_head", "fresh_A_reset_B_head"]:
        for seed in args.restart_seeds:
            restarts.append(train_restart(
                variant=variant,
                seed=seed,
                extractor_checkpoint=args.extractor_checkpoint,
                saved_A=saved_A,
                saved_B=saved_B,
                num_classes=splits.num_classes,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                rank=args.rank,
                epochs=args.epochs,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                grad_clip=args.grad_clip,
            ))

    out = {
        "config": vars(args),
        "provenance": capture_provenance(),
        "saved_persistent_attacker": {
            "train": saved_train,
            "test": saved_test,
        },
        "restarts": restarts,
    }
    out_dir = Path(args.results_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
