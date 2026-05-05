"""Stage 11 diagnostic - learned LoRA-tangent label leakage.

Random tangent sketches were too weak: arbitrary LoRA directions did not predict
post-hoc LoRA success. This script learns label-informed B-only LoRA directions,
then shrinks them back to a small first-order perturbation and asks whether
Cars labels are linearly recoverable from:

    base features
    learned tangent features
    base + learned tangent features

If learned tangent features are predictive even when base features are not,
then harmful label information remains in adapter-reachable directions.

Run example:
    python experiments/run_learned_tangent_probe.py \
      --dataset cars \
      --extractor-checkpoint results/trap_v10a_population_bonly_resnet18_cars/extractor.pt \
      --run-name learned_tangent_v10a_r8_k3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from types import SimpleNamespace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_dataset_by_name
from src.models import get_resnet18_split
from src.provenance import capture_provenance
from src.robust_v7 import forward_with_lora_factored, init_lora_factors
from src.utils import get_device, set_seed


def load_split(checkpoint: str | None, device: torch.device) -> tuple[nn.Module, nn.Module]:
    lower, upper, _ = get_resnet18_split()
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        lower.load_state_dict(ckpt["lower"])
        upper.load_state_dict(ckpt["upper"])
        print(f"Loaded extractor checkpoint from {checkpoint}")
    else:
        print("Using ImageNet-pretrained ResNet18 (no immunization)")

    lower.to(device).eval()
    upper.to(device).eval()
    for p in lower.parameters():
        p.requires_grad = False
    for p in upper.parameters():
        p.requires_grad = False
    return lower, upper


def init_head(num_classes: int, feature_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    omega = torch.empty(num_classes, feature_dim, device=device)
    nn.init.kaiming_normal_(omega, mode="fan_out", nonlinearity="linear")
    omega.requires_grad_(True)
    bias = torch.zeros(num_classes, device=device, requires_grad=True)
    return omega, bias


def b_params(B: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    return list(B.values())


def freeze_a(A: dict[str, torch.Tensor]) -> None:
    for value in A.values():
        value.requires_grad_(False)


def clone_detached_tree(tree: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in tree.items()}


def global_b_norm(B: dict[str, torch.Tensor]) -> torch.Tensor:
    norms = [v.detach().pow(2).sum() for v in B.values()]
    if not norms:
        raise ValueError("empty B dictionary")
    return torch.stack(norms).sum().sqrt()


def scaled_b_direction(B: dict[str, torch.Tensor], epsilon: float) -> dict[str, torch.Tensor]:
    norm = global_b_norm(B).clamp_min(1e-12)
    return {k: (v.detach() / norm) * epsilon for k, v in B.items()}


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


def learn_b_direction(
    *,
    seed: int,
    lower: nn.Module,
    upper: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    rank: int,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    grad_clip: float,
    device: torch.device,
) -> dict:
    set_seed(seed)
    A, B = init_lora_factors(upper, rank=rank, device=device, dtype=torch.float32)
    freeze_a(A)
    omega, bias = init_head(num_classes, 512, device)
    params = b_params(B) + [omega, bias]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    epoch_accs: list[float] = []
    epoch_ces: list[float] = []
    for epoch in range(1, epochs + 1):
        running = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"learn_dir seed={seed} epoch {epoch:02d}", leave=False)
        for x, y in progress:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                z = lower(x)
            feat = forward_with_lora_factored(z, upper, A, B)
            logits = feat @ omega.T + bias
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()

            running += loss.item() * x.size(0)
            seen += x.size(0)

        metrics = evaluate_bonly(lower, upper, A, B, omega, bias, test_loader, device)
        epoch_accs.append(metrics["acc"])
        epoch_ces.append(metrics["ce"])
        print(
            f"learn_dir seed={seed} epoch {epoch:02d} | "
            f"train_loss={running / max(seen, 1):.4f} | "
            f"test_acc={metrics['acc'] * 100:.2f}% | test_ce={metrics['ce']:.4f}"
        )

    return {
        "seed": seed,
        "A": clone_detached_tree(A),
        "B": clone_detached_tree(B),
        "omega": omega.detach().clone(),
        "bias": bias.detach().clone(),
        "epoch_accs": epoch_accs,
        "epoch_ces": epoch_ces,
        "final_acc": epoch_accs[-1],
        "best_acc": max(epoch_accs),
        "b_global_norm": float(global_b_norm(B).item()),
    }


@torch.no_grad()
def extract_base_and_tangent(
    lower: nn.Module,
    upper: nn.Module,
    directions: list[dict],
    loader: DataLoader,
    *,
    epsilon: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_chunks: list[torch.Tensor] = []
    tangent_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []

    scaled_dirs = [
        (direction["A"], scaled_b_direction(direction["B"], epsilon))
        for direction in directions
    ]

    for x, y in tqdm(loader, desc="extract learned tangent", leave=False):
        x = x.to(device, non_blocking=True)
        z = lower(x)
        base = upper(z)
        pieces = []
        for A, B_scaled in scaled_dirs:
            feat_eps = forward_with_lora_factored(z, upper, A, B_scaled)
            pieces.append((feat_eps - base) / epsilon)
        base_chunks.append(base.cpu())
        tangent_chunks.append(torch.cat(pieces, dim=1).cpu())
        label_chunks.append(y.cpu())

    return (
        torch.cat(base_chunks, dim=0).float(),
        torch.cat(tangent_chunks, dim=0).float(),
        torch.cat(label_chunks, dim=0).long(),
    )


def standardize(train_x: torch.Tensor, test_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_x - mean) / std, (test_x - mean) / std, {
        "mean_abs": float(mean.abs().mean()),
        "std_mean": float(std.mean()),
        "std_min": float(std.min()),
        "std_max": float(std.max()),
    }


def train_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> dict:
    train_x, test_x, stats = standardize(train_x, test_x)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

    head = nn.Linear(train_x.shape[1], num_classes).to(device)
    nn.init.kaiming_normal_(head.weight, mode="fan_out", nonlinearity="linear")
    nn.init.zeros_(head.bias)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    test_x = test_x.to(device)
    test_y = test_y.to(device)
    epoch_accs: list[float] = []
    epoch_losses: list[float] = []
    for epoch in range(1, epochs + 1):
        head.train()
        running = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            loss = F.cross_entropy(head(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            seen += xb.size(0)

        head.eval()
        correct = 0
        total = 0
        for start in range(0, test_x.shape[0], batch_size):
            logits = head(test_x[start:start + batch_size])
            yb = test_y[start:start + batch_size]
            correct += logits.argmax(dim=-1).eq(yb).sum().item()
            total += yb.numel()
        acc = correct / max(total, 1)
        epoch_accs.append(acc)
        epoch_losses.append(running / max(seen, 1))
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(f"  probe epoch {epoch:03d} | loss={epoch_losses[-1]:.4f} | test_acc={acc * 100:.2f}%")

    return {
        "epoch_accs": epoch_accs,
        "epoch_losses": epoch_losses,
        "final_acc": epoch_accs[-1],
        "best_acc": max(epoch_accs),
        "feature_dim": int(train_x.shape[1]),
        "standardization": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--extractor-checkpoint", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--num-directions", type=int, default=3)
    parser.add_argument("--direction-epochs", type=int, default=10)
    parser.add_argument("--direction-lr", type=float, default=0.01)
    parser.add_argument("--direction-momentum", type=float, default=0.9)
    parser.add_argument("--direction-weight-decay", type=float, default=0.0)
    parser.add_argument("--direction-grad-clip", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-limit", type=int, default=0, help="Optional train subset size for smoke tests")
    parser.add_argument("--test-limit", type=int, default=0, help="Optional test subset size for smoke tests")
    parser.add_argument("--synthetic-smoke", action="store_true", help="Use random image tensors instead of loading a dataset")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="./results")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.dataset is not None:
        cfg["dataset"] = args.dataset

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    if args.synthetic_smoke:
        train_n = args.train_limit if args.train_limit > 0 else 32
        test_n = args.test_limit if args.test_limit > 0 else 32
        num_classes = 196
        image_size = int(cfg["image_size"])
        train_x = torch.randn(train_n, 3, image_size, image_size)
        test_x = torch.randn(test_n, 3, image_size, image_size)
        train_y = torch.arange(train_n).remainder(num_classes).long()
        test_y = torch.arange(test_n).remainder(num_classes).long()
        splits = SimpleNamespace(
            train=TensorDataset(train_x, train_y),
            test=TensorDataset(test_x, test_y),
            num_classes=num_classes,
        )
        print(f"Using synthetic smoke data: {train_n} train, {test_n} test")
    else:
        splits = load_dataset_by_name(cfg["dataset"], root=cfg["data_root"], image_size=cfg["image_size"])
        if args.train_limit > 0:
            splits.train = Subset(splits.train, range(min(args.train_limit, len(splits.train))))
        if args.test_limit > 0:
            splits.test = Subset(splits.test, range(min(args.test_limit, len(splits.test))))
    num_workers = cfg["probe"]["num_workers"] if args.num_workers is None else args.num_workers
    persist = num_workers > 0
    train_loader = DataLoader(
        splits.train,
        batch_size=args.feature_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persist,
    )
    train_extract_loader = DataLoader(
        splits.train,
        batch_size=args.feature_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persist,
    )
    test_loader = DataLoader(
        splits.test,
        batch_size=args.feature_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persist,
    )
    print(f"Loaded {cfg['dataset']}: {len(splits.train)} train, {len(splits.test)} test, {splits.num_classes} classes")

    lower, upper = load_split(args.extractor_checkpoint, device)
    learned_dirs = []
    for idx in range(args.num_directions):
        seed = args.seed + idx
        learned_dirs.append(learn_b_direction(
            seed=seed,
            lower=lower,
            upper=upper,
            train_loader=train_loader,
            test_loader=test_loader,
            num_classes=splits.num_classes,
            rank=args.rank,
            epochs=args.direction_epochs,
            lr=args.direction_lr,
            momentum=args.direction_momentum,
            weight_decay=args.direction_weight_decay,
            grad_clip=args.direction_grad_clip,
            device=device,
        ))

    train_base, train_tangent, train_y = extract_base_and_tangent(
        lower, upper, learned_dirs, train_extract_loader, epsilon=args.epsilon, device=device
    )
    test_base, test_tangent, test_y = extract_base_and_tangent(
        lower, upper, learned_dirs, test_loader, epsilon=args.epsilon, device=device
    )

    feature_stats = {
        "base_dim": int(train_base.shape[1]),
        "tangent_dim": int(train_tangent.shape[1]),
        "base_train_norm_mean": float(train_base.norm(dim=1).mean()),
        "tangent_train_norm_mean": float(train_tangent.norm(dim=1).mean()),
        "epsilon": args.epsilon,
    }
    print(f"Feature stats: {feature_stats}")

    probes = {
        "base": (train_base, test_base),
        "learned_tangent": (train_tangent, test_tangent),
        "base_learned_tangent": (
            torch.cat([train_base, train_tangent], dim=1),
            torch.cat([test_base, test_tangent], dim=1),
        ),
    }
    probe_results = {}
    for name, (train_x, test_x) in probes.items():
        print(f"\nTraining {name} probe | dim={train_x.shape[1]}")
        probe_results[name] = train_probe(
            train_x,
            train_y,
            test_x,
            test_y,
            num_classes=splits.num_classes,
            epochs=args.probe_epochs,
            batch_size=args.probe_batch_size,
            lr=args.probe_lr,
            weight_decay=args.probe_weight_decay,
            device=device,
        )

    run_name = args.run_name or f"learned_tangent_{cfg['dataset']}_r{args.rank}_k{args.num_directions}"
    out_dir = Path(args.results_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "config": cfg,
        "args": vars(args),
        "feature_stats": feature_stats,
        "learned_directions": [
            {
                "seed": d["seed"],
                "epoch_accs": d["epoch_accs"],
                "epoch_ces": d["epoch_ces"],
                "final_acc": d["final_acc"],
                "best_acc": d["best_acc"],
                "b_global_norm": d["b_global_norm"],
            }
            for d in learned_dirs
        ],
        "probes": probe_results,
        "provenance": capture_provenance(),
    }
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\nSummary")
    for name, result in probe_results.items():
        print(f"{name:22s} final={result['final_acc'] * 100:.2f}% best={result['best_acc'] * 100:.2f}%")
    print(f"Saved results -> {out_path}")


if __name__ == "__main__":
    main()
