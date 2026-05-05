"""LoRA tangent-accessibility diagnostic.

The linear-probe trap can make harmful labels hard to recover from frozen
features z=f_theta(x). LoRA may still succeed if harmful labels are recoverable
from adapter tangent directions around the released model.

This script estimates that accessibility with a random finite-difference sketch.
For each random LoRA-B direction v_m at standard LoRA init (A random, B=0), it
computes

    psi_m(x) = (f_{theta, B=eps*v_m}(x) - f_{theta, B=0}(x)) / eps

Then it trains identical linear classifiers on:

    base          = z
    tangent       = [psi_1, ..., psi_M]
    base_tangent  = [z, psi_1, ..., psi_M]

If `base` is weak but `tangent` or `base_tangent` is strong, the released model
still exposes harmful information through LoRA-reachable first-order directions.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_dataset_by_name
from src.lora import LoRAConv2d, lorafy
from src.models import get_resnet18_full_extractor_from_split, get_resnet18_split
from src.provenance import capture_provenance
from src.utils import get_device, set_seed


def _maybe_subset(dataset: Dataset, max_items: int | None) -> Dataset:
    if max_items is None or max_items <= 0 or max_items >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_items)))


def _load_lorafied_extractor(checkpoint: str | None, rank: int, device: torch.device) -> tuple[nn.Module, list[LoRAConv2d]]:
    lower, upper, _ = get_resnet18_split()
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        lower.load_state_dict(ckpt["lower"])
        upper.load_state_dict(ckpt["upper"])
        print(f"Loaded extractor checkpoint from {checkpoint}")
    else:
        print("Using ImageNet-pretrained ResNet18 (no immunization)")

    for p in lower.parameters():
        p.requires_grad = False
    for p in upper.parameters():
        p.requires_grad = False

    n_wrapped = lorafy(upper, rank=rank)
    modules = [m for m in upper.modules() if isinstance(m, LoRAConv2d)]
    if n_wrapped != len(modules):
        raise RuntimeError(f"lorafy count mismatch: returned {n_wrapped}, found {len(modules)}")

    for module in modules:
        for p in module.parameters():
            p.requires_grad = False

    extractor = get_resnet18_full_extractor_from_split(lower, upper).to(device)
    extractor.eval()
    print(f"LoRA tangent modules wrapped: {len(modules)}")
    return extractor, modules


def _zero_lora_b(modules: Iterable[LoRAConv2d]) -> None:
    with torch.no_grad():
        for module in modules:
            module.lora_B.weight.zero_()


def _make_b_directions(
    modules: list[LoRAConv2d],
    num_directions: int,
    *,
    device: torch.device,
) -> list[list[torch.Tensor]]:
    directions: list[list[torch.Tensor]] = []
    for _ in range(num_directions):
        one_direction: list[torch.Tensor] = []
        for module in modules:
            d = torch.randn_like(module.lora_B.weight, device=device)
            rank = max(module.lora_B.weight.shape[1], 1)
            d = d / math.sqrt(rank)
            one_direction.append(d)
        directions.append(one_direction)
    return directions


def _set_lora_b_direction(modules: list[LoRAConv2d], direction: list[torch.Tensor], epsilon: float) -> None:
    with torch.no_grad():
        for module, d in zip(modules, direction):
            module.lora_B.weight.copy_(epsilon * d)


@torch.no_grad()
def extract_tangent_features(
    extractor: nn.Module,
    modules: list[LoRAConv2d],
    loader: DataLoader,
    directions: list[list[torch.Tensor]],
    *,
    epsilon: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_chunks: list[torch.Tensor] = []
    tangent_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []

    for x, y in tqdm(loader, desc="extract", leave=False):
        x = x.to(device, non_blocking=True)
        _zero_lora_b(modules)
        z0 = extractor(x)
        tangent_parts: list[torch.Tensor] = []
        for direction in directions:
            _set_lora_b_direction(modules, direction, epsilon)
            z_eps = extractor(x)
            tangent_parts.append((z_eps - z0) / epsilon)
        _zero_lora_b(modules)

        base_chunks.append(z0.cpu())
        tangent_chunks.append(torch.cat(tangent_parts, dim=1).cpu())
        label_chunks.append(y.cpu())

    return (
        torch.cat(base_chunks, dim=0).float(),
        torch.cat(tangent_chunks, dim=0).float(),
        torch.cat(label_chunks, dim=0).long(),
    )


def _standardize(train_x: torch.Tensor, test_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_z = (train_x - mean) / std
    test_z = (test_x - mean) / std
    return train_z, test_z, {
        "mean_abs": float(mean.abs().mean()),
        "std_mean": float(std.mean()),
        "std_min": float(std.min()),
        "std_max": float(std.max()),
    }


def train_linear_classifier(
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
    train_x, test_x, stats = _standardize(train_x, test_x)
    train_ds = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

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
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = head(xb)
            loss = F.cross_entropy(logits, yb)
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
            pred = logits.argmax(dim=-1)
            yb = test_y[start:start + batch_size]
            correct += (pred == yb).sum().item()
            total += yb.numel()
        acc = correct / total
        epoch_accs.append(acc)
        epoch_losses.append(running / max(seen, 1))
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(f"  epoch {epoch:03d} | loss={epoch_losses[-1]:.4f} | test_acc={acc * 100:.2f}%")

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
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--extractor-checkpoint", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--num-directions", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.dataset is not None:
        cfg["dataset"] = args.dataset

    set_seed(int(cfg["seed"]))
    device = get_device()
    print(f"Device: {device}")

    splits = load_dataset_by_name(cfg["dataset"], root=cfg["data_root"], image_size=cfg["image_size"])
    train_ds = _maybe_subset(splits.train, args.max_train)
    test_ds = _maybe_subset(splits.test, args.max_test)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.feature_batch_size,
        shuffle=False,
        num_workers=cfg["probe"]["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.feature_batch_size,
        shuffle=False,
        num_workers=cfg["probe"]["num_workers"],
        pin_memory=True,
    )
    print(f"Loaded {cfg['dataset']}: {len(train_ds)} train, {len(test_ds)} test, {splits.num_classes} classes")

    extractor, modules = _load_lorafied_extractor(args.extractor_checkpoint, args.lora_rank, device)
    directions = _make_b_directions(modules, args.num_directions, device=device)

    train_base, train_tangent, train_y = extract_tangent_features(
        extractor, modules, train_loader, directions, epsilon=args.epsilon, device=device
    )
    test_base, test_tangent, test_y = extract_tangent_features(
        extractor, modules, test_loader, directions, epsilon=args.epsilon, device=device
    )

    feature_stats = {
        "base_dim": int(train_base.shape[1]),
        "tangent_dim": int(train_tangent.shape[1]),
        "base_train_norm_mean": float(train_base.norm(dim=1).mean()),
        "tangent_train_norm_mean": float(train_tangent.norm(dim=1).mean()),
    }
    print(f"Feature stats: {feature_stats}")

    probes = {
        "base": (train_base, test_base),
        "tangent": (train_tangent, test_tangent),
        "base_tangent": (torch.cat([train_base, train_tangent], dim=1), torch.cat([test_base, test_tangent], dim=1)),
    }

    results: dict[str, dict] = {}
    for name, (xtr, xte) in probes.items():
        print(f"\nTraining {name} probe | dim={xtr.shape[1]}")
        results[name] = train_linear_classifier(
            xtr,
            train_y,
            xte,
            test_y,
            num_classes=splits.num_classes,
            epochs=args.probe_epochs,
            batch_size=args.probe_batch_size,
            lr=args.probe_lr,
            weight_decay=args.probe_weight_decay,
            device=device,
        )

    run_name = args.run_name or f"tangent_access_{cfg['dataset']}_r{args.lora_rank}_m{args.num_directions}"
    results_dir = Path(cfg["results_dir"]) / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "config": cfg,
        "args": vars(args),
        "feature_stats": feature_stats,
        "probes": results,
        "provenance": capture_provenance(),
    }
    out_path = results_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\nSummary")
    for name, result in results.items():
        print(f"{name:12s} final={result['final_acc'] * 100:.2f}% best={result['best_acc'] * 100:.2f}%")
    print(f"Saved results -> {out_path}")


if __name__ == "__main__":
    main()
