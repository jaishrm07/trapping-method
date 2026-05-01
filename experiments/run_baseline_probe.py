"""Stage 0 — baseline linear probe (no immunization).

Reproduces the "Init θ_0" row of Table 1 in the trapping paper:
    ResNet18 + Cars → ~67.04% test accuracy after linear probing.

Run:
    python experiments/run_baseline_probe.py --config configs/default.yaml

This script trains ONLY a linear head on top of a frozen ImageNet-pretrained
ResNet18. No immunization is applied. Per-epoch test accuracy is logged so
later stages can compute RFD against this trajectory as the baseline `M_base`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_dataset_by_name, make_loaders
from src.models import (
    LinearProbeHead,
    build_probe_pipeline,
    freeze_module,
    get_resnet18_full_extractor_from_split,
    get_resnet18_split,
)
from src.utils import get_device, set_seed


def evaluate(extractor: nn.Module, head: nn.Module, loader, device: torch.device) -> float:
    """Top-1 test accuracy."""
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
            total += y.numel()
    return correct / total


def train_probe(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    device = get_device()
    print(f"Device: {device}")

    # Data
    splits = load_dataset_by_name(cfg["dataset"], root=cfg["data_root"], image_size=cfg["image_size"])
    train_loader, test_loader = make_loaders(splits, batch_size=cfg["probe"]["batch_size"], num_workers=cfg["probe"]["num_workers"])
    print(f"Loaded {cfg['dataset']}: {len(splits.train)} train, {len(splits.test)} test, {splits.num_classes} classes")

    # Model — extractor + fresh linear head.
    # If --extractor-checkpoint is given, load the (frozen lower + immunized
    # upper) state from disk; otherwise use the standard ImageNet-pretrained
    # ResNet18. Either way, the extractor is frozen for adversarial probing.
    ckpt_path = cfg.get("extractor_checkpoint")
    if ckpt_path:
        print(f"Loading extractor checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        lower, upper, _ = get_resnet18_split()
        lower.load_state_dict(ckpt["lower"])
        upper.load_state_dict(ckpt["upper"])
        extractor = get_resnet18_full_extractor_from_split(lower, upper)
        freeze_module(extractor)
        head = LinearProbeHead(feature_dim=512, num_classes=splits.num_classes)
    else:
        extractor, head = build_probe_pipeline(num_classes=splits.num_classes)
    extractor = extractor.to(device)
    head = head.to(device)

    # Optimizer — only the head's parameters are trainable
    optimizer = torch.optim.SGD(
        head.parameters(),
        lr=cfg["probe"]["lr"],
        momentum=cfg["probe"]["momentum"],
        weight_decay=cfg["probe"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    # Per-epoch test accuracy trajectory — this becomes M_base^(t) for RFD later.
    epoch_accs: list[float] = []

    for epoch in range(1, cfg["probe"]["epochs"] + 1):
        head.train()
        # Extractor stays in eval() — frozen backbone, no BN drift.
        running = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch:02d}", leave=False)
        for step, (x, y) in enumerate(progress, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.no_grad():
                features = extractor(x)
            logits = head(features)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item() * x.size(0)
            seen += x.size(0)
            if step % cfg["log_every"] == 0:
                progress.set_postfix(loss=running / max(seen, 1))

        test_acc = evaluate(extractor, head, test_loader, device)
        epoch_accs.append(test_acc)
        print(f"epoch {epoch:02d} | train_loss={running / max(seen, 1):.4f} | test_acc={test_acc * 100:.2f}%")

    return {
        "epoch_accs": epoch_accs,
        "final_acc": epoch_accs[-1],
        "best_acc": max(epoch_accs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default=None, help="override config")
    parser.add_argument("--extractor-checkpoint", type=str, default=None,
                        help="Path to extractor.pt from a CN immunization run; if set, probes against the immunized backbone instead of the pretrained one")
    parser.add_argument("--run-name", type=str, default=None, help="override config run_name")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.dataset is not None:
        cfg["dataset"] = args.dataset
        cfg["run_name"] = f"baseline_probe_resnet18_{args.dataset}"
    if args.extractor_checkpoint is not None:
        cfg["extractor_checkpoint"] = args.extractor_checkpoint
    if args.run_name is not None:
        cfg["run_name"] = args.run_name

    results_dir = Path(cfg["results_dir"]) / cfg["run_name"]
    results_dir.mkdir(parents=True, exist_ok=True)

    out = train_probe(cfg)
    out["config"] = cfg

    out_path = results_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nFinal test accuracy: {out['final_acc'] * 100:.2f}%")
    print(f"Best test accuracy:  {out['best_acc'] * 100:.2f}%")
    print(f"Saved results → {out_path}")


if __name__ == "__main__":
    main()
