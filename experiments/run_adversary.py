"""Generalized adversary probe — operator-transfer experiment.

Runs a 50-epoch downstream-task probe on Cars, with the choice of
adversary update operator:

    linear_probe   - Fresh head only (paper's setup)
    lora_r8        - LoRA-rank-8 on layer3+layer4 + fresh head
    lora_r32       - LoRA-rank-32 on layer3+layer4 + fresh head
    full_ft_upper  - All params of layer3+layer4 + fresh head
    full_ft_all    - Even the frozen lower + everything else + fresh head

Each operator measures: how much does this immunized backbone slow
down THIS adversarial fine-tuning recipe? Trap shaped against linear
probing may or may not bound stronger operators — that's the question.

Optional `--extractor-checkpoint` loads a saved (lower, upper) state
dict; otherwise the standard ImageNet-pretrained ResNet18 is used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_dataset_by_name, make_loaders
from src.lora import lorafy
from src.models import (
    LinearProbeHead,
    freeze_module,
    get_resnet18_extractor,
    get_resnet18_full_extractor_from_split,
    get_resnet18_split,
)
from src.utils import get_device, set_seed


def setup_adversary(
    adversary_type: str,
    num_classes: int,
    extractor_checkpoint: str | None,
):
    """Build (extractor, head, params_to_train) for the given adversary type.

    The extractor module is in either training or eval mode depending on
    whether any of its params are trainable. Caller is responsible for
    moving to device.
    """
    use_split = adversary_type != "linear_probe" or extractor_checkpoint is not None

    if use_split:
        lower, upper, _ = get_resnet18_split()
        if extractor_checkpoint:
            ckpt = torch.load(extractor_checkpoint, map_location="cpu", weights_only=True)
            lower.load_state_dict(ckpt["lower"])
            upper.load_state_dict(ckpt["upper"])
            print(f"Loaded extractor checkpoint from {extractor_checkpoint}")
        else:
            print("Using ImageNet-pretrained ResNet18 (no immunization)")
    else:
        # Plain pretrained backbone for linear-probe-from-pretrained
        extractor = get_resnet18_extractor()
        freeze_module(extractor)
        head = LinearProbeHead(extractor.feature_dim, num_classes)
        return extractor, head, list(head.parameters())

    if adversary_type == "linear_probe":
        # Frozen entire extractor, train only the head.
        freeze_module(lower)
        freeze_module(upper)
        extractor = get_resnet18_full_extractor_from_split(lower, upper)
        head = LinearProbeHead(512, num_classes)
        params = list(head.parameters())
        n_lora_layers = 0

    elif adversary_type in ("lora_r8", "lora_r32"):
        rank = 8 if adversary_type == "lora_r8" else 32
        # Freeze lower entirely. Lorafy upper - its base convs will be frozen
        # in-place by LoRAConv2d, only LoRA params will be trainable.
        freeze_module(lower)
        # Re-enable training mode on upper so BN stats can update — but BN
        # weights/biases are NOT in LoRA's trainable set (frozen above by
        # iterating named params? Actually freeze_module didn't run on upper.
        # So upper's BN params are trainable. Force-freeze them.)
        for p in upper.parameters():
            p.requires_grad = False
        n_lora_layers = lorafy(upper, rank=rank)
        # After lorafy, only the LoRA factors have requires_grad=True.
        extractor = get_resnet18_full_extractor_from_split(lower, upper)
        head = LinearProbeHead(512, num_classes)
        params = [p for p in extractor.parameters() if p.requires_grad] + list(head.parameters())

    elif adversary_type == "full_ft_upper":
        # Freeze lower; unfreeze upper entirely.
        freeze_module(lower)
        for p in upper.parameters():
            p.requires_grad = True
        upper.train()
        extractor = get_resnet18_full_extractor_from_split(lower, upper)
        head = LinearProbeHead(512, num_classes)
        params = list(upper.parameters()) + list(head.parameters())
        n_lora_layers = 0

    elif adversary_type == "full_ft_all":
        # Unfreeze everything.
        for p in lower.parameters():
            p.requires_grad = True
        for p in upper.parameters():
            p.requires_grad = True
        lower.train()
        upper.train()
        extractor = get_resnet18_full_extractor_from_split(lower, upper)
        head = LinearProbeHead(512, num_classes)
        params = list(lower.parameters()) + list(upper.parameters()) + list(head.parameters())
        n_lora_layers = 0

    else:
        raise ValueError(f"Unknown adversary_type: {adversary_type}")

    n_trainable = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in extractor.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"adversary={adversary_type} | trainable params: {n_trainable:,} / {n_total:,} "
          f"({100*n_trainable/n_total:.1f}%) | lora_layers_wrapped: {n_lora_layers}")
    return extractor, head, params


def evaluate(extractor: nn.Module, head: nn.Module, loader, device: torch.device) -> float:
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


def train_adversary(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    device = get_device()
    print(f"Device: {device}")

    splits = load_dataset_by_name(cfg["dataset"], root=cfg["data_root"], image_size=cfg["image_size"])
    train_loader, test_loader = make_loaders(splits, batch_size=cfg["probe"]["batch_size"], num_workers=cfg["probe"]["num_workers"])
    print(f"Loaded {cfg['dataset']}: {len(splits.train)} train, {len(splits.test)} test, {splits.num_classes} classes")

    extractor, head, params = setup_adversary(
        adversary_type=cfg["adversary_type"],
        num_classes=splits.num_classes,
        extractor_checkpoint=cfg.get("extractor_checkpoint"),
    )
    extractor = extractor.to(device)
    head = head.to(device)

    optimizer = torch.optim.SGD(
        params,
        lr=cfg["probe"]["lr"],
        momentum=cfg["probe"]["momentum"],
        weight_decay=cfg["probe"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    epoch_accs: list[float] = []
    is_extractor_trainable = any(p.requires_grad for p in extractor.parameters())

    for epoch in range(1, cfg["probe"]["epochs"] + 1):
        head.train()
        if is_extractor_trainable:
            extractor.train()
        # else extractor stays in eval mode for fully-frozen runs (no BN drift)

        running = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch:02d}", leave=False)
        for step, (x, y) in enumerate(progress, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if is_extractor_trainable:
                features = extractor(x)
            else:
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
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--adversary-type", type=str, required=True,
                        choices=["linear_probe", "lora_r8", "lora_r32", "full_ft_upper", "full_ft_all"])
    parser.add_argument("--extractor-checkpoint", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.dataset is not None:
        cfg["dataset"] = args.dataset
    cfg["adversary_type"] = args.adversary_type
    if args.extractor_checkpoint:
        cfg["extractor_checkpoint"] = args.extractor_checkpoint
    cfg["run_name"] = args.run_name or f"adv_{args.adversary_type}_{cfg.get('dataset', 'cars')}"

    results_dir = Path(cfg["results_dir"]) / cfg["run_name"]
    results_dir.mkdir(parents=True, exist_ok=True)

    out = train_adversary(cfg)
    out["config"] = cfg

    out_path = results_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nFinal test accuracy: {out['final_acc'] * 100:.2f}%")
    print(f"Best test accuracy:  {out['best_acc'] * 100:.2f}%")
    print(f"Saved results → {out_path}")


if __name__ == "__main__":
    main()
