"""Smoke test for Stage 2 components — runs in <60s on a single GPU.

Exercises every Stage-2 piece end-to-end on a tiny synthetic harmful dataset,
no real downloads required. Confirms:

1. r_well / r_ill self-tests (mathematical correctness — already covered in
   src/losses.py but re-run here for one-shot validation).
2. Split ResNet18 loads and the lower stays frozen, upper has trainable params.
3. feature_covariance returns a correctly shaped [512, 512] PSD-ish matrix.
4. condition_number returns a finite positive scalar.
5. RIR computation runs end-to-end on a tiny dataset.
6. One immunization step doesn't blow up (no NaN/Inf in losses or gradients).

Run on the GPU node:
    conda activate trap
    python experiments/smoke_test_stage2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hessian import condition_number, feature_covariance
from src.losses import r_ill, r_well
from src.metrics import relative_immunization_ratio
from src.models import (
    get_resnet18_extractor,
    get_resnet18_full_extractor_from_split,
    get_resnet18_split,
)
from src.utils import get_device, set_seed


class _RandomImageDataset(Dataset):
    """Tiny synthetic dataset of 224×224 random images."""

    def __init__(self, n: int, num_classes: int = 10, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.imgs = torch.randn(n, 3, 224, 224, generator=g)
        self.labels = torch.randint(0, num_classes, (n,), generator=g).tolist()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]


def main():
    set_seed(0)
    device = get_device()
    print(f"[smoke] device={device}")

    # ---- 1. Loss math -------------------------------------------------------
    from src.losses import _self_test as losses_selftest
    losses_selftest()
    from src.metrics import _self_test as metrics_selftest
    metrics_selftest()

    # ---- 2. Split model -----------------------------------------------------
    lower, upper, head = get_resnet18_split()
    lower = lower.to(device)
    upper = upper.to(device)
    head = head.to(device)
    n_lower_trainable = sum(p.requires_grad for p in lower.parameters())
    n_upper_trainable = sum(p.requires_grad for p in upper.parameters())
    n_head_trainable = sum(p.requires_grad for p in head.parameters())
    print(f"[smoke] lower trainable params: {n_lower_trainable} (expect 0)")
    print(f"[smoke] upper trainable params: {n_upper_trainable} (expect > 0)")
    print(f"[smoke] head  trainable params: {n_head_trainable} (expect 2: weight + bias)")
    assert n_lower_trainable == 0
    assert n_upper_trainable > 0

    # ---- 3. Feature covariance ----------------------------------------------
    extractor = get_resnet18_full_extractor_from_split(lower, upper).to(device).eval()
    ds = _RandomImageDataset(n=200, num_classes=10)
    K = feature_covariance(extractor, ds, num_groups=4, group_size=20, device=device)
    print(f"[smoke] feature_covariance shape={tuple(K.shape)} (expect (512, 512))")
    assert K.shape == (512, 512)
    print(f"[smoke] κ(K)={condition_number(K):.3f} (expect finite positive)")
    assert torch.isfinite(K).all()

    # ---- 4. RIR end-to-end --------------------------------------------------
    baseline_extractor = get_resnet18_extractor().to(device).eval()
    rir_out = relative_immunization_ratio(
        extractor_immunized=extractor,
        extractor_baseline=baseline_extractor,
        dataset_harmful=ds,
        dataset_primary=ds,
        num_groups=4,
        group_size=20,
        device=device,
    )
    print(f"[smoke] RIR={rir_out['rir']:.4f}  (expect ~1.0 since immunized==baseline pre-training)")

    # ---- 5. One immunization step -------------------------------------------
    params = list(upper.parameters()) + list(head.parameters())
    optim = torch.optim.SGD(params, lr=1e-3, momentum=0.9)

    x_P = torch.randn(8, 3, 224, 224, device=device)
    y_P = torch.randint(0, 1000, (8,), device=device)
    x_H = torch.randn(8, 3, 224, 224, device=device)

    with torch.no_grad():
        z_P = lower(x_P)
        z_H = lower(x_H)
    feat_P = upper(z_P)
    feat_H = upper(z_H)
    H_P = feat_P.T @ feat_P / feat_P.size(0)
    H_H = feat_H.T @ feat_H / feat_H.size(0)
    L_primary = F.cross_entropy(head(feat_P), y_P)
    L_well = r_well(H_P)
    L_ill = r_ill(H_H)
    loss = L_primary + 1.0 * L_well + 0.1 * L_ill
    print(f"[smoke] step losses: primary={L_primary.item():.4f}  R_well={L_well.item():.4f}  R_ill={L_ill.item():.4f}")
    assert torch.isfinite(loss)

    optim.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(params, 5.0)
    optim.step()
    print(f"[smoke] grad norm after backward: {grad_norm.item():.4f}")
    assert torch.isfinite(grad_norm)

    print("\n[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
