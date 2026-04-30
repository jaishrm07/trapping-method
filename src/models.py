"""Model loading and a thin linear-probe head.

The trapping paper uses ImageNet-pretrained ResNet18 and ViT, with the
adversary doing linear probing on a frozen feature extractor. So we need:
- a frozen `f_θ` that returns features (no final classification head)
- a fresh, trainable linear head `ω_H` mapping features → class logits
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class FeatureExtractor(nn.Module):
    """Wraps a torchvision backbone and exposes its penultimate features."""

    def __init__(self, backbone: nn.Module, feature_dim: int):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def get_resnet18_extractor() -> FeatureExtractor:
    """ImageNet-pretrained ResNet18 with the final fc replaced by Identity.

    Output: [B, 512] features.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    feature_dim = model.fc.in_features  # 512
    model.fc = nn.Identity()
    return FeatureExtractor(model, feature_dim=feature_dim)


class LinearProbeHead(nn.Module):
    """Single linear layer: features → class logits.

    Initialized with Kaiming-normal weights and zero bias by default — matches
    the standard linear probing protocol.
    """

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes)
        nn.init.kaiming_normal_(self.linear.weight, mode="fan_out", nonlinearity="linear")
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def freeze_module(module: nn.Module) -> None:
    """Set requires_grad=False on all parameters and put module in eval mode."""
    for p in module.parameters():
        p.requires_grad = False
    module.eval()


def build_probe_pipeline(num_classes: int) -> Tuple[FeatureExtractor, LinearProbeHead]:
    """Standard pre-trained-frozen-extractor + fresh-linear-head pipeline."""
    extractor = get_resnet18_extractor()
    freeze_module(extractor)
    head = LinearProbeHead(feature_dim=extractor.feature_dim, num_classes=num_classes)
    return extractor, head
