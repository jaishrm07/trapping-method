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


# -----------------------------------------------------------------------------
# Split extractor for *partial* immunization.
#
# Zheng et al. §5.2: "For ResNet18, we immunize only the last two convolutional
# blocks." We mirror that. The lower blocks (conv1 .. layer2) stay frozen at
# the ImageNet-pretrained weights; only layer3 + layer4 are updated during
# immunization.
# -----------------------------------------------------------------------------

class _ResNet18Lower(nn.Module):
    """conv1 / bn1 / relu / maxpool / layer1 / layer2 — frozen during immunization."""

    def __init__(self, base):
        super().__init__()
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class _ResNet18Upper(nn.Module):
    """layer3 / layer4 / avgpool / flatten — the immunization-trainable subnet."""

    def __init__(self, base):
        super().__init__()
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

    def forward(self, x):
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def get_resnet18_split() -> Tuple[nn.Module, nn.Module, nn.Module]:
    """Returns `(lower, upper, primary_head)` — pretrained ResNet18 split into
    parts for partial immunization.

    - `lower`: frozen layers (conv1 .. layer2). Pre-frozen with `freeze_module`.
    - `upper`: trainable layers (layer3 .. avgpool). Returns 512-d features.
    - `primary_head`: a copy of the ImageNet 1000-class fc layer, weights from
      the pretrained model. Used for the primary-task cross-entropy term.
    """
    base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    lower = _ResNet18Lower(base)
    upper = _ResNet18Upper(base)
    primary_head = nn.Linear(in_features=512, out_features=1000)
    primary_head.weight.data.copy_(base.fc.weight.data)
    primary_head.bias.data.copy_(base.fc.bias.data)
    freeze_module(lower)
    return lower, upper, primary_head


def get_resnet18_full_extractor_from_split(lower: nn.Module, upper: nn.Module) -> nn.Module:
    """Re-compose `lower + upper` into a single feature-extractor module."""
    class _Composed(nn.Module):
        def __init__(self, lower, upper):
            super().__init__()
            self.lower = lower
            self.upper = upper
            self.feature_dim = 512

        def forward(self, x):
            return self.upper(self.lower(x))

    return _Composed(lower, upper)
