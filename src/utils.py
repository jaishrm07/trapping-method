"""Small utilities: seeding, device selection, logging boilerplate."""
from __future__ import annotations

import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(module: torch.nn.Module, only_trainable: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not only_trainable))
