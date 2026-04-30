"""Feature-covariance Hessian approximation per Zheng et al. (ICML 2025) §5.2 Eq. 17–18.

For non-linear extractors, the Hessian of the linear-probing loss is approximated
as the feature covariance:

    H̃(θ) = X̃(θ)^T X̃(θ),  where X̃(θ) ∈ ℝ^{N × D_hid} stacks features over D.

Computing this on the full dataset is memory-intensive at scale, so Zheng et al.
sample 20 groups of 100 examples and average. We follow the same protocol.

This module is used both during immunization (when the regularizers act on H̃)
and during evaluation (for the RIR metric).
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler


def _iter_random_minibatches(dataset: Dataset, group_size: int, num_groups: int, seed: int = 0):
    """Yield `num_groups` minibatches of `group_size` examples each, drawn with replacement."""
    g = torch.Generator()
    g.manual_seed(seed)
    n = len(dataset)
    for _ in range(num_groups):
        idx = torch.randint(low=0, high=n, size=(group_size,), generator=g).tolist()
        xs = [dataset[i][0] for i in idx]
        yield torch.stack(xs)


@torch.no_grad()
def feature_covariance(
    extractor: nn.Module,
    dataset: Dataset,
    *,
    num_groups: int = 20,
    group_size: int = 100,
    device: torch.device | str = "cuda",
    seed: int = 0,
) -> torch.Tensor:
    """Compute K̃ = mean over groups of (X̃^T X̃) on `extractor(dataset)` features.

    Returns a [D_hid, D_hid] tensor on `device`.
    """
    extractor.eval()
    accum: torch.Tensor | None = None
    for batch in _iter_random_minibatches(dataset, group_size, num_groups, seed=seed):
        batch = batch.to(device, non_blocking=True)
        feats = extractor(batch)
        if feats.dim() > 2:
            feats = feats.flatten(1)
        K = feats.T @ feats  # [D_hid, D_hid]
        accum = K if accum is None else accum + K
    assert accum is not None, "no groups produced — empty dataset?"
    return accum / num_groups


def condition_number(matrix: torch.Tensor, eps: float = 1e-12) -> float:
    """κ(M) = σ_max(M) / σ_min(M).

    Uses the smallest non-trivially-zero singular value as σ_min. For
    near-singular matrices, the result is dominated by `eps` — interpret with
    care.
    """
    sigmas = torch.linalg.svdvals(matrix)
    sigma_min = sigmas[-1].clamp(min=eps)
    sigma_max = sigmas[0]
    return float(sigma_max / sigma_min)
