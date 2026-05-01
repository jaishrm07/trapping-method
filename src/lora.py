"""Minimal LoRA wrapper for `nn.Conv2d`.

For the operator-transfer experiment, we want adversaries that update
ResNet18's convolutions in a low-rank subspace rather than a fresh head
only. Standard LoRA (Hu et al. 2021) was originally for linear layers;
the conv2d analogue stores a rank-r decomposition `W' = W + B·A` where
both A and B are themselves conv operators.

Implementation: a wrapper module that holds the frozen base conv plus
two small auxiliary convs whose composition is the rank-r delta.

Forward:  y = base_conv(x) + lora_B(lora_A(x))
Init:     lora_A ~ Kaiming-uniform, lora_B = 0  (so initial delta = 0)

The wrapper marks the base conv's parameters as `requires_grad=False`
and leaves only the LoRA factors trainable.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LoRAConv2d(nn.Module):
    """Wrap an existing `nn.Conv2d` with a rank-r additive low-rank delta."""

    def __init__(self, base_conv: nn.Conv2d, rank: int):
        super().__init__()
        if not isinstance(base_conv, nn.Conv2d):
            raise TypeError(f"LoRAConv2d expects nn.Conv2d, got {type(base_conv)}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        self.base_conv = base_conv
        for p in self.base_conv.parameters():
            p.requires_grad = False

        # lora_A: same kernel/stride/padding/dilation as base, in_channels →
        # rank, no bias.
        self.lora_A = nn.Conv2d(
            base_conv.in_channels,
            rank,
            kernel_size=base_conv.kernel_size,
            stride=base_conv.stride,
            padding=base_conv.padding,
            dilation=base_conv.dilation,
            bias=False,
        )
        # lora_B: 1×1 conv, rank → out_channels, no bias.
        self.lora_B = nn.Conv2d(rank, base_conv.out_channels, kernel_size=1, bias=False)
        # Init: A random, B zero so initial delta is the zero map.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_conv(x) + self.lora_B(self.lora_A(x))


def lorafy(module: nn.Module, rank: int) -> int:
    """Replace every `nn.Conv2d` descendant of `module` with `LoRAConv2d(rank)`.

    Modifies `module` in place. Returns the number of conv layers that were
    wrapped, for sanity-checking.
    """
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, LoRAConv2d(child, rank=rank))
            count += 1
        else:
            count += lorafy(child, rank=rank)
    return count


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _self_test() -> None:
    torch.manual_seed(0)

    # Dummy "module" with a few convs
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc = nn.Linear(32 * 4 * 4, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = nn.functional.adaptive_avg_pool2d(x, 4)
            return self.fc(x.flatten(1))

    m = Toy()
    n_conv = lorafy(m, rank=4)
    assert n_conv == 2, f"Expected 2 convs lorafied, got {n_conv}"

    # After lorafying, base conv params should be frozen, lora params trainable.
    n_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in m.parameters() if not p.requires_grad)
    assert n_trainable > 0, "LoRA params should be trainable"
    assert n_frozen > 0, "Base conv weights should be frozen"

    # Initial output should be EXACTLY equal to pre-lora output (B=0).
    m_ref = Toy()
    m_ref.load_state_dict({k.replace('base_conv.', ''): v for k, v in m.state_dict().items() if 'lora' not in k}, strict=False)
    x = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        y_lora = m(x)
        # Forward through original (still embedded in m via base_conv)
        # — easier to just check the LoRA delta is zero
        for mod in m.modules():
            if isinstance(mod, LoRAConv2d):
                # Manually probe: lora_B(lora_A(x)) should be zero on any input
                test_x = torch.randn(1, mod.base_conv.in_channels, 8, 8)
                delta = mod.lora_B(mod.lora_A(test_x))
                assert delta.abs().max() < 1e-6, f"Initial LoRA delta should be 0, got {delta.abs().max()}"

    # After a SGD step on a fake loss, the lora params should change but
    # base conv weights should not.
    optim = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1)
    x = torch.randn(2, 3, 16, 16)
    y = m(x)
    loss = y.pow(2).sum()
    base_w_before = next(mod.base_conv.weight.clone() for mod in m.modules() if isinstance(mod, LoRAConv2d))
    lora_b_before = next(mod.lora_B.weight.clone() for mod in m.modules() if isinstance(mod, LoRAConv2d))
    optim.zero_grad()
    loss.backward()
    optim.step()
    base_w_after = next(mod.base_conv.weight for mod in m.modules() if isinstance(mod, LoRAConv2d))
    lora_b_after = next(mod.lora_B.weight for mod in m.modules() if isinstance(mod, LoRAConv2d))
    assert torch.equal(base_w_before, base_w_after), "Base conv weights should not change"
    assert not torch.equal(lora_b_before, lora_b_after), "LoRA B weights should change"

    print("lora.py self-test passed")
    print(f"  trainable params after lorafy: {n_trainable}, frozen: {n_frozen}")


if __name__ == "__main__":
    _self_test()
