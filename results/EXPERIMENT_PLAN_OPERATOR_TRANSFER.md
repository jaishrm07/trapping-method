# Experiment Plan — Operator-Transfer Evaluation

Date: 2026-05-01

The single empirical question this experiment answers:

> Does trap geometry, shaped against a *linear-probing* adversary,
> bound stronger adversarial operators (LoRA, full FT) on the same
> downstream task?

This is the operator-transfer question raised in the email pitch and in
`papers/trapping_paper_for_llms.md` Direction A. The outcome shapes
whether trap-style immunization is a deployable defense or a narrowly-
calibrated one.

## What we vary — five adversary configurations

| Adversary | What it updates | Trainable params |
|---|---|---|
| `linear_probe` (paper's setup) | Fresh head only | ~100 K |
| `lora_r8` | LoRA-rank-8 on layer3+layer4 + fresh head | ~200 K |
| `lora_r32` | LoRA-rank-32 on layer3+layer4 + fresh head | ~700 K |
| `full_ft_upper` | All params of layer3+layer4 + fresh head | ~9 M |
| `full_ft_all` | Even the frozen lower + everything else + fresh head | ~11 M |

Adversaries are sorted by approximate "expressive capacity." Linear probing
is the paper's training-time inner loop; the other four are out-of-distribution.

## Fixed protocol

- **Target task:** Stanford Cars (196 classes), HuggingFace `tanganke/stanford_cars`.
- **Optimizer:** SGD with momentum 0.9, weight-decay 1e-4.
- **Learning rate:** lr=0.01 for all adversaries (simple/fair comparison).
- **Batch size:** 64. **Epochs:** 50. **Image size:** 224×224.
- **Augmentation:** RandomResizedCrop + RandomHorizontalFlip (train); CenterCrop (test).

A future "strong" version would use per-adversary tuned LRs (LP=1e-2, LoRA=1e-3, FT=1e-4). v1 keeps lr fixed for clean operator-vs-operator comparison.

## What we measure

Per (immunized backbone, adversary) pair:
- 50-epoch test-accuracy trajectory on Cars
- Final acc, peak acc
- RFD against a *matched* baseline (same adversary, un-immunized backbone)

## The minimum-viable matrix

10 runs total:

| Backbone \ Adversary | linear_probe | lora_r8 | lora_r32 | full_ft_upper | full_ft_all |
|---|---|---|---|---|---|
| **Un-immunized** (baseline) | ✓ already have | new | new | new | new |
| **Stage 4.5 (paper-Pareto immunized)** | ✓ already have | new | new | new | new |

8 new runs + 2 already done.

For paper-quality, we'd add 4 more rows (Stage 2, Stage 4, Stage 4.6 immunized backbones — full backbone × adversary matrix). Defer.

## Implementation

Three new files:

1. **`src/lora.py`** — minimal hand-rolled LoRA wrapper for `nn.Conv2d`:

    ```python
    class LoRAConv2d(nn.Module):
        def __init__(self, base_conv, rank):
            ...  # frozen base + low-rank conv2d delta
            # forward: base_conv(x) + lora_B(lora_A(x))
    
    def lorafy(module, rank):
        """Recursively replace each nn.Conv2d in `module` with
        LoRAConv2d. Modifies in place. Result has only LoRA params
        as requires_grad=True."""
    ```

2. **`experiments/run_adversary.py`** — generalization of
   `run_baseline_probe.py` that accepts an adversary type and sets up
   the appropriate gradient mask:

    ```python
    --adversary-type {linear_probe, lora_r8, lora_r32, full_ft_upper, full_ft_all}
    --extractor-checkpoint <path or empty>  # if empty, use pretrained ResNet18
    --run-name <output dir name>
    ```

3. **`scripts/operator_transfer_array.slurm`** — SLURM array job that
   submits all 8 new runs in parallel:

    ```bash
    #SBATCH --array=0-7
    # 0-3 = baselines (no immunization) × 4 adversaries excl. linear_probe
    # 4-7 = Stage 4.5 immunized × 4 adversaries excl. linear_probe
    ```

## Compute estimate

- Linear probe: 25 min on L40S (already known)
- LoRA-r=8/32: ~25 min (similar workload, slightly more params)
- Full FT of upper: ~30 min (more memory, slightly slower per step)
- Full FT all: ~35 min (most memory, slightly slower)

Total compute: 8 runs × ~30 min avg = 4 hours. As an array job with 4-8 parallel slots: 30 min – 1 hour wall-clock.

Cluster is currently busy — actual wall time depends on queue state.

## Possible outcomes and their reading

| Pattern | Reading |
|---|---|
| RFD high (≥30) for all 5 adversaries | Trap is *operator-agnostic*. Surprising and strong; method paper. |
| RFD high for LP, drops sharply for LoRA + FT | Trap is *operator-specific*. Predicted by the email's framing. Motivates multi-operator trap. |
| RFD inversely correlated with adversary param count | Brittleness scales with adversary capacity. Establishes a quantified failure mode. |
| LoRA RFD high but full-FT RFD low | The bottleneck is full-feature updates, not low-rank ones. Suggests *parameter-subspace* trap is enough. |

All four outcomes are publishable.

## Files at completion

```
results/
├── adv_baseline_lora_r8/results.json
├── adv_baseline_lora_r32/results.json
├── adv_baseline_full_ft_upper/results.json
├── adv_baseline_full_ft_all/results.json
├── adv_immunized_45_lora_r8/results.json
├── adv_immunized_45_lora_r32/results.json
├── adv_immunized_45_full_ft_upper/results.json
├── adv_immunized_45_full_ft_all/results.json
└── OPERATOR_TRANSFER_REPORT.md  (after analysis)
```

Plus an analysis script that computes the 1×5 RFD row and any
per-trajectory plots.

## Risks and mitigations

1. **Full FT convergence in 50 epochs.** Linear probing converges in
   50 epochs on Cars; full FT may need more. **Mitigation:** report
   peak accuracy across epochs, not just final, so under-converged
   adversaries don't artifactually inflate RFD.
2. **LoRA rank choice may be unfair.** r=8 is small for ResNet18;
   r=32 is more typical. **Mitigation:** report both ranks; show how
   transfer behavior scales with rank.
3. **Same lr is a confound.** SGD lr=0.01 is right for LP, possibly
   too aggressive for FT. **Mitigation:** v1 result is preliminary;
   the per-adversary-tuned-lr version is the v2 follow-up.
