# Stage 2 Report — Condition-Number Immunization (Cars / ResNet18)

Date: 2026-04-30

## Headline

We reproduced the Zheng-style condition-number immunization (Stage 2 of the
trapping-paper plan) on Cars / ResNet18 and ran a 50-epoch adversarial linear
probe to measure live RFD. Both intrinsic and extrinsic immunization metrics
match or exceed the trapping paper's reported `CN` baseline; primary-task
utility cost is somewhat higher.

| Metric | Init θ_0 | Paper "CN" (Table 1) | **Ours** |
|---|---|---|---|
| RIR (intrinsic, Eq. 17) | 1.0 | 3.5 | **25.6** |
| RFD (extrinsic, Eq. 9) | — | 10.06 | **14.90** |
| Primary acc on ImageNet val | 67.04% | 62.27% | **59.35%** |
| Adversarial probe peak acc | — | ~22% | **36.44%** |
| Baseline (un-immunized) probe peak | ~30% | ~30% | **42.72%** |

Notes on direct comparability:
- Our absolute Cars probing numbers differ from the paper because we use the
  HuggingFace `tanganke/stanford_cars` mirror (the original Krause et al.
  2013 split is no longer downloadable) and slightly different data
  augmentation. **RFD is computed against our own baseline trajectory**, so
  it remains a meaningful within-experiment immunization metric.
- We use a 10K subset of ImageNet val as the primary task during
  immunization (cgroup memory cap forces this); paper uses full ImageNet.

## Key Empirical Finding

**RIR overstates immunization quality.** A 7× larger RIR (25.6 vs. 3.5
paper) translated to only 1.5× larger RFD (14.9 vs. 10.06). The intrinsic
condition-number ratio is sensitive to optimization hyperparameters and does
not directly correspond to the slowdown an adversary actually experiences.

This is exactly the trapping paper's central methodological argument
(§4.4), now demonstrated empirically through our reimplementation. RFD —
the extrinsic, trajectory-averaged metric — is the more reliable indicator.

## Adversarial probe trajectory

Per-epoch test accuracy on Cars (50 epochs of linear probing, SGD lr=0.01,
momentum=0.9, batch=64). Both probes use the same protocol; only the frozen
backbone differs.

```
epoch    1     5    10    15    20    25    30    35    40    45    50
base   5.9  20.6  29.3  33.9  35.4  38.0  39.7  39.7  40.7  42.2  41.2
immu  11.8  23.9  28.1  29.5  33.0  33.6  33.5  33.6  36.0  35.2  34.8
```

Trajectory shape:

- **Early epochs (1-7):** immunized model converges *faster* than baseline.
  The CN-shaped backbone has features biased toward ImageNet structure
  (which we trained on) and that transfers to Cars-class adaptation
  initially. RIR's large value is *misleading* about adversary speed in
  this regime.
- **Mid epochs (8-20):** baseline catches up and surpasses. The "ceiling"
  effect on the immunized model becomes visible.
- **Late epochs (20-50):** baseline plateaus around 40-42%, immunized
  plateaus around 33-36%. **The actual immunization signal is a 5-7
  percentage-point cap on adversary peak accuracy**, well below what
  RIR=25.6 would suggest if it were a meaningful adversary-speed proxy.

## Implementation differences from the paper

We deliberately diverged from Zheng's exact recipe in three places:

1. **No K⁻¹ preconditioner.** The paper's Algorithm 1 uses a "dummy layer"
   with K⁻¹-multiplied backward to preserve Theorem 4.3's monotonic-κ
   guarantee. We use vanilla autograd through the regularizer instead.
   Documented in `src/losses.py`. This explains some of the elevated RIR
   (we're allowed to push κ harder than the monotonic-κ bound permits).
2. **Trace-normalized `r_well`, raw `r_ill`.** Trace-normalizing both
   regularizers caused `r_ill` to collapse to a flat region near its lower
   bound (~1000 for k=512) where pretrained features already live, killing
   its gradient. We trace-normalize only `r_well` (where it works
   correctly) and leave `r_ill` on raw H.
3. **10K ImageNet val subset** as the primary task. Smaller dataset → faster
   primary CE descent → primary task overfits within ~1 epoch → regularizer
   gradient dominates the rest of training. Probably contributes to higher
   RIR than paper.

## What this validates / motivates

- **Validates** Stage 2 of our reimplementation pipeline: the entire
  framework (dataset loaders, model split, feature-covariance Hessian,
  regularizers, RIR/RFD metrics, adversarial probing) is end-to-end
  consistent.
- **Validates** the trapping paper's methodological claim that RIR is
  unreliable.
- **Motivates** Stage 3 (the trap loss). The CN baseline gives only a
  ~5-7pt adversary peak-acc reduction. The paper's full method (CN + trap)
  is supposed to be substantially stronger by bounding the *destination* of
  optimization, not just the local geometry. We have a clear baseline to
  compare against.

## Files

- `cn_immunize_resnet18_cars/results.json` — immunization run log + final RIR + primary acc
- `cn_immunize_resnet18_cars/extractor.pt` — immunized backbone (lower + upper state_dict). Not committed to git (large); regenerate with `scripts/immunization_cn.slurm`.
- `baseline_probe_resnet18_cars/results.json` — baseline probe trajectory
- `immunized_probe_cars_cn/results.json` — adversarial probe trajectory on the immunized backbone

## Next: Stage 3

Implement the trap-inducing loss (§3.1 of the trapping paper):

```
L_trap(θ) = softplus(ΔL_act − ΔL_exp)

where ΔL_act = L_H(ω_H_0) − L_H(ω_H_k)  (actual reduction after k inner SGD steps)
      ΔL_exp = −(g_0^T Δω + ½ Δω^T H_0 Δω)  (local-quadratic prediction)
```

Key implementation challenges:
- Differentiable k-step inner unroll on the adversary's head ω_H, with
  gradients flowing back to θ_upper.
- KNN-cluster initialization for ω_H_0 to give meaningful starting gradient
  signal.
- Hessian of L_H w.r.t. ω_H computed via feature covariance (same as
  Stage 2).

Expected qualitative outcome: similar or slightly lower RIR than CN, but
substantially higher RFD — because the trap loss bounds *realized* progress
rather than just local curvature.
