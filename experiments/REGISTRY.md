# Experiment Registry

One row per canonical run. Append (don't reorder) when a new run completes.
For provenance details (git SHA, SLURM job ID, host), see
`results/<run_name>/results.json` → `provenance` field, populated by
`src/provenance.py` since 2026-05-02.

## Backbone immunization runs (ResNet18 / ImageNet val + Cars)

| Stage | Run name | Method | RIR | Primary acc | Notes |
|---|---|---|---|---|---|
| 2   | `cn_immunize_resnet18_cars`             | CN-only (no trap, no K⁻¹) | ~5707 | 42% | over-aggressive — established RIR is unreliable |
| 4   | `immunized_probe_cars_trap`             | CN + LP trap, no K⁻¹ | high | low | over-aggressive (similar regime) |
| 4.5 | `immunized_probe_cars_kinv_lr1e-3`      | CN + LP trap + K⁻¹ preconditioner | 1.11 | 64.13% | matches paper Pareto neighborhood |
| 4.5 | `immunized_probe_cars_trap_paper_faithful` | Same as above, paper-faithful  | 1.11 | 64.13% | reference for LP-trap baseline |
| 5 v1 | `trap_multiop_lp_lora8_resnet18_cars`  | Multi-op (LP + LoRA-r8), η=0.01 | 2.38 | 63.88% | LoRA branch saturated softplus, no LoRA defense |
| 5 v2 | `trap_multiop_v2_eta1e-1_resnet18_cars`| Multi-op, η=0.1, λ_trap=0.3, grad_clip=1 | 1.295 | 64.98% | first stable multiop run; LoRA-r8 RFD=1.16 |
| 5 v4 | (failed) `trap_multiop_v4_fomaml_predictor_*` | v2 + FOMAML + Σ η‖g_t‖² predictor | — | — | NaN ~step 2000 (predictor HVP cancellation); see `STAGE5_v4a_REPORT.md` |
| 5 v4a | `trap_multiop_v4a_fomaml_only_resnet18_cars` | v2 + FOMAML only (no predictor) | 1.580 | 65.04% | clean null vs v2: LoRA-r8 RFD=1.14 (= v2's 1.16); see `STAGE5_v4a_REPORT.md` |
| 5 v5a | `trap_multiop_v5a_dro_resnet18_cars` | v4a + DRO weighting on operator sampling | 0.533 | 65.43% | DRO null + LP defense erosion: LP RFD 47.95→18.02, LoRA-r8 RFD 1.14→1.04; see `STAGE5_v5_REPORT.md` |
| 5 v5b | (failed) `trap_multiop_v5b_peft_family_*` | v4a + expanded operators {LP, LoRA-r4, r8, r16, r32} | — | — | NaN @ step 1051 (rank-32 likely culprit); see `STAGE5_v5_REPORT.md` |
| 6 | `trap_multiop_v6_taylor_hvp_resnet18_cars` | Exact Taylor-HVP LoRA trap | 0.505 | 65.93% | Stable but negative: LP signed RFD -1.95, LoRA-r8 signed RFD -0.10; see `STAGE6_taylor_hvp_REPORT.md` |
| 8 v8a | `trap_v8a_bonly_resnet18_cars` | LP + B-only LoRA post-attack CE block | 0.724 | 64.09% | Stable negative: B-only RFD 0.20, LoRA-r8 RFD 0.08; see `STAGE8_v8a_REPORT.md` |
| 9 v9a | `trap_v9a_persistent_bonly_resnet18_cars` | Single persistent B-only LoRA attacker | 0.783 | 64.23% | Hard negative: B-only RFD 0.32, LoRA-r8 RFD 0.39; see `STAGE9_v9a_REPORT.md` |
| 10 v10a | `trap_v10a_population_bonly_resnet18_cars` | Population persistent B-only LoRA attackers | 0.657 | 56.54% | Utility collapse; only slight B-only delay: B-only RFD 2.48, LoRA-r8 RFD 0.98; see `STAGE10_v10a_REPORT.md` |
| 12 v12a | `trap_v12a_tangent_removal_resnet18_cars` | Persistent learned LoRA-tangent readout removal | 1.111 | 68.36% | Utility preserved, but negative for LoRA: B-only RFD 0.66, LoRA-r8 RFD 0.37; see `STAGE12_v12a_REPORT.md` |

## Adversarial probes (50-epoch fine-tune on Cars)

Naming: `adv_<context>_<adversary>_cars`. Context tags:
- `baseline` — un-immunized ImageNet ResNet18 backbone
- `immunized_45` — Stage 4.5 LP-trap backbone
- `multiop` — Stage 5 v1 backbone
- `multiop_v2` — Stage 5 v2 backbone
- `v4a` — Stage 5 v4a (FOMAML only) backbone
- `v5a` — Stage 5 v5a (DRO weighting) backbone

Adversaries: `linear_probe | lora_r8 | lora_r32 | full_ft_upper | full_ft_all`.

### LP-trap operator-transfer matrix (Stage 4.5 vs baseline)

| Adversary | Baseline acc | Stage 4.5 acc | RFD |
|---|---|---|---|
| linear_probe | 41.23% | 22.98% | 50.10 |
| lora_r8      | 73.85% | 73.73% | 0.65  |
| lora_r32     | 78.15% | 78.39% | 0.60  |
| full_ft_upper| 80.76% | 80.75% | 0.74  |
| full_ft_all  | 80.64% | 81.40% | 2.02  |

Conclusion: LP trap is narrow — does not transfer to LoRA / full-FT.

### Stage 5 v1 (multi-op LP + LoRA-r8 @ η=0.01)

| Adversary | Stage 5 v1 acc | RFD |
|---|---|---|
| linear_probe | 18.62% | **60.30** ↑ |
| lora_r8      | 74.03% | 0.67 ≈ |
| lora_r32     | 78.54% | 0.60 ≈ |
| full_ft_upper| 80.08% | 0.74 ≈ |
| full_ft_all  | 81.27% | 2.02 ≈ |

Conclusion: η=0.01 + B-init-zero LoRA inner loop saturates softplus. No LoRA bound. LP defense improves as side effect.

### Stage 5 v2 (multi-op LP + LoRA-r8 @ η=0.1, λ_trap=0.3, grad_clip=1)

| Adversary | Stage 5 v2 acc | RFD |
|---|---|---|
| linear_probe | 21.78% | 47.18 |
| lora_r8      | 72.99% | **1.16** ↑ |
| lora_r32     | 78.40% | −0.32 |
| full_ft_upper| 80.15% | 0.76 |
| full_ft_all  | 81.51% | −1.08 |

First non-noise LoRA-r8 defense (0.67 v1 → 1.16 v2). Source: SLURM array 386207.

### Stage 5 v4a (multi-op LP + LoRA-r8, FOMAML only, no predictor)

| Adversary | Stage 5 v4a acc | RFD | Δ vs v2 |
|---|---|---|---|
| linear_probe | 21.46% | 47.95 | +0.77 |
| lora_r8      | 73.01% | **1.14** ≈ | −0.02 |
| lora_r32     | 79.07% | −1.18 | −0.86 |
| full_ft_upper| 80.26% | 0.62 | −0.14 |
| full_ft_all  | 81.41% | −0.95 | +0.13 |

**Clean null**: FOMAML vs second-order MAML doesn't move LoRA RFD at this k=3 setting. RIR rose 1.295 → 1.580 without translating to defense. Full analysis: `results/STAGE5_v4a_REPORT.md`. Source: role-lab GPUs 1–5 (parallel).

### Stage 5 v5a (multi-op LP + LoRA-r8, FOMAML, DRO weighting)

| Adversary | Stage 5 v5a acc | RFD | Δ vs v4a |
|---|---|---|---|
| linear_probe | 33.80% | **18.02** ↓↓↓ | **−29.93** |
| lora_r8      | 73.08% | 1.04 ≈ | −0.10 |
| lora_r32     | 78.92% | −0.99 | +0.19 |
| full_ft_upper| 79.93% | 1.03 | +0.41 |
| full_ft_all  | 81.02% | −0.47 | +0.48 |

**DRO null + LP defense erosion**: as both operators saturate (LP from defender success, LoRA from inner overshoot), DRO down-weights both. LP defense erodes from RFD 47.95 → 18.02; LoRA-r8 RFD essentially unchanged (1.14 → 1.04). Full analysis: `results/STAGE5_v5_REPORT.md`.

### The hard ceiling at LoRA-r8 RFD ≈ 1.1

Across four orthogonal interventions (LR, autograd discipline, operator weighting), LoRA-r8 RFD has not exceeded ~1.2:

| Run | LoRA-r8 RFD |
|---|---|
| Stage 4.5 (LP only) | 0.65 |
| v1 (η=0.01, uniform) | 0.67 |
| v2 (η=0.1, uniform) | 1.16 |
| v4a (η=0.1, FOMAML, uniform) | 1.14 |
| v5a (η=0.1, FOMAML, DRO) | 1.04 |

**The bilevel-trap formulation as currently expressed has a structural ~1pp LoRA-r8 RFD ceiling.** Next-direction work is no longer hyperparameter tuning but trap formulation, inner-adversary strength, or threat-model reframing.

## Open runs / queued

- C1, C2: `lambda_trap=0` variants of paperexact and lill100 — quantifies the trap-induced RIR multiplier in our pipeline. Compares to corresponding trap-on rows.

## RIR replication investigation (2026-05-04)

Full analysis: `results/STAGE_RIR_REPLICATION_REPORT.md`.

Cloned Zheng's reference (`github.com/amberyzheng/model-immunization-cond-num`).
Five specific differences in our pre-2026-05-04 RIR computation: dtype (float32 vs
**float64**), decomp (`svdvals` vs **`eigvalsh + λI`**), σ_min selection
(`clamp_min(1e-12)` vs **`min(eigs > λ_diag)`**), and **per-group κ aggregation**
(was: average K, κ once). Patched `src/metrics.py` to be Zheng-faithful by
default; `legacy=True` preserves the old impl.

### Zheng-faithful RIR scores on 7 candidate extractors

| Variant | RIR_zheng | κH_ratio | κP_ratio | Primary acc |
|---|---|---|---|---|
| paperexact (λ_trap=1, lr=1e-5, paper hyperparams) | 0.882 | 1.23 | 2.28 | 65.69% |
| A1 K⁻¹ OFF | 1.188 | 2.75 | 2.90 | 64.09% |
| B1 +λ_ill 10× | 2.413 | 4.63 | 3.16 | 62.65% |
| **B2 +λ_ill 100×** | **3.066** | 6.27 | 3.17 | 60.63% |
| B3 iter=10k | 2.061 | 3.63 | 1.97 | 43.52% |
| lill100 (4.5b) | 0.136 | 0.38 | 4.81 | 45.27% |
| lill10k (4.5b) | 0.080 | 0.58 | 6.60 | 40.66% |

### What matches and what doesn't

| Paper-reported | Our value | Match? |
|---|---|---|
| Zheng Table 3 "Ours" RIR=2.386 ± 0.442 (CN-only) | B2 = 3.07 | ✓ within ±error band |
| Sarker Table 1 "CN" RIR=3.52, primary 62.27% | B2 = 3.07, primary 60.63% | ✓ within run-to-run noise |
| Sarker Table 1 "Ours" RIR=43.92, primary 65.99% (CN+trap) | paperexact = 0.88, primary 65.69% | ✗ 50× off on RIR; primary matches |
| Sarker LP RFD = 47.19 | Stage 4.5 = 50.10 | ✓ within noise |

The narrow non-replication is **the trap-induced RIR multiplier**. Paper claims
trap takes RIR from 3.52 (CN baseline) to 43.92 (CN+trap) — a 12× boost. Our
trap doesn't produce that boost. RFD-based defense effect DOES reproduce.

### Trap-vs-CN diagnostic + bug hunt (C1, C2, D1, D2)

C1/C2 (CN-only at paperexact and lill100 hyperparameters) reveal that adding
our trap loss **strictly decreases RIR** in our pipeline:

| Variant | RIR_zheng | κH_ratio | κP_ratio | Primary acc |
|---|---|---|---|---|
| paperexact (CN+trap, default) | 0.904 | 1.25 | 1.88 | 65.69% |
| **C1 paperexact NO-trap (CN-only)** | **3.966** | **6.08** | 1.93 | 66.19% |
| C2 lill100 NO-trap | 2.990 | 5.40 | 6.18 | 58.27% |
| D1 K_no_B (CN+trap, K=X^TX) | 0.595 | 0.77 | 1.67 | 64.99% |
| D2 no_detach (CN+trap, centroid grad-flow) | 0.824 | 1.43 | 2.20 | 65.24% |

Removing trap takes RIR from 0.90 → 3.97 (4.4× boost) — **opposite direction**
from paper's reported trap effect (3.52 → 43.92, 12× boost). Two specific
implementation suspects (K normalization, centroid detach) ruled out via D1/D2
ablations. Remaining suspects: softmax-aware Hessian, inner-loop
hyperparameters, deeper sign/path bug. Cannot fully resolve without paper's
source code.

## Stage 7 tangent-access diagnostics

Full analysis: `results/STAGE7_tangent_access_REPORT.md`.

### Random finite-difference LoRA tangent sketch

| Run | M directions | Base final | Tangent final | Base + tangent final |
|---|---:|---:|---:|---:|
| `tangent_access_baseline_r8_m8` | 8 | 34.40% | 21.12% | 23.60% |
| `tangent_access_v4a_r8_m8` | 8 | 20.35% | 15.47% | 15.91% |
| `tangent_access_v6_taylor_hvp_r8_m8` | 8 | 33.50% | 18.55% | 21.14% |
| `tangent_access_v4a_r8_m32` | 32 | 20.23% | 10.72% | 10.53% |

Conclusion: small random sketches are not a good proxy for LoRA accessibility.
They sample arbitrary directions; the attacker optimizes useful directions.

### B-only LoRA-r8 adversary

`lora_bonly_r8` freezes LoRA A and trains only LoRA B plus the harmful head.
This tests the fixed random adapter basis exposed at standard LoRA init.

| Operator | Baseline final | v4a final | v6 final | v4a signed RFD | v6 signed RFD |
|---|---:|---:|---:|---:|---:|
| linear_probe | 41.23% | 21.46% | 43.27% | 55.98 | -1.95 |
| lora_bonly_r8 | 51.19% | 51.10% | 51.37% | 0.71 | -0.04 |
| lora_r8 | 73.85% | 73.01% | 73.50% | 0.28 | -0.10 |
| lora_r32 | 78.15% | 79.07% | 78.31% | 0.06 | -0.04 |

Conclusion: v4a blocks frozen-feature readout but not the optimized adapter
basis. This supports the operator-reachability principle for the next defense.

## Stage 8 v8a B-only immunization result

Full analysis: `results/STAGE8_v8a_REPORT.md`.

| Probe | Baseline final | v8a final | Signed RFD |
|---|---:|---:|---:|
| linear_probe | 41.23% | 41.00% | 5.85 |
| lora_bonly_r8 | 51.19% | 51.25% | 0.20 |
| lora_r8 | 73.85% | 73.98% | 0.08 |
| lora_r32 | 78.15% | 78.37% | 0.19 |
| full_ft_upper | 80.76% | 80.60% | 0.14 |

Conclusion: direct k=10 B-only CE-threshold blocking does not move the
downstream B-only or full-LoRA probes. This rules out the simplest
adapter-basis-aware fix and points toward long-horizon/persistent attackers or
tangent-kernel alignment.

## Stage 9 v9a persistent-attacker result

Plan: `results/STAGE9_PLAN_persistent_bonly.md`.
Implementation log: `results/STAGE9_v9a_IMPLEMENTATION_LOG.md`.
Full analysis: `results/STAGE9_v9a_REPORT.md`.

Immunization result:

```text
run: trap_v9a_persistent_bonly_resnet18_cars
primary ImageNet top-1: 64.23%
final RIR: 0.783
```

Probe matrix:

| Probe | Baseline final | v9a final | RFD | Signed RFD |
|---|---:|---:|---:|---:|
| linear_probe | 41.23% | 42.48% | 5.11 | -4.46 |
| lora_bonly_r8 | 51.19% | 51.18% | 0.32 | 0.02 |
| lora_r8 | 73.85% | 73.32% | 0.39 | -0.08 |
| lora_r32 | 78.15% | 78.76% | 0.41 | 0.07 |
| full_ft_upper | 80.76% | 79.72% | 0.68 | -0.16 |

Conclusion: single persistent-attacker blocking does not generalize to fresh
post-hoc attackers. The model can make one live B/head attack state bad during
immunization, but it does not remove the broader adapter-reachable Cars
information. Next direction is population/replay persistent attackers or
representation-level tangent label-alignment.

Diagnostic: `results/STAGE9_DIAGNOSTIC_v9a_attacker.md`.

## Stage 10 v10a population-attacker result

Plan: `results/STAGE10_PLAN_population_bonly.md`.
Full analysis: `results/STAGE10_v10a_REPORT.md`.

Immunization result:

```text
run: trap_v10a_population_bonly_resnet18_cars
primary ImageNet top-1: 56.54%
final RIR: 0.657
```

Probe matrix:

| Probe | Baseline final | v10a final | RFD | Signed RFD |
|---|---:|---:|---:|---:|
| linear_probe | 41.23% | 43.02% | 4.94 | -2.82 |
| lora_bonly_r8 | 51.19% | 50.75% | 2.48 | 2.48 |
| lora_r8 | 73.85% | 74.22% | 0.98 | 0.32 |
| lora_r32 | 78.15% | 78.87% | 0.81 | 0.50 |
| full_ft_upper | 80.76% | 80.05% | 0.75 | -0.01 |

Conclusion: population attacker pressure slightly delays B-only adaptation, but
does not block full LoRA and collapses primary utility. This argues against
continuing to scale CE-threshold trajectory traps. Next direction should target
adapter-reachable label information directly.

## Stage 11 learned LoRA-tangent diagnostic

Plan: `results/STAGE11_PLAN_learned_tangent_label_leakage.md`.
Full analysis: `results/STAGE11_learned_tangent_REPORT.md`.

Runs:

```text
learned_tangent_baseline_r8_k3_e10_p100
learned_tangent_v9a_r8_k3_e10_p100
learned_tangent_v10a_r8_k3_e10_p100
learned_tangent_v12a_r8_k3_e10_p100
```

Probe matrix:

| Model | Base final | Learned tangent final | Base + tangent final |
|---|---:|---:|---:|
| baseline | 33.96% | 29.14% | 32.36% |
| v9a | 33.62% | 26.03% | 30.36% |
| v10a | 31.99% | 25.52% | 29.97% |
| v12a | 32.88% | 27.93% | 31.69% |

Conclusion: learned LoRA-tangent features alone recover Cars labels at
25-29%. v9a/v10a reduce this by only 3-4pp; v12a reduces it by only 1.21pp.
These defenses do not remove the adapter-reachable derivative-level signal that
supports LoRA adaptation.

## Stage 12 v12a tangent-removal result

Plan: `results/STAGE12_PLAN_tangent_label_removal.md`.
Full analysis: `results/STAGE12_v12a_REPORT.md`.

Immunization result:

```text
run: trap_v12a_tangent_removal_resnet18_cars
primary ImageNet top-1: 68.36%
final RIR: 1.111
```

Probe matrix:

| Probe | Baseline final | v12a final | RFD | Signed RFD |
|---|---:|---:|---:|---:|
| linear_probe | 41.23% | 40.70% | 4.03 | 3.98 |
| lora_bonly_r8 | 51.19% | 51.08% | 0.66 | 0.66 |
| lora_r8 | 73.85% | 73.85% | 0.37 | 0.06 |
| lora_r32 | 78.15% | 78.57% | 0.40 | 0.23 |
| full_ft_upper | 80.76% | 80.15% | 0.58 | 0.07 |

Conclusion: v12a preserves primary utility and directly trains against a live
learned-tangent readout, but it does not suppress fresh post-hoc LoRA or learned
tangent probes. This strengthens the claim that the Cars signal is broad in the
local adapter tangent kernel, not concentrated in a small live attacker
population.

## Conventions

- `results/<run_name>/results.json` always carries: `config`, `provenance`, `final_*`, `history`.
- `results/<run_name>/slurm.{out,err}` captured at end of SLURM script (since 2026-05-02).
- `results/<run_name>/extractor.pt` only for immunization runs (state dict for `lower` + `upper`).
- Configs under `configs/` are git-tracked; their headers carry attempt-by-attempt notes.
