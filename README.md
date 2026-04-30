# Trapping Paper Reimplementation

From-scratch reimplementation of *Model Immunization by Trapping Harmful Finetuning* (Sarker, Hakim, Ishmam, Tang, Thomas — NeurIPS 2025 Lock-LLM Workshop).

Started April 30, 2026. Goal: deeply understand the paper by re-deriving and re-implementing every piece, and reproduce the headline result on ResNet18 / Cars (RFD ≈ 47.19, primary task accuracy ≈ 65.99%).

This is not a research extension yet. It is a learning exercise that becomes the foundation for Phase-2 work (operator-transfer evaluation) — see `../ms_strategy.md`.

## Reading order before coding

Read in this order before starting any implementation. Skipping these will cost more time than it saves.

1. `../papers/trapping_thomas_neurips_lock_llm_2025.pdf` — the paper itself, all 10 pages.
2. `../papers/trapping_paper_summary.md` — the structural notes already written.
3. Zheng et al., *Model Immunization from a Condition Number Perspective*, ICML 2025 — needed for the `R_well` / `R_ill` regularizer formulas. **Not yet downloaded; do this first.**
4. The `R_well` / `R_ill` formulas use a *feature-covariance approximation* of the Hessian. Make sure you understand why this approximation is valid before trying to reproduce it.

## Implementation order (staged, do not skip)

The order matters. Each stage validates the foundation for the next. If a stage's sanity check fails, do not move to the next — debug first.

### Stage 0 — Environment and data (1–2 days)

- Set up `pyproject.toml` / `requirements.txt` with `torch`, `torchvision`, `timm`, `transformers`, `wandb`, `einops`, `pyhessian`.
- Load ImageNet-pretrained ResNet18 from `torchvision`.
- Load Stanford Cars, Food101, Country211 datasets via `torchvision.datasets` or HuggingFace.
- Build a basic train/eval loop for *non-immunized* linear probing on Cars. Confirm you reach the paper's baseline accuracy (~67% test on Cars with ResNet18 + linear probing).

**Sanity check:** `experiments/run_baseline_probe.py` reproduces the "Init θ_0" row from Table 1: 67.04% on Cars, 65.41% on Food101, 67.33% on Country211 (ResNet18 column).

### Stage 1 — RFD metric (1 day)

- Implement RFD per Eq. 9 of the paper.
- Apply it to the baseline probe (RFD vs itself = 0; RFD vs a slightly corrupted model > 0).

**Sanity check:** RFD of model-vs-itself is exactly 0. RFD of model-vs-randomly-perturbed-features is positive.

### Stage 2 — Condition-number regularizers (3–4 days, the first hard piece)

- Read Zheng et al. (ICML 2025) for the Hessian approximation via feature covariance.
- Implement `R_well(H_P)` (Eq. 2) and `R_ill(H_H)` (Eq. 3).
- Run *condition-number-only* immunization (no trap loss yet). This is the "CN" baseline in Table 1.
- Should produce RIR ≈ 3.5 and RFD ≈ 10.06 on Cars / ResNet18.

**Sanity check:** the "CN" row of Table 1 reproduces within ~5% on Cars / ResNet18.

### Stage 3 — Trap-inducing loss (4–7 days, the hardest piece)

- Implement the k-step adversary unroll inside the immunization training loop. This needs:
  - K-NN initialization of the harmful classifier head ω_H (paper Appendix B).
  - Differentiable inner-loop optimization for k steps.
  - Tracking of `ΔL_act` and `ΔL_exp` (per Eq. 4 and 5).
  - `softplus(ΔL_act − ΔL_exp)` as the trap loss (Eq. 6).

The tricky part: gradients must flow through the k-step inner loop back to the immunization parameters θ. Use `torch.func.functional_call` or `higher` library for clean nested optimization.

**Sanity check:** trap-loss-only immunization (no CN regularizers) gives RIR ≈ 1.07 — almost no immunization signal. This matches Table 4 and validates that trap-only is intentionally weak.

### Stage 4 — Combined immunization (2–3 days)

- Combine trap loss + CN regularizers + primary task loss per Eq. 7.
- Tune `λ_trap`, `λ_R_well`, `λ_R_ill` per Table 3.
- Reproduce the "Ours" row of Table 1.

**Sanity check:** RFD ≈ 47 and primary accuracy ≈ 66% on Cars / ResNet18.

### Stage 5 — RIR metric and ablations (1–2 days)

- Implement RIR per Eq. 8.
- Reproduce Table 4 (synergy ablation): trap-only, CN-only, both.
- Reproduce Table 2 (batch-size sensitivity) — confirm RFD is more stable than RIR.

### Stage 6 — ViT and other harmful datasets (3–4 days)

- Swap ResNet18 for ViT (timm or torchvision). The Hessian approximation may need rework due to attention layers.
- Run the same pipeline on Food101 and Country211.
- Reproduce the full Table 1.

## Compute requirements

Running on VT ARC Falcon (`l40s_normal_q`, 48 GB L40S) — see `ARC_SETUP.md` for the full cluster reference.

- **Stages 0–2:** a single L40S finishes a baseline linear probe in <30 min and CN-only immunization in 1–2 hours per dataset.
- **Stage 3+:** the k-step adversary unroll multiplies activation memory by ~k. With L40S's 48 GB and ResNet18, k ∈ {3, 5, 10} all fit comfortably. ViT may need lower k or activation checkpointing.
- **Full Table 1 reproduction** (ResNet18 + ViT × 3 datasets, all baselines + ours): ~30 GPU-hours on L40S. Single-node, single-GPU jobs throughout — no DDP needed for the paper's models.

For local debugging on the Mac, the baseline probe runs on MPS (slow). Don't attempt Stage 3+ on Mac.

## Directory layout

```
trapping_experiment/
├── README.md              ← this file
├── pyproject.toml         ← project metadata, deps
├── src/
│   ├── data.py            ← dataset loaders for ImageNet/Cars/Food101/Country211
│   ├── models.py          ← ResNet18 / ViT loading from torchvision/timm
│   ├── losses.py          ← trap loss, R_well, R_ill, primary
│   ├── hessian.py         ← feature-covariance Hessian approximation
│   ├── metrics.py         ← RIR (Eq. 8), RFD (Eq. 9)
│   ├── immunize.py        ← outer immunization training loop
│   ├── attack.py          ← inner adversarial linear probe
│   └── utils.py           ← K-NN head init, logging, seeding
├── configs/
│   └── default.yaml       ← hyperparameters per Table 3
├── experiments/
│   ├── run_baseline_probe.py
│   ├── run_immunization.py
│   ├── run_attack.py
│   └── run_evaluation.py
├── results/               ← logged metrics, plots, checkpoints
└── notebooks/             ← exploratory debugging
```

## Status

- [x] Directory created
- [ ] Reading: Zheng et al. condition-number paper
- [x] Stage 0 — environment and data (code written, not yet validated on ARC)
- [x] Stage 1 — RFD metric (self-test passes; pending live use after Stage 4)
- [ ] Stage 2 — condition-number regularizers
- [ ] Stage 3 — trap-inducing loss
- [ ] Stage 4 — combined immunization
- [ ] Stage 5 — RIR and ablations
- [ ] Stage 6 — ViT and additional datasets

## Running on VT ARC (Falcon, L40S)

One-time setup on `falcon1.arc.vt.edu`:

```bash
ssh falcon1.arc.vt.edu

# Conda env
export PATH=~/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n trap python=3.10 -y
conda activate trap
pip install torch torchvision timm datasets pillow numpy pyyaml tqdm wandb

# Project location
mkdir -p ~/trapping-immunization
# (sync the trapping_experiment/ contents here from local Mac)
```

Sync from Mac to ARC (run from local trapping_experiment/ dir):

```bash
rsync -avz --exclude='results' --exclude='__pycache__' --exclude='.DS_Store' \
    ./ falcon1.arc.vt.edu:~/trapping-immunization/
```

Run the Stage-0 baseline probe:

```bash
ssh falcon1.arc.vt.edu
cd ~/trapping-immunization
mkdir -p logs

# Submit
sbatch scripts/baseline_probe.slurm cars

# Or for an interactive sanity check (faster turnaround):
interact -A llmalignment -t 2:00:00 -p l40s_normal_q --gres=gpu:1 --mem=48G
# Once on the GPU node:
conda activate trap
python experiments/run_baseline_probe.py --config configs/default.yaml --dataset cars
```

**Expected result on Cars / ResNet18:** test accuracy converges to ~67% (matches paper's `Init θ_0` row of Table 1).
