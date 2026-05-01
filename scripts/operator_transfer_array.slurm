#!/bin/bash
#SBATCH --job-name=op_transfer
#SBATCH --account=llmalignment
#SBATCH --partition=l40s_normal_q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=logs/op_transfer_%A_%a.out
#SBATCH --error=logs/op_transfer_%A_%a.err
#SBATCH --array=0-7

set -euo pipefail
echo "Started: $(date) | task=$SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export PATH=~/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
export HF_HOME=/scratch/jaisharma/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/jaisharma/hf_datasets
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

cd ~/trapping-method
mkdir -p logs results

# 8 tasks total:
#   0–3 = baselines (no immunization)  × {lora_r8, lora_r32, full_ft_upper, full_ft_all}
#   4–7 = Stage-4.5 immunized backbone  × same 4 adversaries
ADVERSARIES=(lora_r8 lora_r32 full_ft_upper full_ft_all)
ADV_IDX=$((SLURM_ARRAY_TASK_ID % 4))
ADV=${ADVERSARIES[$ADV_IDX]}

if [ "$SLURM_ARRAY_TASK_ID" -lt 4 ]; then
    RUN_NAME="adv_baseline_${ADV}_cars"
    CKPT_ARG=""
else
    RUN_NAME="adv_immunized_45_${ADV}_cars"
    CKPT_ARG="--extractor-checkpoint results/trap_paper_faithful_resnet18_cars/extractor.pt"
fi

PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 ~/miniconda3/envs/trap/bin/python \
    experiments/run_adversary.py \
    --config configs/default.yaml \
    --dataset cars \
    --adversary-type "$ADV" \
    --run-name "$RUN_NAME" \
    $CKPT_ARG

echo "Finished: $(date) | task=$SLURM_ARRAY_TASK_ID ($RUN_NAME)"
