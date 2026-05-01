#!/bin/bash
#SBATCH --job-name=multiop_probe
#SBATCH --account=llmalignment
#SBATCH --partition=l40s_normal_q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=logs/multiop_probe_%A_%a.out
#SBATCH --error=logs/multiop_probe_%A_%a.err
#SBATCH --array=0-4

set -euo pipefail
echo "Started: $(date) | task=$SLURM_ARRAY_TASK_ID"

export PATH=~/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
export HF_HOME=/scratch/jaisharma/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/jaisharma/hf_datasets
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

cd ~/trapping-method
mkdir -p logs results

# 5 tasks: each adversary against the Stage-5 multi-op immunized backbone.
ADVERSARIES=(linear_probe lora_r8 lora_r32 full_ft_upper full_ft_all)
ADV=${ADVERSARIES[$SLURM_ARRAY_TASK_ID]}
RUN_NAME="adv_multiop_${ADV}_cars"

PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 ~/miniconda3/envs/trap/bin/python \
    experiments/run_adversary.py \
    --config configs/default.yaml \
    --dataset cars \
    --adversary-type "$ADV" \
    --extractor-checkpoint results/trap_multiop_lp_lora8_resnet18_cars/extractor.pt \
    --run-name "$RUN_NAME"

echo "Finished: $(date) | task=$SLURM_ARRAY_TASK_ID ($RUN_NAME)"
