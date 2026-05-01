#!/bin/bash
#SBATCH --job-name=cn_immunize
#SBATCH --account=llmalignment
#SBATCH --partition=l40s_normal_q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/cn_immunize_%j.out
#SBATCH --error=logs/cn_immunize_%j.err

set -euo pipefail

echo "Started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export PATH=~/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh

# HF cache → scratch (home is small)
export HF_HOME=/scratch/jaisharma/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/jaisharma/hf_datasets
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

cd ~/trapping-method
mkdir -p logs results

CONFIG=${1:-configs/immunize_cn.yaml}
PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 ~/miniconda3/envs/trap/bin/python experiments/run_immunization_cn.py --config "$CONFIG"

echo "Finished: $(date)"
