#!/bin/bash
#SBATCH --job-name=baseline_probe
#SBATCH --account=llmalignment
#SBATCH --partition=l40s_normal_q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=logs/baseline_probe_%j.out
#SBATCH --error=logs/baseline_probe_%j.err

set -euo pipefail

echo "Started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export PATH=~/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trap

# HuggingFace cache → scratch (home is small)
export HF_HOME=/scratch/jaishrm/.cache/huggingface
mkdir -p "$HF_HOME"

cd ~/trapping-immunization

mkdir -p logs results

# Default: Cars. Override by passing the dataset name as the first arg.
DATASET=${1:-cars}

python experiments/run_baseline_probe.py \
    --config configs/default.yaml \
    --dataset "$DATASET"

echo "Finished: $(date)"
