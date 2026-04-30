#!/bin/bash
#SBATCH --job-name=cn_immunize
#SBATCH --account=llmalignment
#SBATCH --partition=l40s_normal_q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/cn_immunize_%j.out
#SBATCH --error=logs/cn_immunize_%j.err

set -euo pipefail

echo "Started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export PATH=~/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trap

export HF_HOME=/scratch/jaisharma/.cache/huggingface
mkdir -p "$HF_HOME"

cd ~/trapping-immunization
mkdir -p logs results

CONFIG=${1:-configs/immunize_cn.yaml}
python experiments/run_immunization_cn.py --config "$CONFIG"

echo "Finished: $(date)"
