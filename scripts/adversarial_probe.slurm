#!/bin/bash
#SBATCH --job-name=adv_probe
#SBATCH --account=llmalignment
#SBATCH --partition=l40s_normal_q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=logs/adv_probe_%j.out
#SBATCH --error=logs/adv_probe_%j.err

set -euo pipefail
echo "Started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export PATH=~/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
export HF_HOME=/scratch/jaisharma/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/jaisharma/hf_datasets
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

cd ~/trapping-method
mkdir -p logs results

# args: dataset (default cars) and extractor_checkpoint (default the CN immunized one)
DATASET=${1:-cars}
CKPT=${2:-results/cn_immunize_resnet18_cars/extractor.pt}
RUN=${3:-immunized_probe_${DATASET}_cn}

PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 ~/miniconda3/envs/trap/bin/python experiments/run_baseline_probe.py \
    --config configs/default.yaml \
    --dataset "$DATASET" \
    --extractor-checkpoint "$CKPT" \
    --run-name "$RUN"

echo "Finished: $(date)"
