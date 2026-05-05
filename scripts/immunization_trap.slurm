#!/bin/bash
#SBATCH --job-name=trap_immune
#SBATCH --account=llmalignment
#SBATCH --partition=l40s_normal_q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --output=logs/trap_immunize_%j.out
#SBATCH --error=logs/trap_immunize_%j.err

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

CONFIG=${1:-configs/immunize_trap.yaml}
PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 ~/miniconda3/envs/trap/bin/python experiments/run_immunization_cn.py --config "$CONFIG"

# Archive SLURM logs into the run dir so they travel with results.json.
RUN_NAME=$(grep -E '^run_name:' "$CONFIG" | awk '{print $2}')
RUN_DIR="results/$RUN_NAME"
if [ -n "$RUN_NAME" ] && [ -d "$RUN_DIR" ]; then
    cp "logs/trap_immunize_${SLURM_JOB_ID}.out" "$RUN_DIR/slurm.out" 2>/dev/null || true
    cp "logs/trap_immunize_${SLURM_JOB_ID}.err" "$RUN_DIR/slurm.err" 2>/dev/null || true
fi

echo "Finished: $(date)"
