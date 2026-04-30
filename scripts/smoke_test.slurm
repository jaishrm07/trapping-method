#!/bin/bash
#SBATCH --job-name=trap_smoke
#SBATCH --account=llmalignment
#SBATCH --partition=l40s_normal_q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:15:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

set -euo pipefail
echo "Started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd ~/trapping-immunization
mkdir -p logs
PYTHONNOUSERSITE=1 ~/miniconda3/envs/trap/bin/python experiments/smoke_test_stage2.py
echo "Finished: $(date)"
