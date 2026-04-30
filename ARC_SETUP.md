# Virginia Tech ARC Cluster Guide

Comprehensive guide for using VT's Advanced Research Computing (ARC) clusters for ML/DL research.

## Clusters Overview

| Cluster | Login Node | Best For | SSH Command |
|---------|-----------|----------|-------------|
| **Falcon** | `falcon1.arc.vt.edu` | GPU-intensive DL (L40S, A30) | `ssh falcon1.arc.vt.edu` |
| **TinkerCliffs** | `tinkercliffs1.arc.vt.edu` | Large-model training (A100, H200) | `ssh tinkercliffs1.arc.vt.edu` |

**Account:** `llmalignment`
**User:** `jaishrm`

Both clusters share the **same home directory** (`$HOME = /home/jaishrm/`), so files synced to one are visible on the other.

---

## Falcon Cluster

### GPU Partitions

| Partition | GPU | VRAM | GPUs/Node | CPUs/Node | RAM/Node | Nodes | Total GPUs |
|-----------|-----|------|-----------|-----------|----------|-------|------------|
| `l40s_normal_q` | NVIDIA L40S | 48 GB | 4 | 64 | 504 GB | 20 | 80 |
| `a30_normal_q` | NVIDIA A30 | 24 GB | 4 | 64 | 504 GB | 32 | 128 |
| `v100_normal_q` | NVIDIA V100 | 32 GB | varies | varies | varies | 36 | varies |
| `t4_normal_q` | NVIDIA T4 | 16 GB | varies | varies | varies | 18 | varies |

Each partition also has a `_preemptable_q` variant (same hardware, can be preempted by normal jobs — useful for longer/lower-priority work).

**Recommended for most work:** `l40s_normal_q` — best balance of VRAM (48 GB), speed, and availability.

### Falcon Quick Start

```bash
# SSH in
ssh falcon1.arc.vt.edu

# Setup environment (first time)
export PATH=~/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pcb   # or whatever environment

# Navigate to project
cd ~/pcb-DDetect

# Submit a job
sbatch scripts/my_job.slurm

# Check your jobs
squeue -u jaishrm

# Check partition availability
sinfo -p l40s_normal_q
```

---

## TinkerCliffs Cluster

### GPU Partitions

| Partition | GPU | VRAM | GPUs/Node | CPUs/Node | RAM/Node | Nodes | Total GPUs |
|-----------|-----|------|-----------|-----------|----------|-------|------------|
| `a100_normal_q` | NVIDIA A100-SXM4 | 80 GB | 8 | 128 | 2 TB | 14 | 112 |
| `h200_normal_q` | NVIDIA H200 | 141 GB | 8 | varies | varies | 6 | 48 |

Also has `_preemptable_q` variants and a large CPU-only `normal_q` partition (312 nodes).

**A100 nodes** (`tc-dgx001` to `tc-dgx010`, `tc-gpu001` to `tc-gpu004`): Best for large models, multi-GPU training.
**H200 nodes** (`tc-xe001` to `tc-xe006`): Newest, most powerful — use for the biggest workloads.

### TinkerCliffs Quick Start

```bash
# SSH in
ssh tinkercliffs1.arc.vt.edu

# Same conda setup (shared home dir)
export PATH=~/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pcb

cd ~/pcb-DDetect
sbatch scripts/my_job.slurm
```

---

## SLURM Job Submission

### Basic SLURM Script Template

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --account=llmalignment
#SBATCH --partition=l40s_normal_q          # or a100_normal_q on TinkerCliffs
#SBATCH --gres=gpu:1                       # 1 GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00                     # max walltime
#SBATCH --output=logs/my_job_%j.out
#SBATCH --error=logs/my_job_%j.err

echo "Started: $(date)"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

export PATH=~/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pcb
cd ~/pcb-DDetect

python src/train.py --arg1 val1

echo "Finished: $(date)"
```

### Array Jobs (Parallel Execution)

Run multiple independent tasks in parallel — each gets its own GPU:

```bash
#!/bin/bash
#SBATCH --job-name=exp_%a
#SBATCH --account=llmalignment
#SBATCH --partition=l40s_normal_q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=logs/exp_%a_%j.out
#SBATCH --error=logs/exp_%a_%j.err
#SBATCH --array=0-5                        # 6 parallel tasks (indices 0-5)

# Map array index to experiment config
DATASETS=(deeppcb dspcbsd pku mendeley gc10 neudet)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# Each task gets its own GPU and runs independently
python src/train.py --dataset ${DATASET}
```

**Array index variable:** `$SLURM_ARRAY_TASK_ID` (0, 1, 2, ... N)

### Multi-GPU Jobs

```bash
#SBATCH --gres=gpu:4                       # 4 GPUs on one node
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

# Use torchrun for distributed training
torchrun --nproc_per_node=4 src/train.py
```

---

## Interactive Sessions

For debugging, prototyping, or quick tests:

```bash
# Falcon — L40S interactive (4 hours)
interact -A llmalignment -t 4:00:00 -p l40s_normal_q --gres=gpu:1 --mem=48G

# TinkerCliffs — A100 interactive (2 hours)
interact -A llmalignment -t 2:00:00 -p a100_normal_q --gres=gpu:1 --mem=48G

# Once on the GPU node:
export PATH=~/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pcb
cd ~/pcb-DDetect
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Environment Setup (One-Time)

### Install Miniconda

```bash
# On either cluster (shared home dir)
ssh falcon1.arc.vt.edu
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh
```

### Create Conda Environment

```bash
export PATH=~/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh

# For PCB defect detection / YOLO work
conda create -n pcb python=3.10 -y
conda activate pcb
pip install ultralytics==8.3.65 pandas matplotlib seaborn scipy

# For LLM work
conda create -n llm python=3.10 -y
conda activate llm
pip install torch transformers bitsandbytes accelerate peft
```

### Clone Repos

```bash
cd ~
git clone git@github.com:jaishrm07/pcb-DDetect.git
# or
git clone https://github.com/jaishrm07/pcb-DDetect.git
```

---

## Storage

| Path | Purpose | Quota | Shared? |
|------|---------|-------|---------|
| `$HOME/` (`/home/jaishrm/`) | Code, configs, small files | ~50 GB | Yes (both clusters) |
| `/scratch/jaishrm/` | Large datasets, model weights, HF cache | 1.4 PB (shared) | Cluster-specific |
| `/projects/llmalignment/` | Shared project storage | Varies | Group-shared |

**Important:** Home directory is shared between Falcon and TinkerCliffs. Scratch is NOT shared — each cluster has its own `/scratch`.

### HuggingFace Cache

```bash
export HF_HOME=/scratch/jaishrm/.cache/huggingface
```

---

## Monitoring & Management

### Job Status

```bash
# Your running/pending jobs
squeue -u jaishrm

# Detailed format
squeue -u jaishrm --format="%.12i %.12P %.10j %.8T %.10M %.4D %R"

# Specific job
squeue -j <JOBID>

# Auto-refresh every 30s
watch -n 30 squeue -u jaishrm
```

### Job History & Accounting

```bash
# Job details after completion
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS

# All recent jobs
sacct -u jaishrm --starttime=2026-04-01 --format=JobID,JobName,State,Elapsed

# Job efficiency (CPU/mem utilization)
seff <JOBID>
```

### Logs

```bash
# Live output
tail -f logs/my_job_<JOBID>.out

# Error log
cat logs/my_job_<JOBID>.err

# Array job logs (task 3)
tail -f logs/my_job_3_<JOBID>.out
```

### Cancel Jobs

```bash
# Cancel specific job
scancel <JOBID>

# Cancel all your jobs
scancel -u jaishrm

# Cancel specific array task
scancel <JOBID>_3

# Cancel all pending jobs only
scancel -u jaishrm --state=PENDING
```

### Partition Info

```bash
# Summary of all partitions
sinfo -s

# Detailed node info for a partition
sinfo -p l40s_normal_q -N --format="%N %G %m %c %T"

# GPU availability
sinfo -p l40s_normal_q --format="%P %a %D %G %T"
```

---

## Syncing Code Between Local and Clusters

Since both clusters share the same home directory, you only need to sync once:

```bash
# From local Mac to cluster
scp src/train.py falcon1.arc.vt.edu:~/pcb-DDetect/src/train.py

# Multiple files
scp src/modules.py src/train_recipe_study.py falcon1.arc.vt.edu:~/pcb-DDetect/src/

# Sync results back to local
rsync -avz falcon1.arc.vt.edu:~/pcb-DDetect/results/ /local/path/results/

# Full directory sync (careful with large model weights)
rsync -avz --exclude='*.pt' --exclude='__pycache__' \
    falcon1.arc.vt.edu:~/pcb-DDetect/ /local/path/pcb-DDetect/
```

### Using Both Clusters in Parallel

Submit the same experiment to both clusters for faster completion:

```bash
# Upload code once (shared home dir)
scp scripts/my_job.slurm falcon1.arc.vt.edu:~/pcb-DDetect/scripts/

# Submit on Falcon (L40S)
ssh falcon1.arc.vt.edu 'cd ~/pcb-DDetect && sbatch scripts/my_job_falcon.slurm'

# Submit on TinkerCliffs (A100)
ssh tinkercliffs1.arc.vt.edu 'cd ~/pcb-DDetect && sbatch scripts/my_job_tc.slurm'
```

**Note:** If both clusters write to the same output directory, use the `is_complete()` skip check in your training script to avoid duplicate runs (Ultralytics checks for existing `best.pt`).

---

## Choosing the Right Cluster & Partition

| Task | Recommended | Why |
|------|------------|-----|
| YOLOv8s/n training (single GPU) | Falcon `l40s_normal_q` | Fast, 48GB VRAM, good availability |
| YOLOv8l/x training | TinkerCliffs `a100_normal_q` | 80GB VRAM for large models |
| LLM fine-tuning (7B-13B) | TinkerCliffs `a100_normal_q` | 80GB needed for QLoRA |
| LLM fine-tuning (70B+) | TinkerCliffs `h200_normal_q` | 141GB VRAM |
| Quick prototyping | Falcon `a30_normal_q` | Most nodes, shortest wait |
| Array jobs (6+ parallel) | Falcon `l40s_normal_q` | 80 GPUs, usually available |
| Long runs (>12h) | Either `_preemptable_q` | No walltime limit |
| Batch inference | Falcon `t4_normal_q` | Cheapest, sufficient for inference |

---

## Common SLURM Gotchas

1. **`QOSMaxGRESPerUser`**: You've hit the max GPU quota. Wait for running jobs to finish or cancel some.

2. **Partition not found**: Falcon and TinkerCliffs have different partitions. `l40s_normal_q` is Falcon-only, `a100_normal_q` is TinkerCliffs-only (and Falcon doesn't have A100s).

3. **Job runs but no output**: Make sure `logs/` directory exists: `mkdir -p logs`

4. **Module not found errors**: Always activate conda in your SLURM script:
   ```bash
   export PATH=~/miniconda3/bin:$PATH
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate pcb
   ```

5. **OOM (Out of Memory)**: Reduce batch size, or request more `--mem`. L40S has 48GB VRAM, A100 has 80GB.

6. **Job stuck PENDING**: Check reason with `squeue -j <JOBID> -o "%R"`. Common reasons:
   - `Priority`: Others have higher priority, wait your turn
   - `Resources`: Not enough free GPUs
   - `QOSMaxGRESPerUser`: GPU quota exceeded

7. **Array job directory collision**: When running same experiment on both clusters, the shared home dir means both write to the same paths. Use `is_complete()` checks or separate `--project` dirs.

---

## Current Projects on Clusters

### PCB Defect Detection (`~/pcb-DDetect/`)

```
~/pcb-DDetect/
├── src/                          # Training scripts, modules
│   ├── train_recipe_study.py     # Main experiment script
│   ├── aggregate_results.py      # Results aggregation
│   └── modules.py                # Custom YOLO modules (EMA, CoordAtt, ZRMI, etc.)
├── scripts/                      # SLURM job scripts
│   ├── ema_all_datasets.slurm    # EMA study (Falcon, sequential)
│   ├── ema_tc_parallel.slurm     # EMA study (TinkerCliffs, parallel)
│   ├── a4r1_falcon.slurm         # A4+R1 (Falcon)
│   ├── a4r1_parallel.slurm       # A4+R1 (TinkerCliffs)
│   ├── zrmi_parallel.slurm       # ZRMI v1 (Falcon)
│   ├── zrmi_v2.slurm             # ZRMI v2 (Falcon)
│   └── backup_experiments.sh     # Backup script
├── configs/                      # Dataset YAML configs
├── data/                         # Datasets (symlinked or downloaded)
├── runs/                         # Experiment outputs
│   ├── recipe_study/             # Custom metadata
│   └── detect/runs/recipe_study/ # Ultralytics outputs + weights
├── results/                      # Aggregated tables, figures
├── logs/                         # SLURM stdout/stderr
└── paper/                        # LaTeX paper
```

### Gita LLM (`~/llm-gita/`)

- Environment: `conda activate llm`
- Best partition: `a100_normal_q` (TinkerCliffs)

---

## Useful One-Liners

```bash
# How many GPUs am I using right now?
squeue -u jaishrm -h -o "%D %b" | awk '{sum+=$1} END {print sum " jobs using GPUs"}'

# Which nodes are my jobs on?
squeue -u jaishrm -o "%i %N %P"

# Check GPU utilization on a running node (from login node)
ssh <node> nvidia-smi

# Disk usage in home
du -sh ~/pcb-DDetect/runs/

# Find all best.pt files (completed training runs)
find ~/pcb-DDetect/runs -name "best.pt" | wc -l

# Quick mAP50 extraction from a run
python3 -c "
import csv
with open('runs/detect/runs/recipe_study/<RUN_NAME>/results.csv') as f:
    rows = list(csv.DictReader(f))
    best = max(float(r['   metrics/mAP50(B)'].strip()) for r in rows)
    print(f'Best mAP50: {best:.4f}')
"
```
