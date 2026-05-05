#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:-role-lab}"
REMOTE_DIR="${REMOTE_DIR:-~/trapping-method}"
PY="${PY:-/home/jaisharma/miniconda3/envs/trap/bin/python}"
SMOKE_GPU="${SMOKE_GPU:-0}"
BASELINE_GPU="${BASELINE_GPU:-1}"
V9_GPU="${V9_GPU:-2}"
V10_GPU="${V10_GPU:-3}"

echo "[stage11] syncing files to ${REMOTE}:${REMOTE_DIR}"
rsync -avR \
  experiments/run_learned_tangent_probe.py \
  results/STAGE11_PLAN_learned_tangent_label_leakage.md \
  experiments/REGISTRY.md \
  "${REMOTE}:${REMOTE_DIR}/"

echo "[stage11] remote compile"
ssh "${REMOTE}" "cd ${REMOTE_DIR} && ${PY} -m py_compile experiments/run_learned_tangent_probe.py"

echo "[stage11] remote smoke"
ssh "${REMOTE}" "cd ${REMOTE_DIR} && mkdir -p logs && CUDA_VISIBLE_DEVICES=${SMOKE_GPU} PYTHONUNBUFFERED=1 ${PY} experiments/run_learned_tangent_probe.py \
  --dataset cars \
  --num-workers 0 \
  --train-limit 128 \
  --test-limit 128 \
  --feature-batch-size 16 \
  --probe-batch-size 32 \
  --num-directions 1 \
  --direction-epochs 1 \
  --probe-epochs 2 \
  --run-name smoke_learned_tangent_cars \
  > logs/smoke_learned_tangent_cars.out \
  2> logs/smoke_learned_tangent_cars.err"

echo "[stage11] launching full diagnostics"
ssh "${REMOTE}" "cd ${REMOTE_DIR} && mkdir -p logs"

echo "[stage11] launching baseline on GPU ${BASELINE_GPU}"
ssh "${REMOTE}" "cd ${REMOTE_DIR} && CUDA_VISIBLE_DEVICES=${BASELINE_GPU} PYTHONUNBUFFERED=1 nohup setsid -f ${PY} experiments/run_learned_tangent_probe.py \
    --dataset cars \
    --rank 8 \
    --num-directions 3 \
    --direction-epochs 10 \
    --probe-epochs 100 \
    --run-name learned_tangent_baseline_r8_k3_e10_p100 \
    > logs/learned_tangent_baseline_r8_k3_e10_p100.out \
    2> logs/learned_tangent_baseline_r8_k3_e10_p100.err \
    < /dev/null"

echo "[stage11] launching v9a on GPU ${V9_GPU}"
ssh "${REMOTE}" "cd ${REMOTE_DIR} && CUDA_VISIBLE_DEVICES=${V9_GPU} PYTHONUNBUFFERED=1 nohup setsid -f ${PY} experiments/run_learned_tangent_probe.py \
    --dataset cars \
    --extractor-checkpoint results/trap_v9a_persistent_bonly_resnet18_cars/extractor.pt \
    --rank 8 \
    --num-directions 3 \
    --direction-epochs 10 \
    --probe-epochs 100 \
    --run-name learned_tangent_v9a_r8_k3_e10_p100 \
    > logs/learned_tangent_v9a_r8_k3_e10_p100.out \
    2> logs/learned_tangent_v9a_r8_k3_e10_p100.err \
    < /dev/null"

echo "[stage11] launching v10a on GPU ${V10_GPU}"
ssh "${REMOTE}" "cd ${REMOTE_DIR} && CUDA_VISIBLE_DEVICES=${V10_GPU} PYTHONUNBUFFERED=1 nohup setsid -f ${PY} experiments/run_learned_tangent_probe.py \
    --dataset cars \
    --extractor-checkpoint results/trap_v10a_population_bonly_resnet18_cars/extractor.pt \
    --rank 8 \
    --num-directions 3 \
    --direction-epochs 10 \
    --probe-epochs 100 \
    --run-name learned_tangent_v10a_r8_k3_e10_p100 \
    > logs/learned_tangent_v10a_r8_k3_e10_p100.out \
    2> logs/learned_tangent_v10a_r8_k3_e10_p100.err \
    < /dev/null"

echo "[stage11] launched baseline/v9a/v10a diagnostics"
