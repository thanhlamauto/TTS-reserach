#!/usr/bin/env bash
set -euo pipefail
export PJRT_DEVICE=TPU
export TPU_VISIBLE_CHIPS=0
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
ROOT=/home/thanhlamtba31/static-bfs-dcs/experiments/tpu_research_loop_20261016
PY=/home/thanhlamtba31/static-bfs-dcs/venv/bin/python
exec "$PY" experiments/tpu_research_loop_20261016/budget_guard.py \
  --root "$ROOT" --phase exploration --max-seconds 3600 \
  --entrypoint experiments/tpu_research_loop_20261016/paired_pilot.py \
  -- --root "$ROOT"
