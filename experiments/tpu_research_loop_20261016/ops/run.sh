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
  --root "$ROOT" --phase SD15_smoke --max-seconds 600 \
  --entrypoint experiments/tpu_research_loop_20261016/reward_tpu_gate.py \
  -- --root "$ROOT" \
  --golden-manifest experiments/tpu_research_loop_20261016/reward_golden_manifest.json
