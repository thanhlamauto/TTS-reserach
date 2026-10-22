#!/usr/bin/env bash
set -Eeuo pipefail
stage="${1:?stage required}"
run_root="${2:?absolute output directory required}"
case "$stage" in compare|focused|heldout) ;; *) exit 2 ;; esac
repo=/home/thanhlamtba31/TTS-reserach
py=/home/thanhlamtba31/static-bfs-dcs/venv/bin/python
for chip in 0 1 2 3; do
  if sudo -n fuser "/dev/vfio/$chip" >/dev/null 2>&1; then
    printf 'chip %s is occupied; refusing duplicate launch\n' "$chip" >&2
    exit 1
  fi
done
for chip in 0 1 2 3; do
  bash "$repo/sd15_sinkhorn_tpu/managed_job.sh" launch "sinkhorn-$stage-chip$chip" \
    "$repo" "$run_root/jobs" -- env TPU_VISIBLE_CHIPS="$chip" PJRT_DEVICE=TPU \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$py" -u -m sd15_sinkhorn_tpu.sweep --stage "$stage" --output "$run_root" \
    --workers 4 --rank "$chip"
done
