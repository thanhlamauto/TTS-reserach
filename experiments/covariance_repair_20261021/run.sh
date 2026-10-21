#!/usr/bin/env bash
set -euo pipefail
COV_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
COV_SOURCE="$COV_REPO/experiments/covariance_repair_20261021"
COV_ROOT="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["root"])' "$COV_SOURCE/protocol.json")"
mkdir -p "$COV_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$COV_SOURCE:$COV_REPO/experiments/mean_noise_bfs_2400_20261017:$COV_REPO/experiments/local_noise_avoidance_20261017:$COV_REPO/experiments/ot_partial_refresh_20261017:$COV_REPO/ot_tts_lab_v2_reimpl:/home/thanhlamtba31/TTS-reserach"
export PJRT_DEVICE=TPU TPU_VISIBLE_CHIPS=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR="$COV_ROOT/tmp"
mkdir -p "$TMPDIR"
COV_REMAINING="$(python3 - "$COV_ROOT" <<'PY'
import pathlib,json,time,sys,fcntl
root=pathlib.Path(sys.argv[1])
with (root/'deadline.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 path=root/'deadline.json'
 if not path.exists():
  start=time.time();path.write_text(json.dumps(dict(started_unix=start,deadline_unix=start+1800,wall_cap_seconds=1800)))
 print(max(0,int(json.loads(path.read_text())['deadline_unix']-time.time())))
PY
)"
if [[ "$COV_REMAINING" -lt 1 ]]; then exit 75; fi
exec timeout --signal=TERM --kill-after=30s "${COV_REMAINING}s" /home/thanhlamtba31/static-bfs-dcs/venv/bin/python -B -u -m cov_repair.campaign --protocol "$COV_SOURCE/protocol.json"
