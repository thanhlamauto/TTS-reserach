#!/usr/bin/env bash
set -euo pipefail
ETA_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ETA_SOURCE="$ETA_REPO/experiments/covariance_eta16_20261021"
ETA_ROOT="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["root"])' "$ETA_SOURCE/protocol.json")"
mkdir -p "$ETA_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ETA_SOURCE:$ETA_REPO/experiments/covariance_repair_20261021:$ETA_REPO/experiments/local_noise_avoidance_20261017:$ETA_REPO/experiments/ot_partial_refresh_20261017:$ETA_REPO/ot_tts_lab_v2_reimpl:/home/thanhlamtba31/TTS-reserach"
export PJRT_DEVICE=TPU TPU_VISIBLE_CHIPS=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR="$ETA_ROOT/tmp"
mkdir -p "$TMPDIR"
ETA_LEFT="$(python3 - "$ETA_ROOT" <<'PY'
import pathlib,json,time,sys,fcntl
root=pathlib.Path(sys.argv[1])
with (root/'deadline.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);path=root/'deadline.json'
 if not path.exists():
  start=time.time();path.write_text(json.dumps(dict(started_unix=start,deadline_unix=start+900,wall_cap_seconds=900)))
 print(max(0,int(json.loads(path.read_text())['deadline_unix']-time.time())))
PY
)"
if [[ "$ETA_LEFT" -lt 1 ]]; then exit 75; fi
exec timeout --signal=TERM --kill-after=30s "${ETA_LEFT}s" /home/thanhlamtba31/static-bfs-dcs/venv/bin/python -B -u -m eta16.campaign --protocol "$ETA_SOURCE/protocol.json"
