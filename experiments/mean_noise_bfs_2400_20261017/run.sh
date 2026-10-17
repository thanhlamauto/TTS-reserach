#!/usr/bin/env bash
set -euo pipefail
MEAN_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
MEAN_SOURCE="$MEAN_REPO/experiments/mean_noise_bfs_2400_20261017"
MEAN_ROOT="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["root"])' "$MEAN_SOURCE/protocol.json")"
mkdir -p "$MEAN_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$MEAN_SOURCE:$MEAN_REPO/experiments/local_noise_avoidance_20261017:$MEAN_REPO/experiments/ot_partial_refresh_20261017:$MEAN_REPO/ot_tts_lab_v2_reimpl:/home/thanhlamtba31/TTS-reserach"
export PJRT_DEVICE=TPU TPU_VISIBLE_CHIPS=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
MEAN_REMAINING="$(python3 - "$MEAN_ROOT" <<'PY'
import pathlib,json,time,sys,fcntl
root=pathlib.Path(sys.argv[1])
with (root/'deadline.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 path=root/'deadline.json'
 if not path.exists():
  start=time.time();path.write_text(json.dumps(dict(started_unix=start,deadline_unix=start+2400,wall_cap_seconds=2400)))
 print(max(0,int(json.loads(path.read_text())['deadline_unix']-time.time())))
PY
)"
if [[ "$MEAN_REMAINING" -lt 1 ]]; then exit 75; fi
exec timeout --signal=TERM --kill-after=30s "${MEAN_REMAINING}s" /home/thanhlamtba31/static-bfs-dcs/venv/bin/python -B -u -m mean_bfs.campaign --protocol "$MEAN_SOURCE/protocol.json"
