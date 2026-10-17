#!/usr/bin/env bash
set -euo pipefail
NOISE_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
NOISE_SOURCE="$NOISE_REPO/experiments/local_noise_radius_20261017"
NOISE_ROOT="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["root"])' "$NOISE_SOURCE/protocol.json")"
mkdir -p "$NOISE_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$NOISE_SOURCE:$NOISE_REPO/experiments/local_noise_avoidance_20261017:$NOISE_REPO/experiments/ot_partial_refresh_20261017:$NOISE_REPO/ot_tts_lab_v2_reimpl:/home/thanhlamtba31/TTS-reserach"
export PJRT_DEVICE=TPU TPU_VISIBLE_CHIPS=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
NOISE_REMAINING="$(python3 - "$NOISE_ROOT" <<'PY'
import pathlib,json,time,sys,fcntl
root=pathlib.Path(sys.argv[1])
with (root/'deadline.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 path=root/'deadline.json'
 if not path.exists():
  start=time.time();path.write_text(json.dumps(dict(started_unix=start,deadline_unix=start+1200,wall_cap_seconds=1200)))
 print(max(0,int(json.loads(path.read_text())['deadline_unix']-time.time())))
PY
)"
if [[ "$NOISE_REMAINING" -lt 1 ]]; then exit 75; fi
exec timeout --signal=TERM --kill-after=30s "${NOISE_REMAINING}s" /home/thanhlamtba31/static-bfs-dcs/venv/bin/python -B -u -m noise_radius.campaign --protocol "$NOISE_SOURCE/protocol.json"
