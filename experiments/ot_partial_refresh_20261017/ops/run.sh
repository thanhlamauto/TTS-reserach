#!/usr/bin/env bash
set -euo pipefail
OTNEXT_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
OTNEXT_SOURCE="$OTNEXT_REPO/experiments/ot_partial_refresh_20261017"
OTNEXT_ROOT=/home/thanhlamtba31/static-bfs-dcs/ot_partial_refresh_20261017
mkdir -p "$OTNEXT_ROOT"
export PYTHONPATH="$OTNEXT_SOURCE:$OTNEXT_REPO/ot_tts_lab_v2_reimpl:/home/thanhlamtba31/TTS-reserach"
export PJRT_DEVICE=TPU
export TPU_VISIBLE_CHIPS=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# Persistent deadline survives implementation repairs; no retry resets the cap.
OTNEXT_REMAINING="$(python3 - "$OTNEXT_ROOT" <<'PY'
import json,pathlib,sys,time,os,fcntl
root=pathlib.Path(sys.argv[1])
with (root/'deadline.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 path=root/'deadline.json'
 if not path.exists():
  start=time.time()
  path.write_text(json.dumps(dict(started_unix=start,deadline_unix=start+3600,wall_cap_seconds=3600)))
 clock=json.loads(path.read_text())
 print(max(0,int(clock['deadline_unix']-time.time())))
PY
)"
if [[ "$OTNEXT_REMAINING" -lt 1 ]]; then
  echo 'Persistent campaign cap exhausted; no neural retry authorized.' >&2
  exit 75
fi
exec timeout --signal=TERM --kill-after=30s "${OTNEXT_REMAINING}s" \
  /home/thanhlamtba31/static-bfs-dcs/venv/bin/python -u -m ot_next.campaign \
  --protocol "$OTNEXT_SOURCE/protocol.json"
