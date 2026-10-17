#!/usr/bin/env bash
set -euo pipefail
OTNEXT_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
OTNEXT_SOURCE="$OTNEXT_REPO/experiments/ot_partial_refresh_20261017"
if [[ -f "$OTNEXT_REPO/experiments/ot_remainder_20261017/protocol.json" ]]; then
  OTNEXT_SOURCE="$OTNEXT_REPO/experiments/ot_remainder_20261017"
fi
OTNEXT_ROOT="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["root"])' "$OTNEXT_SOURCE/protocol.json")"
OTNEXT_MODULE=ot_next.campaign
if [[ "$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["stages"]==["E1"])' "$OTNEXT_SOURCE/protocol.json")" == True ]]; then
  OTNEXT_MODULE=ot_next.e1
fi
if [[ "$OTNEXT_SOURCE" == */ot_remainder_20261017 ]]; then
  OTNEXT_MODULE=ot_remainder.p2
fi
mkdir -p "$OTNEXT_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$OTNEXT_SOURCE:$OTNEXT_REPO/experiments/ot_partial_refresh_20261017:$OTNEXT_REPO/ot_tts_lab_v2_reimpl:/home/thanhlamtba31/TTS-reserach"
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
  path.write_text(json.dumps(dict(started_unix=start,deadline_unix=start+600,wall_cap_seconds=600)))
 clock=json.loads(path.read_text())
 print(max(0,int(clock['deadline_unix']-time.time())))
PY
)"
if [[ "$OTNEXT_REMAINING" -lt 1 ]]; then
  echo 'Persistent campaign cap exhausted; no neural retry authorized.' >&2
  exit 75
fi
exec timeout --signal=TERM --kill-after=30s "${OTNEXT_REMAINING}s" \
  /home/thanhlamtba31/static-bfs-dcs/ot_tts_dinov2_pilot_20261016/worker_venv/bin/python -u -m "$OTNEXT_MODULE" \
  --protocol "$OTNEXT_SOURCE/protocol.json"
