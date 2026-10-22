#!/usr/bin/env bash
# Detached, locked, resumable campaign. All selection rules are frozen in plan.
set -euo pipefail
repo=${1:?repository directory}
dcs=${2:?DCS environment directory}
run=${3:?campaign directory}
python="$dcs/venv/bin/python"
module=scripts.static_bfs_graph.tpu_grid_density_overnight
mkdir -p "$run/logs"
cd "$repo"
exec 9>"$run/driver.lock"
if ! flock -n 9; then
  echo "A campaign driver already owns this run" >&2
  exit 75
fi
stamp() {
  "$python" - "$run" "$1" <<'PY'
import json,os,sys,time
from pathlib import Path
root=Path(sys.argv[1]); dest=root/'status.json'; tmp=root/'status.json.tmp'
tmp.write_text(json.dumps({'stage':sys.argv[2],'updated_unix':time.time(),'driver_parent_pid':os.getppid()},indent=2)+'\n')
tmp.replace(dest)
PY
  echo "STAGE $1 $(date -u +%FT%TZ)"
}
record_time() {
  "$python" - "$run" "$1" <<'PY'
import json,sys,time
from pathlib import Path
dest=Path(sys.argv[1])/'timing.json'; obj=json.loads(dest.read_text()) if dest.exists() else {}
obj[sys.argv[2]]=time.time()
dest.write_text(json.dumps(obj,indent=2)+'\n')
PY
}
trap 'stamp failed' ERR
stamp smoke
record_time smoke_started_unix
TPU_VISIBLE_CHIPS=0 PJRT_DEVICE=TPU "$python" -m "$module" smoke \
  --run-root "$run" --dcs-root "$dcs" --chip 0 >>"$run/logs/smoke.log" 2>&1
record_time smoke_finished_unix
stamp full
record_time full_started_unix
pids=()
for chip in 0 1 2 3; do
  TPU_VISIBLE_CHIPS="$chip" PJRT_DEVICE=TPU "$python" -m "$module" full \
    --run-root "$run" --dcs-root "$dcs" --chip "$chip" >>"$run/logs/full-chip$chip.log" 2>&1 &
  pids+=("$!")
done
printf '%s\n' "${pids[@]}" >"$run/worker.pids"
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
record_time full_finished_unix
"$python" -m "$module" status --run-root "$run"
if (( failed )); then
  stamp failed
  exit 1
fi
stamp export
"$python" -m "$module" export --run-root "$run" >"$run/logs/export.log" 2>&1
stamp analyze
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$python" -m "$module" analyze \
  --run-root "$run" >"$run/logs/analyze.log" 2>&1
stamp completed
