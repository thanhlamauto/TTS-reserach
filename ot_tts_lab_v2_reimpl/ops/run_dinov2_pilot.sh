#!/usr/bin/env bash
set -euo pipefail
OTLAB_SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OTLAB_ROOT=/home/thanhlamtba31/static-bfs-dcs/ot_tts_dinov2_pilot_20261016
OTLAB_WORKER="$OTLAB_ROOT/worker_venv/bin/python"
mkdir -p "$OTLAB_ROOT"
if [[ ! -x "$OTLAB_WORKER" ]]; then
  python3 -m venv "$OTLAB_ROOT/worker_venv"
fi
if [[ ! -f "$OTLAB_ROOT/worker_environment_ready.json" ]]; then
  "$OTLAB_WORKER" -m pip install --disable-pip-version-check --no-cache-dir \
    torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
  "$OTLAB_WORKER" -m pip install --disable-pip-version-check --no-cache-dir \
    transformers==4.56.2 diffusers==0.35.1 numpy==2.2.6 PyYAML==6.0.3 Pillow==12.0.0
  "$OTLAB_WORKER" -m pip freeze > "$OTLAB_ROOT/worker_environment.freeze.txt"
  "$OTLAB_WORKER" - "$OTLAB_ROOT/worker_environment_ready.json" <<'PY'
import importlib.metadata,json,pathlib,sys,torch
assert torch.__version__=='2.8.0+cpu'
assert not torch.cuda.is_available()
versions={k:importlib.metadata.version(k) for k in ['torch','torchvision','transformers','diffusers','numpy','Pillow']}
pathlib.Path(sys.argv[1]).write_text(json.dumps(versions,indent=2)+'\n')
print('WORKER_ENVIRONMENT_READY',versions,flush=True)
PY
fi
export PYTHONPATH="$OTLAB_SOURCE:/home/thanhlamtba31/TTS-reserach"
export PJRT_DEVICE=TPU
export TPU_VISIBLE_CHIPS=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "$OTLAB_SOURCE"
timeout --signal=TERM --kill-after=30s 3600s \
  /home/thanhlamtba31/static-bfs-dcs/venv/bin/python -u -m otlab.campaign \
  --root "$OTLAB_ROOT" --worker-python "$OTLAB_WORKER"
