# Static BFS Graph on the TPU v5p-8 VM

The standalone checkout is `/home/thanhlamtba31/TTS-reserach`. It is separate
from `/home/thanhlamtba31/Self-Flow`, which is a different JAX project. Do not
install the Static BFS dependencies into a Self-Flow environment or alter its
jobs. The historical baseline is SD1.5 BF16, PyTorch/XLA 2.4.0, DDIM100
`eta=1`, N=8, Max+SSP, and ImageReward. The pinned package list is in
[`envs/static-bfs-tpu/requirements.txt`](../../envs/static-bfs-tpu/requirements.txt).

## Preparation

Run commands from the standalone repository root. The Python environment,
official FKD source, model cache, XLA cache, and results stay outside Git.
The official source commit is frozen. Check CPU/XLA package compatibility on
this specific VM before downloading model weights or scheduling a run; do not
silently upgrade the scientific stack.

```bash
export STATIC_BFS_REPO=/home/thanhlamtba31/TTS-reserach
export STATIC_BFS_DCS=/home/thanhlamtba31/static-bfs-dcs
cd "$STATIC_BFS_REPO"
mkdir -p "$STATIC_BFS_DCS/cache/huggingface" "$STATIC_BFS_DCS/cache/ImageReward"
git clone https://github.com/XiangchengZhang/Diffusion-inference-scaling.git "$STATIC_BFS_DCS/official"
git -C "$STATIC_BFS_DCS/official" checkout c158ab9eb058adb24ad0f8d48506ab837d7fae91
python3.10 -m venv "$STATIC_BFS_DCS/venv"
"$STATIC_BFS_DCS/venv/bin/pip" install uv
export UV_CACHE_DIR=/dev/shm/static-bfs-uv-cache
"$STATIC_BFS_DCS/venv/bin/uv" pip install --python "$STATIC_BFS_DCS/venv/bin/python" \
  'torch==2.4.0+cpu' 'torchvision==0.19.0+cpu' \
  --index https://download.pytorch.org/whl/cpu \
  --default-index https://pypi.org/simple --index-strategy unsafe-best-match
"$STATIC_BFS_DCS/venv/bin/uv" pip install --python "$STATIC_BFS_DCS/venv/bin/python" \
  -r envs/static-bfs-tpu/requirements.txt \
  'torch_xla @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.4.0-cp310-cp310-manylinux_2_28_x86_64.whl' \
  --no-build-isolation \
  --find-links https://storage.googleapis.com/libtpu-releases/index.html \
  --index https://download.pytorch.org/whl/cpu \
  --default-index https://pypi.org/simple --index-strategy unsafe-best-match
```

The local model/reward cache must contain the exact SD1.5 revision and
ImageReward-v1.0 required by the pinned baseline; `runner.py` loads them with
`local_files_only=True`. Prefetch and verify assets before the model smoke.
Never write an HF token into Git, shell history, or the report.

## Frozen paired dense-reference diagnostic (not yet run)

This specific diagnostic addresses whether the graph's dense reference is
actually better than the policies being compared. It reuses the 48 held-out
TRAIN prompt indices from N=8 retuning, with four seed windows (42–45). These
prompts have appeared in earlier TRAIN audits, so this is a diagnostic, **not**
untouched confirmation. No VALIDATION, TEST, external prompt pool, graph
search, retuning, or PSP is involved.

The four frozen methods are dense seven-event reference, MANUAL8, the N=4
policy transferred to N=8, and RETUNED8. All have 800 diffusion NFE. The
dense reference makes 64 verifier particle scores versus 32 for each K=3
policy, so the comparison is paired on diffusion compute but **not** on
verifier work. The script checks prompt split, configs, policy IDs, code
hashes, paired initial RNG, and completed-row counts.

```bash
export STATIC_BFS_RUN="$STATIC_BFS_REPO/results/static-bfs-graph/n8-dense-reference-paired-tpu-v1"
cd "$STATIC_BFS_REPO"
"$STATIC_BFS_DCS/venv/bin/python" -m scripts.static_bfs_graph.tpu_reference_compare plan --run-root "$STATIC_BFS_RUN"

# Gate: only graph-fit prompt 0, seed 42, chip 0; do not scale until it passes.
TPU_VISIBLE_CHIPS=0 "$STATIC_BFS_DCS/venv/bin/python" \
  -m scripts.static_bfs_graph.tpu_reference_compare smoke \
  --run-root "$STATIC_BFS_RUN" --dcs-root "$STATIC_BFS_DCS" --seed 42 --chip 0

# After checking the smoke rows, run one worker per chip for each stage.
# Example for chip 0/seed 42; repeat concurrently for chips 1–3/seeds 43–45.
TPU_VISIBLE_CHIPS=0 "$STATIC_BFS_DCS/venv/bin/python" \
  -m scripts.static_bfs_graph.tpu_reference_compare reference \
  --run-root "$STATIC_BFS_RUN" --dcs-root "$STATIC_BFS_DCS" --seed 42 --chip 0
TPU_VISIBLE_CHIPS=0 "$STATIC_BFS_DCS/venv/bin/python" \
  -m scripts.static_bfs_graph.tpu_reference_compare policies \
  --run-root "$STATIC_BFS_RUN" --dcs-root "$STATIC_BFS_DCS" --seed 42 --chip 0

"$STATIC_BFS_DCS/venv/bin/python" -m scripts.static_bfs_graph.tpu_reference_compare status --run-root "$STATIC_BFS_RUN"
"$STATIC_BFS_DCS/venv/bin/python" -m scripts.static_bfs_graph.tpu_reference_compare analyze --run-root "$STATIC_BFS_RUN"
```

The full run expects 192 reference and 576 policy rows. `analyze` refuses
incomplete or nonpaired data, then writes `results.json` and `REPORT.md` with
prompt-level paired bootstrap CIs. For an independent future confirmation,
freeze a design first and use the reserved TEST split in a separate protocol.
