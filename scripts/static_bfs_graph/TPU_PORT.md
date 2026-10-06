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
"$STATIC_BFS_DCS/venv/bin/pip" install wheel setuptools
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
TPU_VISIBLE_CHIPS=0 PJRT_DEVICE=TPU "$STATIC_BFS_DCS/venv/bin/python" \
  -m scripts.static_bfs_graph.tpu_reference_compare smoke \
  --run-root "$STATIC_BFS_RUN" --dcs-root "$STATIC_BFS_DCS" --seed 42 --chip 0

# After checking the smoke rows, run one worker per chip for each stage.
# Example for chip 0/seed 42; repeat concurrently for chips 1–3/seeds 43–45.
TPU_VISIBLE_CHIPS=0 PJRT_DEVICE=TPU "$STATIC_BFS_DCS/venv/bin/python" \
  -m scripts.static_bfs_graph.tpu_reference_compare reference \
  --run-root "$STATIC_BFS_RUN" --dcs-root "$STATIC_BFS_DCS" --seed 42 --chip 0
TPU_VISIBLE_CHIPS=0 PJRT_DEVICE=TPU "$STATIC_BFS_DCS/venv/bin/python" \
  -m scripts.static_bfs_graph.tpu_reference_compare policies \
  --run-root "$STATIC_BFS_RUN" --dcs-root "$STATIC_BFS_DCS" --seed 42 --chip 0

"$STATIC_BFS_DCS/venv/bin/python" -m scripts.static_bfs_graph.tpu_reference_compare status --run-root "$STATIC_BFS_RUN"
"$STATIC_BFS_DCS/venv/bin/python" -m scripts.static_bfs_graph.tpu_reference_compare analyze --run-root "$STATIC_BFS_RUN"
```

The full run expects 192 reference and 576 policy rows. `analyze` refuses
incomplete or nonpaired data, then writes `results.json` and `REPORT.md` with
prompt-level paired bootstrap CIs. For an independent future confirmation,
freeze a design first and use the reserved TEST split in a separate protocol.

## Smaller 20-prompt, one-seed graph study

At the user's request, `tpu_n8_graph20_seed1.py` freezes the first 20 indices
of the existing 60-prompt TRAIN split, **seed 42 only**, and the original 92
edges, step grid, tau grid, N=8, backend, and K=3 additive search. Each of
four chips receives five disjoint prompts; all use seed 42. This is not four
seeds, and its 20 prompt units do not support claims about untouched TEST.

The 20 reference rows and 1,840 logical edge rows consist of 1,680 new edge
interventions plus 160 exact copies of the reference. This is graph fitting
only; the chosen policy's terminal reward is **not** audited. The analysis
reports the K=3 path and bootstrap exact-path frequency at both the 12-prompt
prefix and all 20 prompts, without changing the objective.

```bash
export STATIC_BFS_GRAPH20="$STATIC_BFS_DCS/results/n8-graph20-seed1-tpu-v1"
cd "$STATIC_BFS_REPO"
"$STATIC_BFS_DCS/venv/bin/python" -m scripts.static_bfs_graph.tpu_n8_graph20_seed1 \
  plan --run-root "$STATIC_BFS_GRAPH20"

# One-chip gate: first reference and first new intervention, then inspect rows.
TPU_VISIBLE_CHIPS=0 PJRT_DEVICE=TPU "$STATIC_BFS_DCS/venv/bin/python" \
  -m scripts.static_bfs_graph.tpu_n8_graph20_seed1 reference \
  --run-root "$STATIC_BFS_GRAPH20" --dcs-root "$STATIC_BFS_DCS" \
  --chip 0 --max-new-units 1
TPU_VISIBLE_CHIPS=0 PJRT_DEVICE=TPU "$STATIC_BFS_DCS/venv/bin/python" \
  -m scripts.static_bfs_graph.tpu_n8_graph20_seed1 edges \
  --run-root "$STATIC_BFS_GRAPH20" --dcs-root "$STATIC_BFS_DCS" \
  --chip 0 --max-new-units 1

# Once the gate passes, run reference for chips 0–3, then edges for chips 0–3.
# Each chip is a separate process with TPU_VISIBLE_CHIPS set to its chip index.
"$STATIC_BFS_DCS/venv/bin/python" -m scripts.static_bfs_graph.tpu_n8_graph20_seed1 \
  status --run-root "$STATIC_BFS_GRAPH20"
"$STATIC_BFS_DCS/venv/bin/python" -m scripts.static_bfs_graph.tpu_n8_graph20_seed1 \
  analyze --run-root "$STATIC_BFS_GRAPH20"
```

`analyze` refuses partial rows or mismatched prompt/seed/edge hashes. Do not
combine these TPU records with the earlier Modal GPU graph or the TPU
four-seed graph without a separate protocol and provenance audit.

## Dense-10 grid-density pilot

`tpu_grid_density_10bin.py` is a separate, TRAIN-only diagnostic of temporal
grid resolution. It reuses the same first 20 TRAIN prompt indices and seed 42
as graph20; no production graph or compiled policy is modified. The frozen
comparison is Dense10 at indices `[10,20,...,90]`, ten single-midpoint
policies inserting one of `[5,15,...,95]`, and Dense20 at `[5,10,...,95]`.
Every event uses tau=8. For DDIM100, index 10 means normalized denoising
progress 0.10, **not** the scheduler's descending timestep value.

Run `plan`, then `smoke` on chip 0. The smoke executes Dense10, midpoint 45,
and Dense20 on one prompt and writes all three complete event sequences to
`smoke_gate.json`; inspect it before expanding. Run `full` concurrently on
chips 0–3, one process per chip with `TPU_VISIBLE_CHIPS=<chip>` and
`PJRT_DEVICE=TPU`, from the repository root. `full` refuses to start without
a passing smoke gate. It resumes atomic rows; do not launch duplicate
workers. After 20 + 200 + 20 rows are present, run `account` with the Unix
timestamps bracketing smoke and full, then `analyze`.

```bash
export GRID_RUN="$STATIC_BFS_DCS/results/grid-density-10bin-pilot-v1"
"$STATIC_BFS_DCS/venv/bin/python" -m scripts.static_bfs_graph.tpu_grid_density_10bin \
  plan --run-root "$GRID_RUN"
TPU_VISIBLE_CHIPS=0 PJRT_DEVICE=TPU "$STATIC_BFS_DCS/venv/bin/python" \
  -m scripts.static_bfs_graph.tpu_grid_density_10bin smoke \
  --run-root "$GRID_RUN" --dcs-root "$STATIC_BFS_DCS" --chip 0
# Repeat this full command concurrently for chips 0, 1, 2 and 3:
TPU_VISIBLE_CHIPS=0 PJRT_DEVICE=TPU "$STATIC_BFS_DCS/venv/bin/python" \
  -m scripts.static_bfs_graph.tpu_grid_density_10bin full \
  --run-root "$GRID_RUN" --dcs-root "$STATIC_BFS_DCS" --chip 0
"$STATIC_BFS_DCS/venv/bin/python" -m scripts.static_bfs_graph.tpu_grid_density_10bin \
  status --run-root "$GRID_RUN"
"$STATIC_BFS_DCS/venv/bin/python" -m scripts.static_bfs_graph.tpu_grid_density_10bin \
  account --run-root "$GRID_RUN" --start-unix START_UNIX --end-unix END_UNIX
"$STATIC_BFS_DCS/venv/bin/python" -m scripts.static_bfs_graph.tpu_grid_density_10bin \
  analyze --run-root "$GRID_RUN"
```

The report contains 20,000 paired prompt-bootstrap draws for each comparison,
one PNG plot, verifier-work caveats, and descriptive outcome A/B/C/D. `GPU-hours`
are zero on this TPU VM; the run also records TPU-chip-hours and a clearly
labelled illustrative price scenario, not a cloud invoice. No policy reward
benchmark or final schedule tuning is part of this pilot.
