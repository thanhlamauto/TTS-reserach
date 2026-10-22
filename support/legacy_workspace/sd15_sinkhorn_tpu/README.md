# SD1.5 Sinkhorn steering on the existing TPU VM

This benchmark adds one intervention to cached SD1.5 DDIM trajectories. It does
not update model parameters. It implements A (reward minus self barycentric
field), B (direct barycentric redistribution), C (sampled OT ancestor), and a
nearest-good interpolation control. Each supports soft, top and contrast modes.

## Environment and protocol

Run from `/home/thanhlamtba31/TTS-reserach`, using
`/home/thanhlamtba31/static-bfs-dcs/venv/bin/python`. The project reuses the
repository's pinned model and ImageReward loaders. No environment upgrades are
needed. The historical DDIM100/eta1 Runtime is not modified.

New protocol: cached `runwayml/stable-diffusion-v1-5` revision
`451f4fe16113bff5a5d2269ed5ad43b0592e9a14`, 512px, DDIM30, eta0, CFG7.5,
BF16 UNet/VAE, FP32 scheduler math, BF16 state after each update. The model ID
uses the existing cache rather than downloading an equivalent renamed SD1.5
repository. Safety checking is not invoked on the raw research outputs.
The new prompt files do not alter any existing prompt pool or split. TEST has
24 reserved prompts and an independent seed window; do not use it for tuning.

Installed stack: PyTorch 2.4.0+cpu, XLA 2.4.0, Diffusers 0.31.0.dev0 at
af28ae2d5ba0ef80d99fff7859ebea730e1cf3f8, Transformers 4.38.2.
`requirements.txt` documents this stack; installation instructions for a fresh
environment are in `scripts/static_bfs_graph/TPU_PORT.md`. Do not run pip against
the working environment. A compatible *fresh* Python 3.10 environment can use:

```bash
python3.10 -m venv /path/to/new-venv
/path/to/new-venv/bin/pip install wheel setuptools
/path/to/new-venv/bin/pip install torch==2.4.0+cpu torchvision==0.19.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
/path/to/new-venv/bin/pip install -r sd15_sinkhorn_tpu/requirements.txt \
  'torch_xla @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.4.0-cp310-cp310-manylinux_2_28_x86_64.whl' \
  --find-links https://storage.googleapis.com/libtpu-releases/index.html --no-build-isolation
```

XLA 2.4 does not expose `torch_xla.launch`; the launcher uses its documented
`xmp.spawn` equivalent. The established four-process launch below isolates one
physical chip per process. Never infer eight independent workers from eight
TensorCores. TPU inference never falls back to CPU. `torch.no_grad` is used
because XLA 2.4 fails on CLIP position-buffer slicing under `inference_mode`.

Primary API reference: [PyTorch/XLA 2.4](https://docs.pytorch.org/xla/release/r2.4/index.html).

## Commands

```bash
cd /home/thanhlamtba31/TTS-reserach
export SINKHORN_PY=/home/thanhlamtba31/static-bfs-dcs/venv/bin/python
export SINKHORN_RUN=/home/thanhlamtba31/static-bfs-dcs/experiments/sd15-sinkhorn-v2

# Check owners before taking a chip.
sudo -n fuser -v /dev/vfio/0 /dev/vfio/1 /dev/vfio/2 /dev/vfio/3
TPU_VISIBLE_CHIPS=0 PJRT_DEVICE=TPU "$SINKHORN_PY" -m sd15_sinkhorn_tpu.tpu_runtime \
  --output "$SINKHORN_RUN/device-check"

# CPU math/steering/DDIM/replay tests; no new test framework dependency.
"$SINKHORN_PY" -m unittest discover -s sd15_sinkhorn_tpu/tests -v

# One-time public supplementary CLIP download (pinned revision, separate cache).
# SD1.5 and ImageReward already exist in the original cache.
"$SINKHORN_PY" -m sd15_sinkhorn_tpu.reward \
  --dcs-root /home/thanhlamtba31/static-bfs-dcs

# Full single-chip inference/collection/replay smoke: 8 references, 4 sources,
# all A/B/C modes, A beta=0, nearest-good, and independent fresh BoN14.
TPU_VISIBLE_CHIPS=0 PJRT_DEVICE=TPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$SINKHORN_PY" -u -m sd15_sinkhorn_tpu.sweep --stage smoke --output "$SINKHORN_RUN"
"$SINKHORN_PY" -m sd15_sinkhorn_tpu.analyze --stage smoke --output "$SINKHORN_RUN"

# Collection + replay only. Running sweep afterwards reuses the bank.
TPU_VISIBLE_CHIPS=0 PJRT_DEVICE=TPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$SINKHORN_PY" -u -m sd15_sinkhorn_tpu.collect --stage compare --output "$SINKHORN_RUN"

# Stage 2: eight TRAIN prompts, M16/N8, 12 A/B/C configs plus two nearest
# controls, 50%, epsilon=.1, beta=1, rho=.25, trust=.1, eta=.1/.3.
# This uses tmux, per-job logs/status, and resumes complete immutable rows.
bash sd15_sinkhorn_tpu/launch_four.sh compare "$SINKHORN_RUN"

# After all four jobs complete:
"$SINKHORN_PY" -m sd15_sinkhorn_tpu.analyze --stage compare --output "$SINKHORN_RUN" --select

# Stage 3: one-factor TRAIN tuning around the selected configuration.
bash sd15_sinkhorn_tpu/launch_four.sh focused "$SINKHORN_RUN"
"$SINKHORN_PY" -m sd15_sinkhorn_tpu.analyze --stage focused --output "$SINKHORN_RUN" --select

# Stage 4: the two frozen configurations, 24 untouched prompts, new seed.
bash sd15_sinkhorn_tpu/launch_four.sh heldout "$SINKHORN_RUN"
"$SINKHORN_PY" -m sd15_sinkhorn_tpu.analyze --stage heldout --output "$SINKHORN_RUN"

# Alternative all-visible-device launcher: do not combine with launch_four.
# Omit TPU_VISIBLE_CHIPS; the parent must not initialize XLA before spawn.
PJRT_DEVICE=TPU "$SINKHORN_PY" -m sd15_sinkhorn_tpu.sweep --stage compare \
  --workers 4 --multi --output /path/to/separate-run-with-passing-smoke
```

`validate_replay` is always executed by collect/sweep and cannot be bypassed.
The gate compares the entire source batch's final latents and decoded RGB
against the uninterrupted trajectory. Tolerance: latent atol .001, rtol .01;
RGB mean absolute error <= .5 on the 0–255 scale. These allow BF16 rounding but
are deliberately tighter than one BF16 ULP for typical unit-scale values.
Observed errors are saved; passing tolerance does not imply bitwise equality.
Default eta0 has no subsequent sampling RNG. Optional eta>0 uses deterministic
per-step noise and must be evaluated with the same batch size.

## Definitions and limitations

Checkpoint `p` stores the state **after ceil(p*30) updates**. At 50% that is
update 15, sampling_index14; replay uses original timesteps[15:]. At 35% the
rounded checkpoint is 11/30, at 65% it is 20/30. The full scheduler config and
all 30 timesteps are serialized. A new shorter schedule is never created.

Source RNG uses `SeedSequence([trial_seed,prompt_index])`; reference and fresh
control streams add tags 1 and 2, and collisions are checked. Within each stream
one Gaussian latent is drawn at a time, so particle values do not depend on the
batch size. Intervention RNG has a separate seed/prompt/checkpoint event key.
Every steering trial loads the same unmodified source checkpoint. Source and
reference seeds, terminal rewards, images, all three checkpoint states, initial
and final latents, versions, and source hashes are retained.

The balanced log-domain CPU float32 solver minimizes `<P,C> + eps KL(P|a*w)`
with exact zero-weight handling. Costs are normalized by median off-diagonal
reference cost. All target, self and before/after divergence couplings use that
same scale. Unconverged/NaN plans fail loudly. Positive/negative ties use stable
sorting; good/bad sets are disjoint. Contrast uses the top/bottom rho fractions.

A's self term uses the independent base reference bank. This approximates the
base distribution, not the particle law after previous interventions; only one
intervention is allowed per trial. B/C negative repulsion is a heuristic.
Pooled costs with raw-latent barycenters are also a heuristic. Clipping bounds
each particle's update RMS by trust_region times the reference RMS; the actual
post-BF16 magnitude is logged too. Optional pooling, lowpass filtering and
seeded jitter are implemented. C's finite-batch ancestor counts do not exactly
preserve target masses. Interpolating unrelated semantic modes can damage
images; trust limits and the nearest-good control help measure this failure.

Primary reward is CPU ImageReward-v1.0. A `Reward.score(prompt, images)` interface
allows replacement. CLIP ViT-B/32 reports text-image cosine similarity and mean
pairwise image-embedding cosine distance, with its revision pinned in YAML.
Neither CLIP nor ImageReward is used to alter model parameters.

## Cost and outputs

At stage 2, each independent trial costs `(16+8)*30 + 8*15 = 840`
sample-steps, including the complete source trajectories required by the replay
protocol. Compare to 28 **fresh** DDIM30 samples, with both maximum reward and
the top-eight set's mean/diversity reported. CFG means 1,680 UNet sample-branch
evaluations. The trial uses 32 ImageReward evaluations versus BoN's 28, so the
comparison matches diffusion compute, not verifier work. An unmodified source
baseline costs only 240 sample-steps. Reporting reward gains over that baseline
alone does not establish better test-time scaling.

Batch shape is fixed at four; incomplete batches are padded and padding work is
reported. Smoke BoN14 physically computes 16 trajectories; it is a numerical
smoke, not an exact matched-compute result. Stage 2 BoN28 has no padding. Other
checkpoints have integer-sample rounding; account for the saved unused budget
and padding before claiming matched compute.

Each stage saves `results.csv`, `summary.csv`, `analysis.json`, `REPORT.md`, and
`reward_diversity_pareto.png`. Each bank/trial saves lossless PNG images, BF16
latents and raw scores. Atomic result markers support resume; protocol changes
require a new run namespace. Four workers write disjoint prompt paths. Per-chip
XLA caches sit inside the new output namespace, separate from historical work.

Warmup runs the original schedule twice. Report first-pass time, warm-pass time,
their difference as an **estimate** of compilation overhead, and the actual XLA
CompileTime metric in `xla_warmup*.txt`. Total generated sample-steps, padding,
warmup, replay, and reward evaluations are separate accounting categories.
Failed-attempt logs are retained. Prompt-level paired bootstrap intervals are
exploratory on TRAIN; held-out configurations are frozen before TEST evaluation.
