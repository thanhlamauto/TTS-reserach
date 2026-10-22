# Paired BFS versus Sinkhorn

This follow-up reads the completed `sd15-sinkhorn-v2` benchmark without modifying
it. It uses the installed TPU environment and the existing
`StaticGraphBFSAdapter` / `PaperBFSAdapter` plus the official SSP allocator.
No reward/model packages or historical BFS files are changed.

Two policies are frozen from previous, independent BFS work:

- MANUAL: gamma=.024, base temperature10, Max+SSP at zero-based indices
  `[20,40,80]` in DDIM100.
- GRAPH: `[40,60,90]`, temperatures `[8,8,32]` in DDIM100.

The historical verification audit reports selected endpoint IR 1.0604434 and
1.0514580 respectively. These are historical TRAIN results on another pool,
not the score of either method on the new Sinkhorn prompts.

## Design

The frozen experiment is at
`/home/thanhlamtba31/static-bfs-dcs/experiments/sd15-sinkhorn-bfs-v1`.
It records `frozen_plan.json`, a separate pre-run `analysis_plan.json`, per-chip
logs/status, immutable source snapshots, images, latents and resampling events.

1. DDIM30 eta0: use the same eight TRAIN and 24 previously evaluated TEST
   prompts/seeds as Sinkhorn. Reuse cached A/B/C TRAIN results, the two frozen
   A TEST configurations, and fresh BoN28. Run manual BFS N8/N28 and graph N28.
2. DDIM100 eta1: the same 24 follow-up prompts, same frozen A configurations
   transferred without retuning, M16/N8 banks, both N28 BFS policies, N8 manual
   BFS, and fresh BoN28.

Within each protocol, BFS N28 has the exact Sinkhorn diffusion budget:
`28*steps = (16+8)*steps + 8*(steps/2)`, i.e. 840 or 2800 sample-steps. N8 BFS
is a lower-budget control (240 or800). CFG doubles sample-branch evaluations.
BFS N28 uses112 ImageReward evaluations, A uses32 and BoN uses28; verifier work
is reported separately.

All methods share the new sampler: BF16 UNet/VAE, FP32 CFG/DDIM arithmetic, BF16
states and stateless eta noise keyed by prompt seed, original timestep index
and particle-slot microbatch offset. This is an algorithm comparison using a
shared numerical loop; it is not a byte-identical reproduction of the historical
pipeline's BF16 scheduler arithmetic and generator-consumption order.

For DDIM30, transfer **completed update progress** using
`ceil((old_index+1)*30/100)-1`; manual becomes `[6,12,24]`, graph becomes
`[12,18,27]`. Preserve the original event temperatures. Resampling seeds use
the existing event seed function at the actual event index. Max history starts
at zero, so early negative rewards are floored exactly as in the old adapter.

Particle workloads run in static batches of four. At an event, the complete
population is assembled on CPU, x0 predictions are decoded/scored, then the
unchanged Max+SSP adapter resamples all N particles jointly. It does not run
independent four-particle BFS islands. Every post-resampling slot receives its
own deterministic eta noise. At eta0, clones cannot diversify through new
sampling noise; final unique latent counts are therefore logged.

## Outputs and interpretation

Primary metric: mean of the best terminal reward per prompt, as in the existing
BFS reports. Supplementary: mean reward of the returned eight images, CLIP
alignment and pairwise CLIP diversity. N28 methods use top8 for set metrics;
full BFS population mean is also saved.

Report both guided-only A and **recycle** A. Recycle selects top8 from all fully
generated reference/source/guided terminal images, removing pixel-identical
duplicates. These images were already generated and scored in the stated A
budget, so there is no extra NFE or reward evaluation. This prevents comparing
BFS selection against an A workflow that discards its already-known reference
endpoints. A bank24-only selection control has a lower budget. BFS decoded x0
event predictions are not counted as completed terminal trajectories.

The same previously evaluated prompts are used for adding a fixed comparator;
this is not a newly untouched TEST. There is no new hyperparameter selection.
Paired 20,000-draw prompt bootstrap CIs are exploratory and not multiplicity
adjusted. Do not attribute a DDIM30-versus-DDIM100 difference to BFS alone.

## Commands on the VM

```bash
cd /home/thanhlamtba31/TTS-reserach
export BFS_COMPARE_PY=/home/thanhlamtba31/static-bfs-dcs/venv/bin/python
export BFS_COMPARE_RUN=/home/thanhlamtba31/static-bfs-dcs/experiments/sd15-sinkhorn-bfs-v1

"$BFS_COMPARE_PY" -m unittest sd15_sinkhorn_bfs_compare.test_bfs sd15_sinkhorn_bfs_compare.test_analysis -v
"$BFS_COMPARE_PY" -m sd15_sinkhorn_bfs_compare.run --prepare --output "$BFS_COMPARE_RUN"

TPU_VISIBLE_CHIPS=0 PJRT_DEVICE=TPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$BFS_COMPARE_PY" -u -m sd15_sinkhorn_bfs_compare.run --smoke --output "$BFS_COMPARE_RUN"

# Only after smoke_gate.json passed; check owners first. Repeat for chip0–3.
TPU_VISIBLE_CHIPS=0 PJRT_DEVICE=TPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$BFS_COMPARE_PY" -u -m sd15_sinkhorn_bfs_compare.run \
  --output "$BFS_COMPARE_RUN" --workers 4 --rank 0

"$BFS_COMPARE_PY" -m sd15_sinkhorn_bfs_compare.analyze --output "$BFS_COMPARE_RUN"
```

The actual campaign uses managed tmux jobs, one per physical chip, via the saved
`launch_after_smoke.sh`; it gates four-worker execution on the smoke status.
Smoke checks both eta0 and eta1 segmented execution against the uninterrupted
sampler and validates source-checkpoint replay before the large comparison.

Final tables: `analysis/results.csv`, `analysis/summary.csv`,
`analysis/paired_comparisons.csv`, `analysis/REPORT_VI.md`,
`analysis/reward_diversity.png`. Incomplete analysis refuses to report a complete
comparison; `--partial` is available for progress inspection.

For a completed run, use analyze/audit directly. `--prepare` is for a fresh
namespace; existing frozen plans reject source changes. Experiment sources are
snapshotted under frozen_source, and analysis/audit sources under analysis_source.
