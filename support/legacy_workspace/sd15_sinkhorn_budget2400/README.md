# Fixed 2,400 NFE follow-up

SD1.5 BF16, DDIM100 eta1, CFG7.5, 512px; 24 existing follow-up prompts,
seed20361015. No tuning. Existing environments, source packages and historical
results remain unchanged.

Each method returns eight images. Primary metric: maximum terminal ImageReward
per prompt, then average over prompts. Secondary metrics: mean reward of the eight
outputs and their pairwise CLIP cosine diversity.

* BFS manual/graph: fresh N24 runs, 100 steps, 2,400 NFE. Three global SSP events,
  original frozen schedules and temperatures. Top8 terminal images are reported.
* BoN24: reuse the first24 draws of previous fresh BoN28, top8 by reward.
  No reward-based truncation; microbatch4 and stateless per-offset step RNG make
  those trajectories independent of whether 24 or 28 draws are requested.
* Sinkhorn guided8: reference16 x100 + source8 x50 + guided8 x50 =2,400.
  Reuse exact prior frozen guided outputs; source terminal images/rewards are
  excluded from the recipe. Each epsilon is a separate recipe and budget.
* Sinkhorn reference+guided: same generation, top8 from reference16+guided8.
* Independent bank24: top8 from reference16+unguided8, no steering suffix.
  This separate control helps isolate whether steering improves using the budget.

All diffusion budgets match, but CPU verifier budgets differ: 96 evaluations for
BFS vs24 for the other methods. Runtime warmup and audits are reported separately
from per-method cold inference budgets. Old execution overhead is not relabeled
as new work; the reconstructed minimal Sinkhorn recipe does not need its old
400-NFE unguided comparison suffix.

```bash
cd /home/thanhlamtba31/TTS-reserach
# Existing completed runs: analyze/audit only. For prepare, use a fresh namespace.
/home/thanhlamtba31/static-bfs-dcs/venv/bin/python \
  -m sd15_sinkhorn_budget2400.analyze --output OUTPUT
/home/thanhlamtba31/static-bfs-dcs/venv/bin/python \
  -m sd15_sinkhorn_budget2400.audit --output OUTPUT
```

`frozen_plan.json` records the protocol before execution. A single-chip TRAIN
smoke precedes four independent TEST workers. `campaign.sh` and
`sd15_sinkhorn_tpu/managed_job.sh` provide durable status/logs. The old source
checkpoint-to-intervention computation is reconstructed without source final
rewards in the audit. Raw PNGs, finite latent tensors, event particle counts,
plan/source hashes and inherited score hashes are verified at completion.
