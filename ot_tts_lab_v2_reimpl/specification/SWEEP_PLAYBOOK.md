# Sweep playbook

## Fixed first pass

SD1.5 / existing DDIM100 eta1 / CFG7.5 / 512px / source8 / bank16.
Do not change sampler, timestep, encoder and target all at once.

- Index50.
- DINOv3 ViT-B/16 global feature, tensor preprocessing frozen.
- Clean-latent block projection factor4.
- Actual x_t trust RMS .025.
- P1: top50% bank, at most3 proposals/source, lookahead4, max4 groups, soft capacity ceil(N/G), duplication penalty .1.
- P2: two gradient steps, target beta1, group balance .5, epsilon_kappa1.
- P3: same as P2, 32 fixed random projections.

These are engineering starting values, not empirically established optimums.
Default SigLIP guard is OFF; enable only after calibrating a meaningful cosine
tolerance. There is no fabricated universal prompt threshold.

## Order

1. Unit tests and synthetic demo.
2. Actual encoder/VAE gradient preflight on one checkpoint; inspect current clean image.
3. One-prompt, one-tape smoke. Confirm no-op replay, finite objectives, active edits and actual post-cast RMS.
4. Six preselected development prompts, two tapes; run the most plausible branch first.
5. P1: sweep bound x selector/penalty, reusing previews.
6. P2: bound x epsilon_kappa with feature-mean and shuffled controls.
7. P3: bound x slice count with feature-mean and shuffled controls.
8. Lock finalists, then one axis at a time: spatial25%, index40/60, target balance/fraction, lookahead2/8.
9. New prompts and new banks/prefixes. Average repeated banks/suffixes inside prompt blocks for inference.

## Go/no-go rules

P1: does optimal allocation improve quality-diversity relative to independent AND
a simple greedy selector using the SAME scored proposals? If no, do not attribute
proposal-search gains to OT.

P2/P3: do transport objectives improve terminal results relative to feature mean
with the SAME encoder, trust, steps and target, and relative to shuffled weights?
If not, a non-common latent field is not evidence that OT adds value.

All: final pool best/top8 and good-distinct must improve at a worthwhile magnitude.
A larger source average that never changes selected outputs is not enough for the
stated TTS goal. Do not increase sample size merely to manufacture a small p-value.

## Comparisons

- Same-bank/prefix no-steering control is the paired anchor.
- Best-of-N should have the same post-selection diversity policy when comparing
  quality-diversity frontiers.
- Include the project's existing BFS/diversity methods under their actual verifier
  and UNet budgets; this kit does not recreate them from a paper description.
- Plot separate compute axes where necessary: UNet NFE, verifier calls, feature
  forward/backward, and measured cold wall time.
- Preserve the old reward/CLIP metrics and add independent metrics. Do not rename
  DINO cosine or DreamSim distance as the prior CLIP metric.

## Budget worksheet per prompt

M=16,N=8,T=100. Existing baseline2400.
P1 K=3 and lookahead4: minimum2464 before auxiliary current-clean evaluations.
With reference adapter, full-active preview prediction adds24, initial source
prediction adds8 when not reused:2496. Bank predictions may add further costs if
not stored along the original reference trajectories.

P2/P3: base2400 + initial clean-prediction overhead, plus feature/VAE operations.
Two-step VJP: initial source+target features, two forward feature passes per
optimization step, final surrogate features. Log feature backward separately.

Selector sweeps reuse previews for development, but each cold method pays its own
proposal/preview costs. Terminal suffix cache hits reduce development execution,
not reported cold recipe. If budget metadata is unknown, report a lower bound and
null exact cold cost rather than filling a guessed number.
