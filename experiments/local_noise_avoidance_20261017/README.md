# Local uncertainty-aware noise avoidance: stage1 diagnostic

This is an independent implementation of the supplied refined proposal on the unchanged SD1.5 XLA backend. It tests degradation prediction, not yet efficiency or method superiority.

## Frozen protocol

- SD1.5 revision451f4fe16113bff5a5d2269ed5ad43b0592e9a14, DDIM100 eta1 CFG7.5, BF16 states, original fused UNet/DDIM transition.
- Exact checkpoint after40 completed updates. Control variance-noise at zero-based indices40..44 (updates41..45). Do not edit latent states directly.
- Four Gaussian spacetime basis directions, whitened with the existing DDIM sigma-weighted RMS Gram matrix. A nearly singular matrix aborts rather than amplifying a direction or silently changing dimension.
- Fixed injected energy radius r=.025 is an exploratory engineering default, not tuned or demonstrated optimal. It is not claimed equivalent to an old latent-edit RMS cap.
- Root0 is selected before reward observations. Common action-block noise is frozen across A/B/C; future indices45..99 use independent keyed tapes. No model transfer to other anchors, no recentering, no guided-batch online update.
- At each anchor:18 axis/noop scouting suffixes across A/B,9 antithetic dense mixed/noop suffixes under fresh C. Both predictors use identical18 scouts.
- TRAIN prompts1/2/3 calibrate separate linear/quadratic epsilon values. Freeze epsilon before launching TRAIN prompts4/5/6. Reserved final24 and all other held-out pools remain unused.
- Epsilon=max(0,90th-percentile higher of C_delta-max(A_prediction,B_prediction)) on the24 calibration mixed actions. This is an empirical allowance, not a confidence bound.
- delta_reward=.05, scale=.1, gamma=1, rho=.25, weight_floor=.2. Positive-guidance diagnostic uses the same information, mixture/floor and beta1. All directions have the same radius. Flat weights recover uniform allocation.

## Registered stage1 decision

Finite-action weighting on the8 mixed C directions is only a diagnostic of a proposed distribution; it is not a newly sampled guided batch. For each model require all of: at least4 actions with raw weight<=.8 across the3 screen anchors, observed harm precision>=.75, mean weighted harm-rate decrease>=.05, nonnegative change on at least2/3 anchors, and matched within-anchor shuffled-weight tail fraction<=.1 over10k permutations. These are practical screening criteria chosen before results, not population confirmation. If neither model passes, stop. If one passes, its frozen implementation may enter a separately tracked mechanism experiment; no automatic final nomination.

## Accounting and limits

Per anchor conceptual diagnostic40+27*60=1660 sampling-NFE. Six anchors cost9960. Original full100/segmented replay of4 roots adds760 per anchor,4560 total. Actual nominal sampling14520; microbatch padding360; physical sample steps14880, CFG branch sample evaluations29760. Auxiliary UNet calls0. Reward/DINO/CLIP/VAE overhead is separate. Calibration cost is never hidden inside the3-anchor screen.

Single-chip complete calibration smoke precedes expansion. Three chips run the3 calibration anchors, error allowances freeze, then the3 screen anchors. Fourth chip unused because each phase has3 anchors. Persistent1200s cap, no reset on repair. Existing TPU only, no paid GPU, no backend/dependency change. Raw outputs, exact counter receipts, identities and source/config hashes retained.

## Outcomes to report

Degradation and useful-action suppression, matched shuffled null, positive guidance, per-anchor response predictions, actual injected/state/terminal changes, pixel duplication and diversity. Semantic review is blinded and separate from ImageReward; UNKNOWN abstains. Do not call reward loss semantic damage. Existing good/distinct thresholds and prompt rubrics remain frozen. Scouting/new pool coverage can be audited descriptively, but no sampler or compute advantage follows from this stage1 screening.

Still conditional and NOT RUN here: actual9-sample post-scouting mechanism batches; a2360/2400-NFE full allocation comparison, random/BoN freely allocating their budget, and active CMA-ES; all final confirmation. The supplied27-rollout screen is used twice because model-error calibration must be independent of evaluation.

## Execution

Run through OpenResearch using the inherited fixed command `bash experiments/ot_partial_refresh_20261017/ops/run.sh`, SSH backend `otlab-dinov2-tpu`. The committed dispatcher selects this protocol; source is immutable before remote execution. No GitHub push is required.
