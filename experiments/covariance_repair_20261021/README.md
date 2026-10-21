# PDF covariance repair: mechanism pilot, round 1

This child of the completed 2400-NFE mean-gradient experiment implements a new
local diffusion-prior ensemble and direct covariance correction at index50.
It does not relabel the old injected-noise filter as the PDF algorithm.

The fixed rectangle uses six existing TRAIN development prompt texts, two fresh
seeds2026102101/02, four independently seeded roots per unit, and eight scouts
per parent. The original SD1.5/DDIM100 eta1 CFG7.5 BF16 stack is unchanged.
No reserved confirmation prompts, paid GPU, model downloads, or backend change.

A VP forward kernel raises index50 to the noise level at index45. Five original
DDIM transitions return eight variants to index50. All scouts, baseline and
correction rollouts for one root share the same future tape. Different roots
and units use independent streams. Compute covariance from eight terminal
badness values [tau-IR]+. Fixed eta=1, caps .25/.5 times median actual scout RMS;
never normalize weak covariance up to a cap. Signal gate: parent below tau,
badness std>1e-6, and cosine of first4/last4 covariance>0. Otherwise exact no-op.

Controls use the same measured scouts: no correction, covariance and matched
random within the same local ensemble span. The latter matches actual BF16
state displacement within2% and is never larger. Its direction is fixed across
caps. Every scored scout remains eligible in every method. Acceptance is a
badness improvement>=.05; no backtracking or outcome-driven tuning. All no-ops
are actually rolled out, matched to baseline bitwise, and charged.

The quality and CLIP novelty cutoffs are inherited exploratory thresholds from
one old calibration prompt. They are not universal quality/semantic validators.
No independent semantic-preservation guarantee is claimed. Parent quality
already above threshold causes no repair rather than inventing a gradient.

Cold NFE per unit: no-correction2160, each correction2360 including160NFE to
construct ensembles. Shared actual execution2960/unit. Total35920 including400
original full100 replay NFE. Zero padding/extra clean-prediction UNet; CFG branch
work is twice the sample NFE. This round is not an equal-budget BFS benchmark.

Primary: average per-parent accepted endpoint improvement beyond its complete
shared baseline+8scout pool. Average four parents then two seeds per prompt;
bootstrap six prompt means20k times, exact one-sided sign flips, Holm two caps.
Open later design work only if at least one covariance endpoint adds beyond
scouting, CI95 lower>0 and Holm p<=.05 against matched random. No claim based
solely on best scout selection. All twelve units complete regardless of reward.

Numerical tests check covariance sign, flat-badness and good-parent no-op,
weak-direction nonamplification, actual BF16 cap, and matched random. One-chip
full100/replay smoke gates four-chip execution. A persistent30-minute deadline
cannot reset on retries. Work is stored under /dev/shm to avoid the nearly-full
VM disk; it must be preserved externally before any machine restart.

Launch the committed source through the inherited fixed run command:

```sh
orx exp run 9ed38e89-17f9-4a4e-87e6-975bde25a498 --backend ssh --host otlab-dinov2-tpu --no-container
```

The root output path, gates, seeds, complete cold/actual accounting, and deferred
round2/3 plan are in protocol.json. Old experiment nodes are frozen. No push is
needed for compute transport.
