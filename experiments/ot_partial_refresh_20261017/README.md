# E0 proposal ceiling and R1 partial refresh smoke

## Current child: authorized calibrated E1

The parent engineering study completed E0 and all4 R1 methods on prompt0.
Prediction materialization did NOT pass bitwise compatibility; see
`FUSED_TRANSITION_AMENDMENT.md`. The current child preserves original fused
DDIM transitions and charges24 auxiliary preview UNet evaluations per method.
The fixed launcher now selects the committed stage/root in `protocol.json`.
Parent source/history remain immutable; no budget clock is reset.

The user then explicitly chose: calibrate on the old pilot and run R1. E1 is
6 fixed TRAIN prompts1..6, seed2026101702,4 methods on all4 existing chips,
with no outcome-dependent expansion. Cold instrumented recipe per method:
2400 sampling+24 auxiliary=2424 total;6x4x2424=58176 nominal. Shared prefix
and6 original-sampler correctness gates yield42864 actual UNet sample
evaluations expected. Minimal uninstrumented none would not need a preview;
no exact2400-total comparison with another method is claimed.

`calibration.json` freezes IRq75=-0.9384542256593704 and independent CLIP
pair-distanceq25=0.07785658785589741 from ONLY the24 original baseline
terminal images (bank16+source8) of prompt0/seed20261016. These are exploratory
one-prompt percentile cutoffs, not validated universal image-quality criteria.
They do not use any new R1 outcomes. Semantic success remains a separate
required conjunction, with UNKNOWN excluded in primary and an optimistic
upper count reported. The DINO medoid merge distance0.1457543222586308
comes from the parent old-bank q10 calibration, unchanged across E1 prompts.

Two actual8-output sets are recorded: historical top8 IR, and a common
quality-diversity selector (IR eligibility + greedy DINO farthest-first,
deterministic IR fallback). DINO remains the optimized geometry; CLIP is the
supplementary distance used in the primary exact good-distinct count. Primary
selector is quality-diversity. Method/reward-blinded, shuffled image-hash grids
are evaluated with frozen `semantic_rubrics.json`; this is a transparent Codex
visual annotation protocol, not a validated human benchmark. Labels are shared
for exact duplicate pixels, never changed by method or output selector.

Primary OT-vs-reward and OT-vs-group contrasts average within prompt before
20000-draw descriptive prompt bootstrap; approximate Bonferroni intervals are
reported beside95% intervals. Six TRAIN prompts/one prefix cannot establish
general superiority. A preregistered practical signal recommends another
independent-prefix study only; no sweep, antithetic stage, final24x2 evaluation
or new neural job is automatically opened.

The rest of this README records the parent setup and its original gates;
subsequent execution amendments above take precedence for this E1 child.

This is a new follow-up to the actual `ot_tts_lab_v2_reimpl` pilot, not a claim
that the unavailable original OT package has been installed. Original files,
noise, cache, failed P2 evidence and source stay immutable. No new encoder,
sampler, reward, weights, GPU or Modal allocation is used.

The attached Vietnamese specification is preserved in `SUPPLIED_REQUEST.txt`
and `SUPPLIED_README_VI.md`. The fixed run command is
`bash experiments/ot_partial_refresh_20261017/ops/run.sh`.

## This bounded run

1. Verify36 original input files by SHA256. Resume actual index54 P1 previews.
   Replay all8 no-op and6 previously selected edits exactly. Then complete
   exactly10 missing edits (460 new suffix sample-NFE; gates/padding separate).
2. Measure the full proposal ceiling, all6561 source-preserving oracle
   combinations, unconstrained oracle, preview/terminal gains and group-label
   stability against fixed preview centers. Oracles are diagnostics only.
3. R1 engineering smoke: one old TRAIN prompt, fresh prefix seed2026101701,
   24 online particles,18 survivors,6 replaced slots, top12 donors, index40.
   None/reward-only/hard-group/semi-relaxed OT share masks, logits, random
   uniforms and child-slot noise. Copy exact donor state and its prediction.
   The event UNet prediction is reused for the DDIM update, not an unreported
   extra sampling evaluation. Surviving18 trajectories must remain bitwise
   equal across methods.

Gamma1/epsilon1 are fixed, no sweep. Merge distance is calibrated by a frozen
q10 rule on old16 terminal bank features before any new R1 outcomes. This
implementation choice was not specified numerically by the supplied README;
it is recorded explicitly, not claimed as a supplied or tuned optimum.

One visible TPU chip runs the smoke, as in the old pilot;4 physical chips are
allocated to the existing VM. DINOv2 and ImageReward retain their CPU worker
backend. No claim of fully TPU neural evaluation is made. Cap3600s from the
first launch, retained across retries. No neural concurrency with other jobs.

## Explicit gates for later work

The six-prompt E1 quality experiment is not opened by this engineering run.
Reward threshold, independent diversity threshold, semantic scoring rubric and
shared quality-diversity output selector remain null pending calibration.
Do not invent them, tune them on new outcomes, or replace semantic success with
an ImageReward cutoff. Output top8 by IR is retained as the historical view.
The antithetic factorial, R2, R3, P2 repair and final24x2 evaluation are separate
conditional stages, not a large automatic sweep. No held-out pool is opened.

CPU checks:
`PYTHONPATH=experiments/ot_partial_refresh_20261017 python -m unittest discover -s experiments/ot_partial_refresh_20261017/ot_next/tests -v`

Every run uses the committed immutable source through `orx exp run` SSH backend.
Never push merely to launch and never execute the neural command via raw SSH.
