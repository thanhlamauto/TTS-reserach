# E0 proposal ceiling and R1 partial refresh smoke

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
