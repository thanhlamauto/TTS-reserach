# Static-BFS TPU-only research loop — completed offline phase

**Campaign status: BUDGET_UNVERIFIED; UNDERIDENTIFIED. No new neural experiment or confirmation has run.** This report closes the executed offline phase; it does not declare the full research objective achieved.

## What actually executed

1. Read both supplied protocols in full; inspected original raw SD1.5 verification, forensic, same-trajectory and Ultra artifacts. The frozen source/input hashes are in `evidence/input_manifest.json`, `evidence/environment_checkpoint_hashes.json`, and `evidence/weight_hashes.json`.
2. Read-only SSH/metadata established the existing TPU project, zone, node, four physical device nodes, packages, source commit and storage. Cloud Billing API query failed SERVICE_DISABLED. No API was enabled and no quota was treated as money. No resource was created, resized, stopped or deleted.
3. Ran the actual installed adapter on CPU toy score vectors through SSH stdin. `evidence/cpu_semantics_test.json` records successful exact Manual↔explicit-multiplier replay, Max history, multiplier-vs-denominator distinction, SSP population size and zero-based index. No neural model or TPU initialization occurred.
4. Executed `python3 experiments/tpu_research_loop_20261016/offline_loop.py --workspace /Users/nguyenthanhlam/Documents/ChatGPT/TQK-research` from the TTS-reserach checkout. Log:`logs/offline_run.log`. This revalidated576 historical audit rows and96 Ultra rows, decomposed paired variance, audited19,701 geometry pairs and executed CPU synthetic tests. All source data remain unchanged.
5. Executed `python3 -m unittest discover -s experiments/tpu_research_loop_20261016 -p 'test_*.py' -v`. All9 local tests passed; log:`logs/CPU_unit_tests.log`. Tests cover budget denial/reserves/restart retention/kill switch, direct/graph/verification selector behavior, deduplication, shared-edge cancellation and cache identity isolation. These are CPU unit tests, not accelerator parity gates.

## Main diagnostic: what limits the SD1.5 comparison?

Fixed epsilon=.02IR, family alpha=.05 across three historical contrasts, bootstrap20,000 independent-prompt resamples after averaging four seeds. These data have already been used for development.

| Historical contrast | Mean delta | Descriptive95%CI | Seed share of variance at4seeds | Approximate prompt count for80%power at epsilon.02 |
|---|---:|---|---:|---:|
| VERIFIED−GRAPH_ONLY |−.00369|[−.03000,+.02403]|72.1%|124|
| VERIFIED−MANUAL8 |−.01267|[−.05442,+.02908]|62.4%|296|
| GRAPH_ONLY−MANUAL8 |−.00899|[−.04417,+.02181]|71.5%|187|

The power column assumes4seeds/prompt, the pilot's paired variance and a normal approximation with alpha/3. It is not guaranteed power, a sub-Gaussian assumption for IR, or a forecast for the unmeasured direct-search contrast. Current24prompt precision corresponds to approximate detectable differences around.045–.070IR, larger than.02. This supports UNDERPOWERED at the intended effect size under that planning approximation.

Measured within-prompt covariance is positive for these contrasts. Ignoring pairing gives2.8–6.0times the within-prompt variance. This is an observed property of this panel, not an assumption imposed on future RNG coupling. With a fixed total PS, estimated B/P+W/(PS) favors more prompts; seed replication remains important to estimate W and check seed sensitivity. No result shows that merely increasing seeds or prompt count makes the compiler better.

## Coverage and baseline repair

The new *declared* finite class contains six unique semantic policies, including exact Manual,Transfer,Fine and the existing shortlisted variants. Node50 and Manual's actual score multipliers[6.0693804426,15.8224987809,56.6801443288] are represented. Semantic aliases are deduplicated; no alias counts as a replicate. `covered_bank_graph_manifest.json` requires16 union edges for this fixed bank, rather than a brute-force Cartesian search. These are declared logical measurements, NOT newly collected observations.

Pure selector implementations include graph-only, frozen-shortlist full-policy verification and uniform direct full-policy evaluation on the same bank. They were unit-tested only. No equal-budget neural compiler comparison has run. All-in occupied-slice budget must match per compiler replicate; NFE, verifier calls, compilation/retry and wall-time must also be logged. If the work dimensions cannot all match, report a Pareto comparison rather than claiming equal compute in every sense.

On the original audit, Manual has the largest point estimate among the reported main roles(1.06044), ahead of Graph/Transfer(1.05146) and Verified(1.04777). Pairwise uncertainty does not establish a winner. Retaining Transfer as the historical incumbent is not a claim that it is best or safe on every distribution. No production schedule changed.

## Ultra geometry audit

The original96runs and same lone success across all four policies are verified from raw data. The map has199free cells and19,701unordered connected pairs; geodesic quantiles[min,Q1,median,Q3,max]=[1,23,49,85,160]. Old pilot distances148–153 were near the map's extreme.

Without using any new reward outcomes, the new development proposal freezes8new pairs in the middle two quartile bins, at exact distances36 and67, with matched low/high-detour tiers. It excludes old pilot and official task pairs. It records expected turns, articulation/bottleneck proxies, degree2corridor fractions and world-length/horizon ratios. Grid connectivity and unitful length/horizon ratios do not prove dynamical reachability or training-distribution coverage. Actual training-trajectory coverage remains UNVERIFIED. Original four policies, horizon2800,N16,recurrence6 and checkpoint remain frozen.

No TPU PointMaze port or new policy evaluation has run. Pooled20–80%success would only avoid floor/ceiling heuristically; paired policy discordance and cost/precision must also support a later graph. No Ultra graph is authorized automatically.

## Theory and synthetic evidence

`THEORY.md` proves the requested conditional surrogate regret bound, its separate full-policy selection decomposition, the finite-temperature weight ratio bound and the elementary simultaneous-pair implication. Coverage and uniform model/estimation errors are not known for real diffusion. Local Langevin/SMC results do not supply static-selector guarantees.

CPU tests include1,000temperature-bound cases,1,000surrogate-decomposition cases and500Monte Carlo repeats per high-noise/additive, interaction and missing-optimum toy setting. The synthetic selection bounds use *known Gaussian paired variances* and finitely allocated looks. They retain common-block correlation and yielded no observed false epsilon certificate in these runs. The interaction case can defeat the additive selector; a within-class certificate in the missing-optimum case does not close its known synthetic coverage gap. None validates ImageReward certificates or diffusion quality.

## Correctness and infrastructure not yet passed

The old SD1.5 harness runs neural ImageReward on CPU and is therefore incompatible with the new TPU-only requirement. A minimal TPU FP32 reward gate is implemented using eight saved images and historical CPU golden scores; no CPU neural reference is recomputed. Tolerances were fixed before accelerator evaluation(atol=rtol=.001), neural inputs/weights must be XLA, cold/warm timing and installed-version compile/fallback/transfer metrics are captured. **Gate NOT RUN.** If it fails, stop scientific comparisons and retain the failure; do not silently relax tolerances or fall back to GPU/CPU neural inference.

Full denoiser/reward/resampling/prefix correctness on the resulting new reward device remains pending. A cache key contract includes model,scheduler,precision,code,prefix,input,initial noise and RNG continuation; suffix keys also include the branch policy. Tests prove key isolation only, not full prefix numerical parity. Existing historical within-backend exact replay does not discharge these new gates.

## Cost, authorization and decision

- New GPU-hours:0; no Modal or CUDA compute called.
- New TPU/neural NFE/verifier evaluations:0.
- Local and remote CPU-only analysis/checks executed; their logs are retained. CPU wall timing for the main analysis is appended in `cost_ledger.jsonl`.
- Physical slice:4chips, presently externally managed. Its total provisioning/idle/billed cost is UNKNOWN. READY may accrue charges even while our jobs are absent; no authority to shut it down is assumed.
- Technical proposal cap:16chip-hours; remaining user-authorized allocation and billing/credits remain unverified, so effective new execution allowance is0.
- Enforced future accounting: cumulative worst-case reservations, whole-slice occupancy4×jobwalltime recorded separately from billed cost, at least40%confirmation reserve, PointMaze port<=min(1,.1×effective budget), deadline and kill switch. Unresolved reservations survive restarts.

**No deploy. No new method or compiler superiority claim.** The highest-value diagnostic completed is variance/precision planning, not another schedule sweep. The next permissible accelerator action is the TPU-only neural-verifier gate after actual allocation and billing confirmation. If that and subsequent correctness pass, a new multiseed DEVELOPMENT pilot can establish comparable runtime and allocation; graph-only,verification and direct search then require new selection data and equal all-in offline budgets. No confirmation set was opened, and independent compilation replicates this campaign remain0.

`RESUME.md` gives existing entrypoints and their prerequisites. The full research goal remains active; this report explicitly leaves neural execution, direct-search comparison, independent selection/confirmation and any reliable-method conclusion unachieved.

## Continuation: budget-guard interruption repair

The current guard was additionally exercised with real CPU subprocesses. All12tests pass(logs/CPU_interruption_tests.log): missing/nonfinite permission is rejected; SIGTERM of the supervisor reaps its worker and records elapsed occupancy; a worker native alarm terminates without an active supervisor. The independent alarm is armed before TPU/neural imports. This repairs the prelaunch deadline contract; no GPU/TPU or neural computation was used. The original9-test result above remains the initial-phase record. Allocation/billing verification is still absent, so scientific comparison and confirmation remain NOT RUN.
