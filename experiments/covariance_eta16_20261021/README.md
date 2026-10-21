# Same cached scouts, eta-correction16, binding actual caps

User-requested follow-up to round1. Keep all six TRAIN development prompts,
two seeds, four parents per unit, K8 local prior scouts, index50, quality cutoff,
split-cosine gate, acceptance and future tapes unchanged. Change only correction
eta1→16. **DDIM stochastic eta stays1.** The new eta is the next power of2 above
max(.5rQ/RMS(v)) computed from the two old active directions. No new reward
selects eta, cap, parent, or gate. Finish once; no search until a win.

Restore the old immutable public HF archive into a RAM-only cache and verify
its full SHA256 plus every extracted member and12raw bundles. Recompute the
old covariance from the same8scout rewards; require identical48signal gates.
Check before neural work that both actual BF16 caps bind and that large/small
actual displacement ratio is1.9–2.1 for BOTH active parents. Zero/ineligible
updates stay zero. Matched random uses exactly the old independent ensemble-span
RNG and must match actual BF16 covariance RMS within2%, never larger.

Only two old directions are active (both on prompt6); the other46parents are
identity no-ops. This is a diagnostic reused cohort, not confirmation. Replay
four cached baseline suffixes first, requiring latent/pixel/reward exact match.
Run the eight changed suffixes, every one with its own parent's original tape.
One existing TPU chip suffices for two microbatches4; no paid backend or GPU.
No-op endpoints and shared baseline/scout features are reused and individually
hash checked, not billed as new physical compute.

Cold ledger for EACH covariance/random method per prompt-seed:

- Prefix4×50=200NFE.
- Prior construction4×8×5=160NFE.
- Scouts4×8×50=1600NFE.
- Baselines4×50=200NFE.
- Corrections4×50=200NFE.
- **Total2360NFE; scout construction+rollout1760 included.**

The no-correction recipe costs2160. Across12units and5recipes:139200cold NFE.
Shared cached common work is25920NFE;9200 correction-slot NFE are identity
no-ops reused at execution;400NFE are eight new suffixes. A separate200-NFE
four-root continuation replay makes **600new physical NFE**,1200CFG branch
evaluations, zero padding/auxiliaryUNet. Do not replace2360 by600 in method
comparisons. The old pilot already executed35920NFE; the combined two studies
physically consumed36520NFE, so original scouting expenditure is not erased.

Eight new terminal IR scores and DINO/CLIP encodes; four replay IR scores;
432cached common terminal records. Preserve all scouts in every pool. Log
unaccepted raw candidate IR as well as accepted incremental best beyond scouts.
Primary inference and acceptance remain unchanged; active-parent effect sizes
are important because only one prompt has a nonzero update.

Fix the dormant NumPy scalar count serialization issue on this child branch and
test positive contributions/acceptance before launch. Old source remains frozen.
Persistent15min wall cap includes restoration/startup. Exact config, source
hashes, replay, caps and cache receipts are saved. Small delta.pt files store
only new neural outputs; PARENT_CACHE_DEPENDENCY.json records immutable old raw.

Inherited run command launches through:

```sh
orx exp run ec2abe74-bd84-490a-84fe-ba2b33c29d6c --backend ssh --host otlab-dinov2-tpu --no-container
```
