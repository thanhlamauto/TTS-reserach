# State: Static-BFS research loop, 16 October2026

## Verified starting evidence

- Original latest SD1.5 verification raw has576 unique audit runs =24 historical TRAIN prompts ×4 paired seeds ×6 unique policies. Nine role labels contain duplicates; GRAPH_ONLY and TRANSFER are the same schedule, and shortlist roles overlap.640 graph edge observations and128 verification runs are confirmed by original artifacts. No independent compiler replicate is created by these aliases or report copies.
- V1's VERIFIED−GRAPH_ONLY mean IR is−0.00369 and VERIFIED−MANUAL8−0.01267. Their descriptive paired intervals cross zero. The original selection status is UNDERIDENTIFIED, not an epsilon certificate. Fine and exact Manual were audit-only in that campaign.
- Original Ultra rows validate96 runs,12 tasks ×2 seeds ×4 frozen policies, and one success per policy on the same task-seed. All12 old pairs are near the longest geometry region. This is a floor effect, not evidence of policy equivalence. Giant's saturation findings are historical evidence from one map.
- Prior exact-prefix/replay evidence exists within its own backend and configuration. It does not prove CPU–TPU neural verifier parity, a new reward-device implementation, uniform graph accuracy or algorithm-level optimality.
- Forensics supports candidate support limitations (Fine needs node50; Manual's exact event multipliers differ from{2,8,32}) and fallback mismatch on some historical audits. Same-trajectory residual estimates have wide intervals. No reproducible composition-error correction is justified yet.

## Actual implementation and deployment state

Repo is /home/thanhlamtba31/TTS-reserach on the existing VM; remote HEAD9cee8a582abe2653af1add30ecc058ef5ad89d8b. The pinned CPU-torch/XLA stack is2.4.0; SD1.5 runs BF16, DDIM100 eta1. Backbone revision451f4fe16113bff5a5d2269ed5ad43b0592e9a14. Old neural ImageReward executes on CPU. That implementation is NOT compliant with the new TPU-only-neural requirement; no new scientific run may use it unchanged.

The old fallback schedule remains Transfer:[40,60,90],multipliers[8,8,32],N8. This is a recorded incumbent, not a universally best policy; Manual is stronger on some later audits. No production file has been changed. A new finite bank now represents exact Manual,Transfer,Fine and the existing shortlist variants, including node50. Historical claims stay attached to their original candidate classes and execution protocols.

Local work is on branch codex/tpu-research-loop-20261016. Pre-existing untracked experiment/scripts are preserved and not staged as this campaign's changes. No push, model upgrade, new resource, GPU job or neural CPU forward has been launched.

## Infrastructure and budget audit

Read-only metadata identifies project neurips2025-6631, zone us-central1-a, node tpu-v5p-8-01, hostname t1v-n-21e731f5-w-0, IP136.116.235.178. Metadata accelerator type and four physical /dev/vfio nodes corroborate4 chips; naming suffix alone is not used. Slice state READY. At this preflight no experiment processes/device owners were observed; a fresh check is still required immediately before any future launch. Root disk has only about7.5GiB free, /dev/shm about221GiB: new XLA caches must use a separate tmpfs namespace, without deleting existing artifacts. Other sd15-sinkhorn experiment directories exist on the VM; they are outside this campaign and were not read or modified.

Billing API read failed SERVICE_DISABLED. No API was enabled. Credits, account rate, remaining authorized allocation and billable slice exposure are unverified. Quota and READY do not imply free compute. The proposed16chip-hour ceiling is a technical cap only. Current effective execution allowance is0 pending human allocation and billing verification; GPU budget remains0. The existing externally managed READY slice may continue billing outside this job; no permission to stop it is assumed.

Budget guard reserves whole-slice occupancy conservatively as4 physical chips × elapsed wall-time during this campaign's job, with a separate actual-billing UNKNOWN field. It preserves unresolved reservations on restart, reserves40% for confirmation, enforces the PointMaze port subcap, and supports deadline/process-group kill and a KILL_SWITCH file. Active-device hours and a cloud invoice are not inferred from this conservative occupancy measure.

## What this turn established

- Revalidated historical SD1.5 and Ultra raw data and policy/report deduplication.
- Ran genuine adapter CPU tensor tests on the VM, without loading a neural model or initializing TPU. Exact Manual and explicit-multiplier events, Max history, SSP offspring, fixed shape and index semantics passed.
- Completed the primary paired prompt/seed variance diagnostic on576 historical rows; no new statistical independence is claimed. Within-seed noise accounts for about62–72% of per-prompt mean-difference variance at four seeds. Measured pairing reduces within-prompt variance by2.8–6.0× for these contrasts. All uncertainty statements are descriptive.
- Audited all19,701 Ultra unordered free-cell pairs against the map on CPU. Geodesic quantiles are1/23/49/85/160; selected8 new DEVELOPMENT pairs at exact lengths36 and67, matched by detour tier within each middle bin. No outcomes were evaluated. Grid connectedness is not dynamical reachability, and training-trajectory coverage is unavailable locally.
- Proved the elementary surrogate-regret and finite-temperature sensitivity bounds in THEORY.md and tested them numerically. Ran bounded-look Gaussian synthetic cases covering high noise, interaction and missing candidate optimum; none is a diffusion confirmation.

## Unresolved and blocked requirements

BUDGET_UNVERIFIED is the accelerator gate. TPU ImageReward FP32 parity and full denoiser/verifier correctness remain NOT RUN. A minimal golden-score reward gate is implemented but not deployed or invoked. PointMaze TPU port is NOT VALIDATED. Consequently no new Ultra policy pilot, graph, direct-search comparison or confirmation has run. Graph-only, v1 verification and direct full-policy search still need genuinely new selection data and equal all-in offline budget. The new covered candidate bank is development-derived; old audit rows cannot be reused to confirm it. No fresh prompt inventory/target distribution or independent compilation replicates have been opened. Current secondary independent reward metrics are NOT_EVALUATED.
