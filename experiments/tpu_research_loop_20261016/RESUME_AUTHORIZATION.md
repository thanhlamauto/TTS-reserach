# Owner-authorized continuation, 16 October 2026

The owner explicitly requested continuation and stated that the existing TPU is
not charged to them. This is accepted as owner attestation for use of the existing
VM; it is not independent verification of a Cloud invoice, credits or rate.
Actual provider billing remains UNKNOWN. No resource creation, GPU job, model
upgrade or VM lifecycle operation is authorized by this continuation.

Adopt the previously proposed technical ceiling of16 physical-chip-hours for
this Static-BFS campaign, of which at least40percent is reserved for
confirmation/replication. The old offline phase used0 accelerator hours. The
unrelated DINOv2 pilot is preserved and is not resumed by this experiment.

One persistent campaign clock starts at the first accelerator reservation and
expires after4hours for the verified4-chip slice. Exploration may not extend
beyond60percent of that wall allowance. Compile, retries, evaluation and idle
time between campaign jobs consume the same clock. Restarting a process or
changing an experiment branch does not reset it. The cumulative job reservation
ledger is preserved independently, conservatively retaining unresolved leases.

First action is the existing TPU FP32 ImageReward parity gate with its original
atol=rtol=.001, on8 frozen saved RGB images and CPU golden scores. The gate has
a600second maximum including cleanup. No scientific comparison runs until it
passes. Keep exact Manual, Transfer and Fine semantics and the covered6-policy
bank; no objective or temperature retuning is authorized implicitly.

User request source: attachment094d5ed5-9f82-426e-af74-b3b8b553ab94 and the
accompanying message requesting continuation on the no-charge existing TPU.
External authorization and the mutable cost ledger live only at
`/home/thanhlamtba31/static-bfs-dcs/experiments/tpu_research_loop_20261016`.
No confirmation data has been opened by this authorization record.
