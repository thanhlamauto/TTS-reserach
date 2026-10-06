# Static BFS Graph research code

This repository publishes the Static BFS Graph SD1.5 experiments, their Modal
GPU runners, compact result reports, and instructions for reproducing them.
The full [experiment README](scripts/static_bfs_graph/README.md) explains the
92-edge graph, results, uncertainty, frozen prompt splits, and execution order.

The code was developed in
[`NguyenNgocMinh30012005/arc-modulo`](https://github.com/NguyenNgocMinh30012005/arc-modulo)
on branch `ngocminh` from commit
`34f17bd984921aeba046e8a5b7936e4a7a8346c0`. This standalone copy also
includes the SD1.5 baseline runner and config that the graph imports. The
official FKD implementation is fetched separately at commit
`c158ab9eb058adb24ad0f8d48506ab837d7fae91` during reproduction;
model weights and third-party source are not committed here.

Start with the [Modal offline compiler runbook](modal_n8_gpu/OFFLINE_RUNBOOK.md)
and the [experiment README](scripts/static_bfs_graph/README.md) before running
any billable GPU stage. Set `STATIC_BFS_OFFICIAL_ROOT` to the pinned official
checkout and `MODAL_PROFILE` to your own authorized workspace. Historical
result files and raw rollouts are not required to *read* the reports, but some
later-stage analyses require the frozen artifacts produced by earlier stages.

The most important current limitation is that the N=8 dense reference was
measured on graph-fit prompts only. It was not compared head-to-head with the
three N=8 policies on the 48 held-out TRAIN prompts. The reference is a fixed
anchor for edge interventions, not a known optimum. No new GPU experiment was
run as part of publishing this repository.

The [TPU v5p-8 port runbook](scripts/static_bfs_graph/TPU_PORT.md) describes
the separate checkout, pinned environment, and frozen paired dense-reference
diagnostic. Merely preparing the port does not run the diagnostic.
