# TTS research — Static BFS, OT-TTS và SD1.5

**Chuyển sang Windows:** đọc [README_WINDOWS_VI.md](README_WINDOWS_VI.md), rồi
[handoff cho agent](docs/migration/AGENT_HANDOFF_VI.md).

- [Kết quả và raw trên Hugging Face](https://huggingface.co/datasets/LamTNguyen/TQK-research-artifacts).
- [Công cụ restore theo checksum](tools/migration/restore_artifacts.py), hỗ trợ Windows/Linux/macOS.
- [Notebook Best-of-N SD1.5](notebooks/best_of_n/README_VI.md): CUDA hoặc kernel trên TPU.
- [Cây và run OpenResearch](docs/migration/OPENRESEARCH_STATE.json).

Code và protocol nằm trong Git; dữ liệu lớn tải theo nhu cầu từ HF. Các branch
experiment đã hoàn tất được giữ nguyên. Không cần MacBook để phục hồi artifact.
Một số kết quả chỉ có trên SSD vắng mặt được [liệt kê rõ](docs/migration/SSD_DEPENDENCIES.md).

Phần dưới là README xuất bản Static BFS trước đây; các giới hạn/kết quả hiện tại
đọc trong handoff và report đúng experiment, không suy từ mô tả lịch sử này.

---

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
