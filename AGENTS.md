# Research handoff

Start with README_WINDOWS_VI.md and docs/migration/AGENT_HANDOFF_VI.md.
Results live in the public Hugging Face dataset LamTNguyen/TQK-research-artifacts;
use tools/migration/restore_artifacts.py to inspect/download only needed files.
Read both the new migration catalog and the older STORAGE_MANIFEST.json.
Do not assume data pointed at an absent Extreme SSD symlink is on HF.

Keep completed experiment branches and their numerical protocols immutable.
New scientific changes belong on a new child branch. Account for scouting,
replay and failed compute; distinguish cold method cost from incremental reuse.
The Windows controller, CUDA notebook and remote TPU sampler are separate
execution environments. The existing TPU environment is pinned; do not upgrade
it as part of setup. Check device ownership and exact replay before new work.

Migration does not launch experiments or authorize new paid resources. Follow
the owner's current task, respect reserved splits, and keep secrets out of
public GitHub/HF artifacts. The original OpenResearch IDs are exported evidence;
a fresh machine needs its own local registration and SSH credentials.
