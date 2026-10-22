# Research handoff

Start with README_WINDOWS_VI.md, WINDOWS_COLAB_SETUP_PROMPT.md and
docs/migration/AGENT_HANDOFF_VI.md.
As of 2026-10-22 the owner no longer has access to the old TPU. Use Google
Colab GPU for future execution and Windows as controller. Do not SSH to old
TPU hosts/aliases, relaunch TPU runners, or silently rent Modal/cloud GPUs.
Results live in the public Hugging Face dataset LamTNguyen/TQK-research-artifacts;
use tools/migration/restore_artifacts.py to inspect/download only needed files.
Read the migration catalog, older STORAGE_MANIFEST.json, and
ssd_completion_20261022/SSD_CATALOG.json. Prefix ssd/ resolves the SSD completion
backup; use its completeness receipt rather than relying on old Mac symlinks.

Keep completed experiment branches and their numerical protocols immutable.
New scientific changes belong on a new child branch. Account for scouting,
replay and failed compute; distinguish cold method cost from incremental reuse.
Windows controller and Colab CUDA are separate environments. Historical TPU
numerics remain provenance, not an available backend or reusable CUDA cache.
Re-establish baselines and exact replay on CUDA before comparing methods.

Migration does not launch experiments or authorize new paid resources. Follow
the owner's current task, respect reserved splits, and keep secrets out of
public GitHub/HF artifacts. The original OpenResearch IDs are exported evidence;
a fresh machine needs its own local registration and upload credentials.
