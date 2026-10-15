# Resume contract

## Safe now: reproduce CPU diagnostics

From the local TTS-reserach checkout:

```bash
python3 experiments/tpu_research_loop_20261016/offline_loop.py \
  --workspace /Users/nguyenthanhlam/Documents/ChatGPT/TQK-research
python3 -m unittest discover -s experiments/tpu_research_loop_20261016 -p 'test_*.py' -v
python3 experiments/tpu_research_loop_20261016/budget_guard.py --check
```

The last command must fail BUDGET_UNVERIFIED until actual user allocation and billing conditions are confirmed. Existing derived outputs are immutable: re-running verifies deterministic equality; a changed scientific config requires a new round/output namespace. The historical input hashes must match. No model is loaded by these commands.

## Blocked accelerator gate: source prepared, not launched

`reward_tpu_gate.py` and `budget_guard.py` exist here. They require the existing VM's Python environment, cached pinned weights and saved golden images. They have not been run on TPU. Do not relaunch old graph/policy drivers: they use CPU neural ImageReward and do not enforce this new budget.

Before deployment: inspect `git status` and source commit, preserve all existing work, inspect actual device owners/processes, require verified allocation and billing conditions in authorization.json, and verify root/golden files. `authorization.example.json` is deliberately denying and is not evidence of permission. No credential, token or account upgrade is required or authorized. Compile caches go to a campaign-specific /dev/shm directory, not the nearly-full root disk.

For tracked accelerator compute use an OpenResearch SSH backend bound to this exact existing VM, not its configured default GPU provider. Read the installed `orx skill compute/ssh`, register this repository/node with the installed interface, commit the source snapshot and use `orx exp run` with the fixed contract below. There is currently no registered ORX project for this checkout, no accelerator run ID and no scheduled remote job for this campaign. Do not invent either. Copy authorization as external run state rather than committing it into a transported source snapshot; cumulative ledger must reside at one persistent campaign root on the VM, never at an ephemeral source extraction path.

The audited guarded command inside that immutable SSH run is:

```bash
/home/thanhlamtba31/static-bfs-dcs/venv/bin/python \
  experiments/tpu_research_loop_20261016/budget_guard.py \
  --root /home/thanhlamtba31/static-bfs-dcs/experiments/tpu_research_loop_20261016 \
  --phase SD15_smoke --max-seconds 600 \
  --entrypoint experiments/tpu_research_loop_20261016/reward_tpu_gate.py \
  -- --root /home/thanhlamtba31/static-bfs-dcs/experiments/tpu_research_loop_20261016 \
  --golden-manifest experiments/tpu_research_loop_20261016/reward_golden_manifest.json
```

Run this contract from the root of the verified actual ORX source checkout. The relative Python entrypoints exist in this source snapshot; no remote deployment or execution is being claimed. It reserves600seconds×4chips=.667chip-hour including cleanup, under the exploration allocation; no TPU initialization occurs if permission/ledger fails. Ten-minute termination is a gate cap, not a runtime forecast. A timeout/NaN/fallback/tolerance failure answers infrastructure only and must not trigger automatic fallback or threshold relaxation.

Kill only the campaign's child process group by creating `KILL_SWITCH` in that persistent campaign root. The guard terminates its own child before the deadline and records occupied slice time. It never stops/deletes the externally managed TPU VM. A missing handle requires authoritative inspection before a retry; unresolved reservation is charged at its full bound, not reset.

## Scientific comparison remains gated

After the TPU reward gate: complete denoiser,real resampling,counters,padding/masking,prefix and branch-RNG gates; measure full cold/warm throughput; freeze a new prompt/task inventory, selection/confirmation split, profiling budget and compiler rules. No historical audit is fresh selection. Use at least three independent compilation sets if budget permits, not bootstrap copies of one set; evaluate all outputs and account for shared confirmation prompts. Keep at least40%of actual allocation for confirmation/replication. A source-code gate passing is not permission to run the whole campaign.

PointMaze port has its separate min(1chip-hour,10%budget)cap. Its source/golden parity,variable-population shape/mask semantics and throughput are not validated. If it cannot pass within the cap, record INFRA_BLOCKED and keep SD1.5 as the main branch. No automatic graph or policy pilot is enabled by the present guard: only the reward gate entrypoint is allow-listed.

The guarded worker now arms a native real-time SIGALRM independently of the supervisor. Parent SIGTERM triggers worker cleanup; unresolved reservations still count at their full bound if the supervisor is forcibly killed. Use the latest committed source, not the initial guard snapshot. Entry-point validation requires the exact audited sibling path.
