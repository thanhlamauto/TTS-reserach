"""Frozen TRAIN-only, paired N=8 dense-reference diagnostic on TPU v5p-8.

This is a diagnostic of the reference assumption, not graph fitting, search,
retuning, VALIDATION/TEST evaluation, or a new policy-selection procedure.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess

from .graph_builder import dense_reference
from .policy import StaticBFSPolicy


REPO = Path(__file__).resolve().parents[2]
POOL = REPO / "modal_n8_gpu/train_pool.json"
CONFIG = REPO / "configs/static_bfs_graph_t2i.json"
BASELINE = REPO / "configs/diffusion_classical_search_t2i_table12.json"
PROTOCOL = "STATIC_BFS_N8_DENSE_REFERENCE_PAIRED_TPU_V1"
RUN_ID = "n8-dense-reference-paired-tpu-v1"
SEEDS = (42, 43, 44, 45)
FIT = (0, 2, 3, 4, 5, 11, 14, 15, 16, 18, 19, 20)
HELDOUT = (21, 27, 29, 30, 32, 33, 35, 36, 37, 39, 40, 41, 43, 45,
           47, 48, 49, 51, 54, 56, 57, 58, 59, 60, 62, 63, 65, 66,
           68, 69, 71, 73, 74, 75, 76, 77, 79, 80, 81, 84, 85, 87,
           88, 89, 92, 97, 98, 99)
STEP_GRID = (10, 20, 30, 40, 60, 80, 90)
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20261006


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, value) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError(f"frozen file changed: {path}")
    else:
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(raw)
        tmp.replace(path)


def records() -> dict:
    ref = dense_reference(STEP_GRID, 8.0, 8)
    transfer = StaticBFSPolicy((40, 60, 90),
                               ((40, 8.0), (60, 8.0), (90, 32.0)), 8)
    retuned = StaticBFSPolicy((40, 60, 90),
                              ((40, 2.0), (60, 8.0), (90, 32.0)), 8)
    if (transfer.id, retuned.id) != ("static_df7ddcbdb5f12bd4",
                                      "static_220ce94dc634388e"):
        raise RuntimeError("frozen N=8 policy IDs changed")
    def static(p):
        return {"kind": "static", "id": p.id, **p.record()}
    manual = {"kind": "manual", "id": "manual_bfs_g0p024_n8",
              "resampling_steps": [20, 40, 80], "base_temperature": 10.0,
              "tempering": "increase", "gamma": 0.024,
              "particles": 8, "selection_mode": "manual"}
    return {"DENSE_REFERENCE": static(ref), "MANUAL8": manual,
            "TRANSFER4_TO_8": static(transfer), "RETUNED8": static(retuned)}


def checked_inputs() -> tuple[dict, dict, dict]:
    pool, cfg, old = read(POOL), read(CONFIG), read(BASELINE)
    if (tuple(pool["train_indices"]) != FIT + HELDOUT or
        sorted(map(int, pool["prompts"])) != list(FIT + HELDOUT) or
        pool["benchmark_sha256"] != old["benchmark"]["sha256"] or
        cfg["particles"] != 4 or cfg["steps"] != 100 or cfg["eta"] != 1.0 or
        cfg["scoring"] != "max" or cfg["resampling"] != "ssp" or
        cfg["execution_dtype"] != "bfloat16" or
        cfg["pilot"]["candidate_steps"] != list(STEP_GRID) or
        cfg["pilot"]["reference_tau"] != 8.0 or
        old["models"]["sd15"]["revision"] !=
        "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"):
        raise RuntimeError("frozen TRAIN split or SD1.5 protocol changed")
    return pool, cfg, old


def freeze(root: Path) -> dict:
    checked_inputs()
    items = records()
    plan = {"protocol": PROTOCOL, "run_id": RUN_ID,
            "purpose": "same-backend dense-reference versus frozen N8 policies",
            "fit_12_not_scored": list(FIT), "heldout_train_48": list(HELDOUT),
            "seeds": list(SEEDS), "policies": items,
            "model": "SD1.5", "N": 8, "DDIM_steps": 100, "eta": 1.0,
            "dtype": "bfloat16", "scoring": "Max", "resampling": "SSP",
            "verifier": "ImageReward-v1.0", "GPU_data_mixed": False,
            "validation_touched": False, "test_touched": False,
            "external_pool_touched": False, "search_or_retune": False,
            "previously_explored_TRAIN_prompts": True,
            "diffusion_NFE_per_policy": 800,
            "reference_verifier_particle_scores": 64,
            "K3_verifier_particle_scores": 32,
            "bootstrap_replicates": BOOTSTRAPS,
            "bootstrap_seed": BOOTSTRAP_SEED}
    write_once(root / "plan.json", plan)
    manifest = {"plan_sha256": digest(root / "plan.json"),
                "code_sha256": digest(Path(__file__)),
                "runner_sha256": digest(REPO / "scripts/static_bfs_graph/runner.py"),
                "adapter_sha256": digest(REPO / "scripts/static_bfs_graph/adapter.py"),
                "config_sha256": digest(CONFIG),
                "baseline_config_sha256": digest(BASELINE),
                "train_pool_sha256": digest(POOL),
                "baseline_runner_sha256": digest(
                    REPO / "scripts/diffusion_classical_search_t2i/tpu_runner.py"),
                "repo_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()}
    write_once(root / "manifest.json", manifest)
    return {"plan_sha256": manifest["plan_sha256"], "prompt_count": 48,
            "seed_count": 4, "policy_count": 4}


def verify(root: Path) -> tuple[dict, dict, dict, dict]:
    pool, cfg, old = checked_inputs()
    plan, manifest = read(root / "plan.json"), read(root / "manifest.json")
    if (plan["protocol"] != PROTOCOL or plan["heldout_train_48"] != list(HELDOUT) or
        plan["policies"] != records() or digest(root / "plan.json") != manifest["plan_sha256"] or
        digest(Path(__file__)) != manifest["code_sha256"] or
        digest(REPO / "scripts/static_bfs_graph/runner.py") != manifest["runner_sha256"] or
        digest(REPO / "scripts/static_bfs_graph/adapter.py") != manifest["adapter_sha256"] or
        digest(CONFIG) != manifest["config_sha256"] or
        digest(BASELINE) != manifest["baseline_config_sha256"] or
        digest(REPO / "scripts/diffusion_classical_search_t2i/tpu_runner.py") !=
        manifest["baseline_runner_sha256"] or
        digest(POOL) != manifest["train_pool_sha256"]):
        raise RuntimeError("frozen plan or scientific source changed")
    return pool, cfg, old, plan


def run_worker(root: Path, dcs_root: Path, stage: str, seed: int, chip: int) -> dict:
    if (stage not in {"smoke", "reference", "policies"} or
        seed not in SEEDS or chip != SEEDS.index(seed)):
        raise ValueError("one paired seed per physical chip is required")
    if stage == "smoke" and (seed, chip) != (42, 0):
        raise ValueError("smoke is fixed to graph-fit prompt 0, seed 42, chip 0")
    pool, cfg, old, plan = verify(root)
    if not (dcs_root / "official/text_to_image/fkd_diffusers").is_dir():
        raise RuntimeError("pinned official FKD checkout is absent")
    from .runner import Runtime
    cfg = {**cfg, "particles": 8, "protocol_version": PROTOCOL}
    prompts = [{"prompt": pool["prompts"].get(str(i))} for i in range(100)]
    target = root / "smoke" if stage == "smoke" else root
    runtime = Runtime(target, dcs_root, seed, chip, cfg, old, prompts)
    reference = dense_reference(STEP_GRID, 8.0, 8)
    items = plan["policies"]
    prompt_ids = (FIT[0],) if stage == "smoke" else HELDOUT
    names = ("DENSE_REFERENCE", "MANUAL8", "TRANSFER4_TO_8", "RETUNED8")
    if stage == "reference":
        names = names[:1]
    elif stage == "policies":
        names = names[1:]
    completed = 0
    for prompt_index in prompt_ids:
        for name in names:
            label = ("n8_smoke" if stage == "smoke" else
                     "n8_reference" if name == "DENSE_REFERENCE" else "n8_policy")
            runtime.run(label, items[name], prompt_index, reference)
            completed += 1
    return {"stage": stage, "seed": seed, "chip": chip,
            "prompt_count": len(prompt_ids), "rows": completed}


def load_scores(root: Path, plan: dict):
    wanted = {rec["id"]: name for name, rec in plan["policies"].items()}
    units = {}
    total = 0
    for stage in ("n8_reference", "n8_policy"):
        for path in (root / "raw" / stage / "n8").glob("seed*/*.json"):
            row = read(path)
            prompt, seed, pid = row["prompt_index"], row["trial_seed"], row["policy_id"]
            if (prompt not in HELDOUT or seed not in SEEDS or pid not in wanted or
                row["unit_key"]["stage"] != stage or
                (pid == plan["policies"]["DENSE_REFERENCE"]["id"]) !=
                (stage == "n8_reference") or row["diffusion_NFE"] != 800 or
                row["execution_dtype"] != "torch.bfloat16" or
                row["verifier_calls"] != (64 if stage == "n8_reference" else 32) or
                not math.isfinite(row["selected_final_score"])):
                raise RuntimeError(f"invalid rollout: {path}")
            key = (prompt, seed, wanted[pid])
            if key in units:
                raise RuntimeError(f"duplicate rollout: {key}")
            units[key] = row
            total += 1
    if total != len(HELDOUT) * len(SEEDS) * len(wanted):
        raise RuntimeError(f"incomplete paired rollouts: {total}/768")
    for p in HELDOUT:
        for seed in SEEDS:
            rows = [units[p, seed, name] for name in plan["policies"]]
            if len({r["initial_generator_state_sha256"] for r in rows}) != 1:
                raise RuntimeError(f"initial RNG is not paired: {p},{seed}")
    return units


def analyze(root: Path) -> dict:
    _, _, _, plan = verify(root)
    units = load_scores(root, plan)
    names = list(plan["policies"])
    means = {name: [statistics.mean(units[p, s, name]["selected_final_score"]
                                     for s in SEEDS) for p in HELDOUT] for name in names}
    rng = random.Random(BOOTSTRAP_SEED)
    samples = [[rng.randrange(len(HELDOUT)) for _ in HELDOUT]
               for _ in range(BOOTSTRAPS)]
    comparisons = []
    for name in names[1:]:
        differences = [a-b for a, b in zip(means[name], means["DENSE_REFERENCE"])]
        boot = sorted(statistics.mean(differences[i] for i in sample) for sample in samples)
        comparisons.append({"method": name, "delta_vs_reference": statistics.mean(differences),
                            "ci95": [boot[int(.025 * BOOTSTRAPS)],
                                     boot[int(.975 * BOOTSTRAPS)]],
                            "prompt_wins": sum(d > 0 for d in differences),
                            "prompt_ties": sum(d == 0 for d in differences),
                            "prompt_losses": sum(d < 0 for d in differences)})
    result = {"protocol": PROTOCOL, "plan_sha256": digest(root / "plan.json"),
              "paired_prompt_seed_units": len(HELDOUT) * len(SEEDS),
              "raw_rollouts": len(units), "prompt_level_bootstraps": BOOTSTRAPS,
              "mean_ImageReward": {name: statistics.mean(x) for name, x in means.items()},
              "comparisons": comparisons, "validation_touched": False,
              "test_touched": False, "previously_explored_TRAIN_prompts": True}
    write_once(root / "results.json", result)
    lines = ["# N=8 dense reference: paired TPU TRAIN diagnostic", "",
             "48 held-out TRAIN prompts × four seeds; SD1.5 BF16, DDIM100 eta=1,",
             "N=8, Max+SSP, ImageReward. Same 800 diffusion NFE per policy;",
             "the dense reference uses 64 verifier particle scores versus 32",
             "for each K=3 policy. This is not a verifier-budget-matched claim.", "",
             "| Method | Mean IR | Delta vs reference | 95% prompt-bootstrap CI | W/T/L |",
             "| --- | ---: | ---: | --- | --- |"]
    lines.append(f"| DENSE_REFERENCE | {result['mean_ImageReward']['DENSE_REFERENCE']:.6f} | — | — | — |")
    for row in comparisons:
        lo, hi = row["ci95"]
        lines.append(f"| {row['method']} | {result['mean_ImageReward'][row['method']]:.6f} | "
                     f"{row['delta_vs_reference']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | "
                     f"{row['prompt_wins']}/{row['prompt_ties']}/{row['prompt_losses']} |")
    lines += ["", "Prompts were previously used in TRAIN diagnostics. No VALIDATION,",
              "TEST, external pool, search, or retuning was used. TPU results are",
              "paired within this run; do not numerically pair them with GPU runs.", ""]
    report = "\n".join(lines)
    path = root / "REPORT.md"
    if path.exists() and path.read_text() != report:
        raise RuntimeError("existing report differs")
    path.write_text(report)
    return result


def status(root: Path) -> dict:
    counts = {stage: len(list((root / "raw" / stage / "n8").glob("seed*/*.json")))
              for stage in ("n8_reference", "n8_policy")}
    return {"run": str(root), "expected_reference": 192,
            "expected_policies": 576, **counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("plan", "smoke", "reference", "policies", "status", "analyze"))
    parser.add_argument("--run-root", type=Path, default=REPO / "results/static-bfs-graph" / RUN_ID)
    parser.add_argument("--dcs-root", type=Path, default=Path.home() / "static-bfs-dcs")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--chip", type=int)
    args = parser.parse_args()
    root = args.run_root.resolve()
    if args.stage == "plan":
        output = freeze(root)
    elif args.stage in ("smoke", "reference", "policies"):
        if args.seed is None or args.chip is None:
            parser.error("worker stages require --seed and --chip")
        output = run_worker(root, args.dcs_root.resolve(), args.stage, args.seed, args.chip)
    elif args.stage == "analyze":
        output = analyze(root)
    else:
        output = status(root)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
