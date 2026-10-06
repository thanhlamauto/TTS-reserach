"""Paired TRAIN-only diagnostic of a 10-bin BFS resampling grid on TPU.

This compares Dense10, ten single-midpoint additions, and Dense20 at fixed
N=8, tau=8. It does not fit a graph or change a deployed BFS policy.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess

from .adapter import RNG_PROTOCOL, event_seed
from .policy import StaticBFSPolicy
from .runner import Runtime, _unit_key, digest as unit_digest, static_record
from .tpu_reference_compare import BASELINE, CONFIG, POOL, REPO, checked_inputs, digest, read, write_once


PROTOCOL = "STATIC_BFS_GRID_DENSITY_10BIN_TPU_V1"
RUN_ID = "grid-density-10bin-pilot-v1"
SEED = 42
N_PROMPTS = 20
CHIPS = 4
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20261006
BASE_STEPS = tuple(range(10, 100, 10))
MIDPOINTS = tuple(range(5, 100, 10))
DENSE20_STEPS = tuple(range(5, 100, 5))
STAGES = {"Dense10": "grid_dense10", "Dense20": "grid_dense20"}


def policies() -> dict[str, StaticBFSPolicy]:
    out = {"Dense10": StaticBFSPolicy(BASE_STEPS, tuple((s, 8.0) for s in BASE_STEPS), 8)}
    for mid in MIDPOINTS:
        steps = tuple(sorted((*BASE_STEPS, mid)))
        out[f"Midpoint{mid:02d}"] = StaticBFSPolicy(steps, tuple((s, 8.0) for s in steps), 8)
    out["Dense20"] = StaticBFSPolicy(DENSE20_STEPS, tuple((s, 8.0) for s in DENSE20_STEPS), 8)
    return out


def static_gate() -> dict:
    methods = policies()
    if (BASE_STEPS != (10, 20, 30, 40, 50, 60, 70, 80, 90) or
        MIDPOINTS != (5, 15, 25, 35, 45, 55, 65, 75, 85, 95) or
        DENSE20_STEPS != tuple(range(5, 100, 5)) or len(methods) != 12):
        raise RuntimeError("normalized progress to zero-based sampler-index mapping changed")
    expected_progress = tuple(k / 20 for k in range(1, 20))
    if tuple(s / 100 for s in DENSE20_STEPS) != expected_progress:
        raise RuntimeError("Dense20 normalized progress mapping changed")
    base_set = set(BASE_STEPS)
    for mid in MIDPOINTS:
        policy = methods[f"Midpoint{mid:02d}"]
        if (set(policy.resampling_steps) - base_set != {mid} or
            base_set - set(policy.resampling_steps) or
            len(policy.resampling_steps) != 10 or
            tuple(t for _, t in policy.temperature_by_step) != (8.0,) * 10 or
            not math.isclose(mid / 100, (mid // 10 + .5) / 10, abs_tol=1e-12)):
            raise RuntimeError(f"midpoint {mid} is not one exact tau=8 addition")
    for name, policy in methods.items():
        if (policy.particles != 8 or policy.selection_mode != "raw_tau" or
            any(t != 8.0 for _, t in policy.temperature_by_step)):
            raise RuntimeError(f"non-temporal BFS setting changed: {name}")
    fixed_records = [{k: v for k, v in static_record(policy).items()
                      if k not in {"id", "resampling_steps", "temperature_by_step"}}
                     for policy in methods.values()]
    if any(record != fixed_records[0] for record in fixed_records[1:]):
        raise RuntimeError("non-temporal policy configuration differs")
    return {"passed": True, "sampler_steps": 100, "normalized_boundaries":
            [s / 100 for s in BASE_STEPS], "normalized_midpoints":
            [s / 100 for s in MIDPOINTS], "policy_count": len(methods),
            "constant_non_temporal_settings": {"N": 8, "tau": 8.0,
                 "selection_mode": "raw_tau", "scoring": "Max", "resampling": "SSP"}}


def inputs():
    pool, cfg, baseline = checked_inputs()
    prompt_ids = tuple(pool["train_indices"][:N_PROMPTS])
    if len(prompt_ids) != N_PROMPTS or prompt_ids != tuple(sorted(prompt_ids)):
        raise RuntimeError("frozen 20 TRAIN prompt prefix changed")
    return pool, cfg, baseline, prompt_ids


def source_hashes():
    paths = {"driver": Path(__file__),
             "runner": REPO / "scripts/static_bfs_graph/runner.py",
             "adapter": REPO / "scripts/static_bfs_graph/adapter.py",
             "policy": REPO / "scripts/static_bfs_graph/policy.py",
             "baseline_runner": REPO / "scripts/diffusion_classical_search_t2i/tpu_runner.py",
             "config": CONFIG, "baseline_config": BASELINE, "train_pool": POOL}
    return {key: digest(path) for key, path in paths.items()}


def freeze(root: Path):
    _, cfg, baseline, prompt_ids = inputs()
    gate = static_gate()
    methods = policies()
    plan = {"protocol": PROTOCOL, "run_id": RUN_ID, "purpose": "grid-density diagnostic only",
            "prompt_split": "TRAIN", "prompt_indices": list(prompt_ids), "seed": SEED,
            "N": 8, "sampler": "DDIM", "sampler_steps": 100, "eta": 1.0,
            "dtype": "bfloat16", "scoring": "Max", "resampling": "SSP",
            "verifier": "ImageReward-v1.0", "tau": 8.0,
            "progress_definition": "zero at initial noise; one at final sample",
            "progress_to_index": "index = 100 * normalized_progress",
            "base_steps": list(BASE_STEPS), "midpoints": list(MIDPOINTS),
            "dense20_steps": list(DENSE20_STEPS),
            "policies": {name: static_record(policy) for name, policy in methods.items()},
            "chip_assignment": "prompt_indices[chip::4]", "physical_chips": CHIPS,
            "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": BOOTSTRAP_SEED,
            "expected_logical_rollouts": N_PROMPTS * len(methods),
            "diffusion_NFE_per_rollout": 800,
            "historical_TRAIN_reuse": True, "validation_touched": False,
            "test_touched": False, "external_pool_touched": False,
            "policy_search_or_tuning": False}
    write_once(root / "config.json", plan)
    write_once(root / "static_gate.json", gate)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    manifest = {"plan_sha256": digest(root / "config.json"),
                "static_gate_sha256": digest(root / "static_gate.json"),
                "source_sha256": source_hashes(), "repo_commit": commit,
                "official_source_commit": baseline["official_source"]["commit"],
                "model_revision": baseline["models"]["sd15"]["revision"]}
    write_once(root / "manifest.json", manifest)
    return {"plan_sha256": manifest["plan_sha256"], "prompts": N_PROMPTS,
            "seed": SEED, "policies": len(methods), "logical_rollouts": N_PROMPTS * len(methods)}


def verify(root: Path):
    pool, cfg, baseline, prompt_ids = inputs()
    plan, manifest = read(root / "config.json"), read(root / "manifest.json")
    if (plan["protocol"] != PROTOCOL or plan["prompt_indices"] != list(prompt_ids) or
        plan["seed"] != SEED or plan["policies"] !=
            {name: static_record(policy) for name, policy in policies().items()} or
        digest(root / "config.json") != manifest["plan_sha256"] or
        digest(root / "static_gate.json") != manifest["static_gate_sha256"] or
        read(root / "static_gate.json") != static_gate() or
        source_hashes() != manifest["source_sha256"]):
        raise RuntimeError("frozen grid-density plan or scientific source changed")
    cfg = {**cfg, "particles": 8, "protocol_version": PROTOCOL}
    return pool, cfg, baseline, plan, prompt_ids


def stage_for(name: str) -> str:
    return "grid_midpoint" if name.startswith("Midpoint") else STAGES[name]


def record_path(root: Path, cfg: dict, name: str, policy: StaticBFSPolicy, prompt: int):
    rec = static_record(policy)
    stage = stage_for(name)
    reference = policies()["Dense10"]
    key = _unit_key(cfg, reference, rec, prompt, SEED, stage)
    unit_hash = unit_digest(key)
    return root / "raw" / stage / "n8" / f"seed{SEED}" / f"{unit_hash}.json", unit_hash


def check_row(root: Path, cfg: dict, name: str, prompt: int, row: dict):
    policy = policies()[name]
    _, unit_hash = record_path(root, cfg, name, policy, prompt)
    if (row.get("unit_hash") != unit_hash or row.get("policy_id") != policy.id or
        row.get("prompt_index") != prompt or row.get("trial_seed") != SEED or
        row.get("diffusion_NFE") != 800 or row.get("rng_protocol") != RNG_PROTOCOL or
        row.get("execution_dtype") != "torch.bfloat16" or
        len(row.get("final_particle_scores", [])) != 8 or
        row.get("verifier_calls") != (len(policy.resampling_steps) + 1) * 8 or
        [e["sampling_index"] for e in row.get("resampling_events", [])] !=
            list(policy.resampling_steps) or
        any(e["temperature"] != 8.0 or e["event_seed"] !=
            event_seed(SEED, prompt, e["sampling_index"])
            for e in row["resampling_events"]) or
        not math.isfinite(row["selected_final_score"])):
        raise RuntimeError(f"invalid rollout: {name}, prompt {prompt}")


def runner(root: Path, dcs_root: Path, mode: str, chip: int):
    if chip not in range(CHIPS) or (mode == "smoke" and chip != 0):
        raise ValueError("invalid chip for stage")
    pool, cfg, baseline, plan, prompt_ids = verify(root)
    official = dcs_root / "official"
    official_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=official, text=True).strip()
    if official_commit != baseline["official_source"]["commit"]:
        raise RuntimeError("pinned official FKD source changed")
    if os.environ.get("PJRT_DEVICE") != "TPU":
        raise RuntimeError("PJRT_DEVICE=TPU is required")
    if mode == "full":
        smoke = read(root / "smoke_gate.json")
        if smoke.get("passed") is not True or smoke.get("plan_sha256") != digest(root / "config.json"):
            raise RuntimeError("manual one-prompt smoke gate has not passed")
    own_prompts = (prompt_ids[0],) if mode == "smoke" else prompt_ids[chip::CHIPS]
    names = ("Dense10", "Midpoint45", "Dense20") if mode == "smoke" else tuple(policies())
    prompt_records = [{"prompt": pool["prompts"].get(str(i))} for i in range(100)]
    runtime = Runtime(root, dcs_root, SEED, chip, cfg, baseline, prompt_records)
    out = []
    for prompt in own_prompts:
        for name in names:
            policy = policies()[name]
            row = runtime.run(stage_for(name), static_record(policy), prompt,
                              policies()["Dense10"])
            check_row(root, cfg, name, prompt, row)
            out.append((name, prompt))
    if mode == "smoke":
        rows = {}
        for name in names:
            path, _ = record_path(root, cfg, name, policies()[name], prompt_ids[0])
            rows[name] = read(path)
        if len({r["initial_generator_state_sha256"] for r in rows.values()}) != 1:
            raise RuntimeError("initial RNG differs across smoke policies")
        if len({r["prompt_seed"] for r in rows.values()}) != 1:
            raise RuntimeError("prompt seed differs across smoke policies")
        event_sequences = {name: [e["sampling_index"] for e in rows[name]["resampling_events"]]
                           for name in names}
        if (event_sequences["Dense10"] != list(BASE_STEPS) or
            event_sequences["Midpoint45"] != list(sorted((*BASE_STEPS, 45))) or
            event_sequences["Dense20"] != list(DENSE20_STEPS)):
            raise RuntimeError("manual event sequence gate failed")
        gate = {"passed": True, "plan_sha256": digest(root / "config.json"),
                "prompt_index": prompt_ids[0], "seed": SEED,
                "initial_generator_state_sha256":
                    rows["Dense10"]["initial_generator_state_sha256"],
                "event_sequences": event_sequences,
                "all_temperatures": 8.0,
                "non_temporal_config_identical": True,
                "scores": {name: rows[name]["selected_final_score"] for name in names}}
        write_once(root / "smoke_gate.json", gate)
        print("COMPLETE_EVENT_SEQUENCES", json.dumps(event_sequences), flush=True)
    return {"mode": mode, "chip": chip, "prompt_indices": list(own_prompts),
            "checked_units": len(out), "plan_sha256": digest(root / "config.json")}


def status(root: Path):
    return {"run": str(root), "dense10": len(list((root / "raw/grid_dense10/n8").glob("seed*/*.json"))),
            "midpoint": len(list((root / "raw/grid_midpoint/n8").glob("seed*/*.json"))),
            "dense20": len(list((root / "raw/grid_dense20/n8").glob("seed*/*.json"))),
            "failures": sum(len(list((root / "failures" / stage).glob("*.json")))
                            for stage in ("grid_dense10", "grid_midpoint", "grid_dense20"))}


def account(root: Path, start_unix: float, end_unix: float):
    verify(root)
    counts = status(root)
    if ((counts["dense10"], counts["midpoint"], counts["dense20"], counts["failures"]) !=
        (20, 200, 20, 0) or not start_unix < end_unix):
        raise RuntimeError(f"cannot account for incomplete run: {counts}")
    wall = end_unix - start_unix
    rate = 4.20
    result = {"protocol": PROTOCOL, "plan_sha256": digest(root / "config.json"),
              "started_utc": datetime.fromtimestamp(start_unix, timezone.utc).isoformat(),
              "finished_utc": datetime.fromtimestamp(end_unix, timezone.utc).isoformat(),
              "wall_seconds": wall, "wall_hours": wall / 3600,
              "physical_tpu_chips": CHIPS, "allocated_tpu_chip_hours": CHIPS * wall / 3600,
              "gpu_hours": 0.0, "logical_policy_rollouts": 240,
              "actual_executed_rollouts": counts["dense10"] + counts["midpoint"] + counts["dense20"],
              "shared_tree_or_snapshot_reuse": False,
              "total_diffusion_NFE": 240 * 800,
              "illustrative_public_v5p_rate_usd_per_chip_hour": rate,
              "rate_source": "https://cloud.google.com/tpu/pricing",
              "rate_region_note": "public us-east1 reference rate; actual us-central1/account rate unknown",
              "illustrative_cost_usd": CHIPS * wall / 3600 * rate,
              "actual_billed_cost_usd": None,
              "billing_note": "TPU node billing may include READY idle time outside this experiment"}
    write_once(root / "run_accounting.json", result)
    return result


def paired_stats(base: list[float], refined: list[float], seed: int):
    deltas = [b - a for a, b in zip(base, refined)]
    n = len(deltas)
    rng = random.Random(seed)
    draws = sorted(statistics.mean(deltas[rng.randrange(n)] for _ in range(n))
                   for _ in range(BOOTSTRAPS))
    return {"mean_dense10": statistics.mean(base),
            "mean_refined": statistics.mean(refined),
            "mean_delta": statistics.mean(deltas),
            "se": statistics.stdev(deltas) / math.sqrt(n),
            "ci95": [draws[int(.025 * BOOTSTRAPS)], draws[int(.975 * BOOTSTRAPS)]],
            "wins": sum(x > 0 for x in deltas),
            "ties": sum(x == 0 for x in deltas),
            "losses": sum(x < 0 for x in deltas)}


def analyze(root: Path):
    _, cfg, _, plan, prompt_ids = verify(root)
    counts = status(root)
    if ((counts["dense10"], counts["midpoint"], counts["dense20"], counts["failures"]) !=
        (20, 200, 20, 0)):
        raise RuntimeError(f"incomplete grid-density experiment: {counts}")
    by_name = {}
    records = {}
    for name, policy in policies().items():
        rows = []
        for prompt in prompt_ids:
            path, _ = record_path(root, cfg, name, policy, prompt)
            row = read(path)
            check_row(root, cfg, name, prompt, row)
            rows.append(row)
        by_name[name] = rows
        records[name] = [row["selected_final_score"] for row in rows]
    for index, prompt in enumerate(prompt_ids):
        if (len({by_name[name][index]["initial_generator_state_sha256"] for name in by_name}) != 1 or
            len({by_name[name][index]["prompt_seed"] for name in by_name}) != 1):
            raise RuntimeError(f"unpaired prompt/seed at {prompt}")
    accounting = read(root / "run_accounting.json")
    if (accounting["protocol"] != PROTOCOL or accounting["plan_sha256"] !=
        digest(root / "config.json") or accounting["logical_policy_rollouts"] != 240 or
        accounting["actual_executed_rollouts"] != 240):
        raise RuntimeError("run accounting differs from frozen workload")
    for fname, names in (("dense10_records.jsonl", ("Dense10",)),
                         ("midpoint_records.jsonl", tuple(f"Midpoint{m:02d}" for m in MIDPOINTS)),
                         ("dense20_records.jsonl", ("Dense20",))):
        lines = [json.dumps({"method": name, "prompt_id": row["prompt_index"],
                             "seed": row["trial_seed"], "terminal_ImageReward": row["selected_final_score"],
                             "runtime_seconds": row["elapsed_seconds"],
                             "resampling_event_log": row["resampling_events"],
                             "initial_generator_state_sha256": row["initial_generator_state_sha256"],
                             "raw_unit_hash": row["unit_hash"]}, sort_keys=True)
                 for name in names for row in by_name[name]]
        path = root / fname
        content = "\n".join(lines) + "\n"
        if path.exists() and path.read_text() != content:
            raise RuntimeError(f"existing {fname} differs")
        path.write_text(content)
    summaries = []
    for k, mid in enumerate(MIDPOINTS):
        item = paired_stats(records["Dense10"], records[f"Midpoint{mid:02d}"], BOOTSTRAP_SEED + k)
        summaries.append({"interval": f"{k/10:.1f}-{(k+1)/10:.1f}",
                          "midpoint_progress": mid / 100, "midpoint_index": mid,
                          **item, "refinement_flag": item["mean_delta"] > .01 or item["ci95"][0] > 0})
    global_item = paired_stats(records["Dense10"], records["Dense20"], BOOTSTRAP_SEED + 20)
    csv_path = root / "interval_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    plot_grid(root / "grid_density_plot.png", summaries, global_item)
    mean_max = max(s["mean_delta"] for s in summaries)
    if global_item["mean_delta"] > .01 and global_item["ci95"][0] > 0:
        outcome = "C. Dense20 globally improves enough that the 10-bin grid looks too coarse."
    elif any(s["ci95"][0] > 0 for s in summaries):
        outcome = "B. One/few localized intervals merit refinement."
    elif all(abs(s["mean_delta"]) <= .01 and s["ci95"][0] >= -.01 and s["ci95"][1] <= .01
             for s in summaries) and global_item["ci95"][0] >= -.01 and global_item["ci95"][1] <= .01:
        outcome = "A. 10-bin grid appears saturated on this pilot."
    else:
        outcome = "D. Pilot is too noisy to decide."
    result = {"protocol": PROTOCOL, "plan_sha256": digest(root / "config.json"),
              "prompt_count": N_PROMPTS, "seed": SEED,
              "logical_policy_rollouts": 240, "diffusion_NFE_per_rollout": 800,
              "total_diffusion_NFE": 240 * 800,
              "dense10_mean_ImageReward": statistics.mean(records["Dense10"]),
              "intervals_sorted_by_mean_delta": sorted(summaries,
                   key=lambda item: item["mean_delta"], reverse=True),
              "dense20_vs_dense10": global_item,
              "max_midpoint_mean_gain": mean_max,
              "flagged_midpoints": [s["midpoint_progress"] for s in summaries if s["refinement_flag"]],
              "outcome": outcome, "bootstrap_replicates": BOOTSTRAPS,
              "run_accounting": accounting,
              "validation_touched": False, "test_touched": False,
              "policy_search_or_tuning": False}
    write_once(root / "bootstrap_summary.json", result)
    lines = ["# Grid-density 10-bin pilot", "",
             "TRAIN diagnostic on 20 previously explored prompts, seed 42;",
             "SD1.5 BF16, DDIM100 eta=1, N=8, Max+SSP, ImageReward.",
             "Policies differ only by resampling-event positions; all use tau=8.",
             "The one-prompt event-sequence gate is in smoke_gate.json.",
             "No VALIDATION/TEST/external prompts or final-policy tuning.", "",
             "| Interval | Midpoint | Mean ΔIR | SE | 95% paired prompt CI | W/T/L | Flag |",
             "| --- | ---: | ---: | ---: | --- | --- | --- |"]
    for s in result["intervals_sorted_by_mean_delta"]:
        lo, hi = s["ci95"]
        lines.append(f"| {s['interval']} | {s['midpoint_progress']:.2f} | "
                     f"{s['mean_delta']:+.5f} | {s['se']:.5f} | "
                     f"[{lo:+.5f}, {hi:+.5f}] | "
                     f"{s['wins']}/{s['ties']}/{s['losses']} | "
                     f"{'yes' if s['refinement_flag'] else 'no'} |")
    lo, hi = global_item["ci95"]
    lines += ["", f"Dense20 − Dense10: {global_item['mean_delta']:+.5f} IR, "
              f"SE {global_item['se']:.5f}, CI [{lo:+.5f}, {hi:+.5f}], "
              f"W/T/L {global_item['wins']}/{global_item['ties']}/{global_item['losses']}.",
              f"Maximum midpoint mean gain: {mean_max:+.5f} IR.", "",
              "The midpoint flags are pilot triage rules, not confirmation.",
              "Dense20 and midpoint policies make more verifier evaluations than Dense10;"]
    lines += ["all use the same 800 diffusion NFE per prompt-policy rollout.", "",
              "Logical work: 240 full policy rollouts × 800 diffusion NFE.",
              f"Full-run wall-clock: {accounting['wall_hours']:.3f} h; four-chip TPU allocation: "
              f"{accounting['allocated_tpu_chip_hours']:.3f} chip-h; GPU-hours: 0.",
              f"Illustrative cost at $4.20/chip-h: ${accounting['illustrative_cost_usd']:.2f}; "
              "this public tariff is for us-east1, while this VM is in us-central1.",
              "Actual account rate and billing cannot be inferred from the experiment.", "",
              outcome, ""]
    report = "\n".join(lines)
    path = root / "REPORT.md"
    if path.exists() and path.read_text() != report:
        raise RuntimeError("existing report differs")
    path.write_text(report)
    return result


def plot_grid(path: Path, summaries: list[dict], global_item: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [s["midpoint_progress"] for s in summaries]
    ys = [s["mean_delta"] for s in summaries]
    lo = [s["mean_delta"] - s["ci95"][0] for s in summaries]
    hi = [s["ci95"][1] - s["mean_delta"] for s in summaries]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color="#175cd3", ecolor="#7199d7",
                elinewidth=1.5, capsize=3, label="One added midpoint")
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xlim=(0, 1), xticks=[i / 10 for i in range(11)],
           xlabel="Normalized midpoint progress (0 = noise, 1 = final image)",
           ylabel="Paired ImageReward gain over Dense10",
           title="Grid-density pilot: single-midpoint gains")
    gl, gh = global_item["ci95"]
    ax.text(.98, .98, f"Dense20 − Dense10: {global_item['mean_delta']:+.3f}\n95% CI [{gl:+.3f}, {gh:+.3f}]",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": .95})
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("plan", "smoke", "full", "status", "account", "analyze"))
    parser.add_argument("--run-root", type=Path, default=REPO / "results/static-bfs-graph" / RUN_ID)
    parser.add_argument("--dcs-root", type=Path, default=Path.home() / "static-bfs-dcs")
    parser.add_argument("--chip", type=int)
    parser.add_argument("--start-unix", type=float)
    parser.add_argument("--end-unix", type=float)
    args = parser.parse_args()
    root = args.run_root.resolve()
    if args.stage == "plan":
        out = freeze(root)
    elif args.stage in ("smoke", "full"):
        if args.chip is None:
            parser.error("smoke/full require --chip")
        out = runner(root, args.dcs_root.resolve(), args.stage, args.chip)
    elif args.stage == "status":
        out = status(root)
    elif args.stage == "account":
        if args.start_unix is None or args.end_unix is None:
            parser.error("account needs --start-unix and --end-unix")
        out = account(root, args.start_unix, args.end_unix)
    else:
        out = analyze(root)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
