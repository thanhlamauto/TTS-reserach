"""TRAIN-only N=8, 20-prompt, one-seed graph diagnostic on four TPU chips.

The four chips process disjoint prompts with the same seed 42. This fits one
additive K=3 graph; it does not evaluate a learned policy or touch TEST.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess

from .adapter import RNG_PROTOCOL, event_seed
from .graph_builder import all_edges, all_paths, dense_reference, edge_intervention, policy_from_path
from .runner import Runtime, _unit_key, digest as unit_digest, static_record, write_atomic
from .tpu_reference_compare import BASELINE, CONFIG, POOL, REPO, checked_inputs, digest, read, write_once


PROTOCOL = "STATIC_BFS_N8_GRAPH20_SEED1_TPU_V1"
RUN_ID = "n8-graph20-seed1-tpu-v1"
SEED = 42
PROMPT_COUNT = 20
CHIPS = 4
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 20261006
STAGES = ("n8_graph20_reference", "n8_graph20_edges")


def graph_inputs():
    pool, cfg, baseline = checked_inputs()
    prompts = tuple(pool["train_indices"][:PROMPT_COUNT])
    if len(prompts) != PROMPT_COUNT or prompts != tuple(sorted(prompts)):
        raise RuntimeError("20-prompt TRAIN prefix changed")
    steps = tuple(cfg["pilot"]["candidate_steps"])
    taus = tuple(map(float, cfg["pilot"]["taus"]))
    edges = all_edges(steps, taus)
    if len(edges) != 92 or cfg["pilot"]["reference_tau"] != 8.0:
        raise RuntimeError("frozen graph action space changed")
    return pool, cfg, baseline, prompts, steps, taus, edges


def manifest_inputs():
    paths = {
        "driver": Path(__file__),
        "runner": REPO / "scripts/static_bfs_graph/runner.py",
        "adapter": REPO / "scripts/static_bfs_graph/adapter.py",
        "graph_builder": REPO / "scripts/static_bfs_graph/graph_builder.py",
        "policy": REPO / "scripts/static_bfs_graph/policy.py",
        "baseline_runner": REPO / "scripts/diffusion_classical_search_t2i/tpu_runner.py",
        "config": CONFIG,
        "baseline_config": BASELINE,
        "train_pool": POOL,
    }
    return {name: digest(path) for name, path in paths.items()}


def freeze(root: Path):
    _, cfg, _, prompts, steps, taus, edges = graph_inputs()
    plan = {
        "protocol": PROTOCOL, "run_id": RUN_ID,
        "split": "TRAIN", "prompt_indices": list(prompts),
        "seed": SEED, "N": 8, "K": 3,
        "chips": CHIPS, "assignment": "prompt_indices[chip::4]",
        "candidate_steps": list(steps), "taus": list(taus),
        "edge_keys": [e.key for e in edges], "reference_tau": 8.0,
        "edge_target": "same-prompt-and-seed dense-reference terminal IR minus edge-intervention terminal IR",
        "DDIM_steps": 100, "eta": 1.0, "dtype": "bfloat16",
        "scoring": "Max", "resampling": "SSP", "verifier": "ImageReward-v1.0",
        "graph_objective": "mean additive first-order edge cost, K=3",
        "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": BOOTSTRAP_SEED,
        "previously_explored_TRAIN_prompts": True,
        "validation_touched": False, "test_touched": False,
        "external_pool_touched": False, "policy_reward_audit": False,
        "expected_reference": PROMPT_COUNT, "expected_edges": PROMPT_COUNT * len(edges),
        "exact_reference_edges_per_prompt": 8,
        "new_intervention_rollouts": PROMPT_COUNT * (len(edges) - 8),
    }
    write_once(root / "plan.json", plan)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    manifest = {
        "plan_sha256": digest(root / "plan.json"),
        "source_sha256": manifest_inputs(), "repo_commit": commit,
        "model_revision": read(BASELINE)["models"]["sd15"]["revision"],
        "torch": "2.4.0+cpu", "torch_xla": "2.4.0",
    }
    write_once(root / "manifest.json", manifest)
    return {"plan_sha256": manifest["plan_sha256"],
            "prompt_count": PROMPT_COUNT, "seed_count": 1,
            "edge_count": len(edges), "actual_intervention_rollouts": plan["new_intervention_rollouts"]}


def verify(root: Path):
    pool, cfg, baseline, prompts, steps, taus, edges = graph_inputs()
    plan, manifest = read(root / "plan.json"), read(root / "manifest.json")
    if (plan["protocol"] != PROTOCOL or plan["prompt_indices"] != list(prompts)
            or plan["seed"] != SEED or plan["edge_keys"] != [e.key for e in edges]
            or digest(root / "plan.json") != manifest["plan_sha256"]
            or manifest_inputs() != manifest["source_sha256"]):
        raise RuntimeError("frozen graph plan or scientific code changed")
    cfg = {**cfg, "particles": 8, "protocol_version": PROTOCOL}
    ref = dense_reference(steps, 8.0, 8)
    return pool, cfg, baseline, plan, prompts, edges, ref


def record_path(root: Path, cfg: dict, stage: str, ref, rec, prompt: int, edge=None):
    key = _unit_key(cfg, ref, rec, prompt, SEED, stage, edge)
    unit_hash = unit_digest(key)
    return root / "raw" / stage / "n8" / f"seed{SEED}" / f"{unit_hash}.json", key, unit_hash


def worker(root: Path, dcs_root: Path, stage: str, chip: int, max_new_units: int | None):
    if chip not in range(CHIPS) or stage not in ("reference", "edges"):
        raise ValueError("invalid stage or chip")
    pool, cfg, baseline, plan, prompts, edges, ref = verify(root)
    if not (dcs_root / "official/text_to_image/fkd_diffusers").is_dir():
        raise RuntimeError("pinned official FKD checkout is absent")
    official_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=dcs_root / "official", text=True).strip()
    if official_commit != baseline["official_source"]["commit"]:
        raise RuntimeError("official FKD source commit changed")
    if os.environ.get("PJRT_DEVICE") != "TPU":
        raise RuntimeError("PJRT_DEVICE=TPU is required")
    own_prompts = prompts[chip::CHIPS]
    prompt_records = [{"prompt": pool["prompts"].get(str(i))} for i in range(100)]
    runtime = Runtime(root, dcs_root, SEED, chip, cfg, baseline, prompt_records)
    refrec = static_record(ref)
    new = copied = 0
    if stage == "reference":
        for p in own_prompts:
            path, _, _ = record_path(root, cfg, STAGES[0], ref, refrec, p)
            if path.exists():
                continue
            runtime.run(STAGES[0], refrec, p, ref)
            new += 1
            if max_new_units and new >= max_new_units:
                break
    else:
        for edge in edges:
            policy = edge_intervention(edge, ref)
            rec = static_record(policy)
            for p in own_prompts:
                path, key, unit_hash = record_path(root, cfg, STAGES[1], ref, rec, p, edge)
                if path.exists():
                    row = read(path)
                    if row.get("unit_hash") != unit_hash or row.get("policy_id") != rec["id"]:
                        raise RuntimeError(f"invalid resumed edge: {path}")
                    continue
                if policy == ref:
                    source, _, _ = record_path(root, cfg, STAGES[0], ref, refrec, p)
                    original = read(source)
                    if original["policy_id"] != rec["id"] or original["diffusion_NFE"] != 800:
                        raise RuntimeError("exact-reference reuse failed")
                    write_atomic(path, {**original, "unit_key": key, "unit_hash": unit_hash,
                                        "edge": edge.record(), "reused_identical_reference": True})
                    copied += 1
                else:
                    runtime.run(STAGES[1], rec, p, ref, edge)
                    new += 1
                    if max_new_units and new >= max_new_units:
                        break
            if max_new_units and new >= max_new_units:
                break
    return {"stage": stage, "chip": chip, "seed": SEED,
            "prompts": list(own_prompts), "new_rollouts": new, "copied_reference_edges": copied}


def status(root: Path):
    return {"run": str(root), "expected_reference": PROMPT_COUNT,
            "expected_edges": PROMPT_COUNT * 92,
            "reference": len(list((root / "raw" / STAGES[0] / "n8").glob("seed*/*.json"))),
            "edges": len(list((root / "raw" / STAGES[1] / "n8").glob("seed*/*.json"))),
            "failures": sum(len(list((root / "failures" / stage).glob("*.json"))) for stage in STAGES)}


def analyze(root: Path):
    _, cfg, _, plan, prompts, edges, ref = verify(root)
    state = status(root)
    if (state["reference"], state["edges"], state["failures"]) != (PROMPT_COUNT, PROMPT_COUNT * 92, 0):
        raise RuntimeError(f"incomplete graph: {state}")
    refrec = static_record(ref)
    deltas = []
    ref_scores = []
    for p in prompts:
        source, _, rh = record_path(root, cfg, STAGES[0], ref, refrec, p)
        rr = read(source)
        if (rr["unit_hash"] != rh or rr["policy_id"] != ref.id or
                rr["prompt_index"] != p or rr["trial_seed"] != SEED or
                rr["diffusion_NFE"] != 800 or rr["rng_protocol"] != RNG_PROTOCOL or
                len(rr["final_particle_scores"]) != 8 or
                [ev["sampling_index"] for ev in rr["resampling_events"]] != list(ref.resampling_steps) or
                not math.isfinite(rr["selected_final_score"])):
            raise RuntimeError(f"invalid reference prompt {p}")
        ref_scores.append(rr["selected_final_score"])
        row = []
        for edge in edges:
            policy = edge_intervention(edge, ref)
            rec = static_record(policy)
            path, _, unit_hash = record_path(root, cfg, STAGES[1], ref, rec, p, edge)
            er = read(path)
            if (er["unit_hash"] != unit_hash or er["policy_id"] != rec["id"] or
                    er["prompt_index"] != p or er["trial_seed"] != SEED or
                    er["initial_generator_state_sha256"] != rr["initial_generator_state_sha256"] or
                    er["diffusion_NFE"] != 800 or er["rng_protocol"] != RNG_PROTOCOL or
                    len(er["final_particle_scores"]) != 8 or
                    [ev["sampling_index"] for ev in er["resampling_events"]] != list(policy.resampling_steps) or
                    any(ev["event_seed"] != event_seed(SEED, p, ev["sampling_index"])
                        for ev in er["resampling_events"]) or
                    not math.isfinite(er["selected_final_score"])):
                raise RuntimeError(f"invalid edge {edge.key}, prompt {p}")
            delta = rr["selected_final_score"] - er["selected_final_score"]
            if policy == ref and delta != 0:
                raise RuntimeError("identical edge has nonzero cost")
            row.append(delta)
        deltas.append(row)
    paths = list(all_paths(tuple(plan["candidate_steps"]), tuple(plan["taus"]), 3))
    paths.sort(key=lambda x: policy_from_path(x, 8).id)
    if len(paths) != 945:
        raise RuntimeError("K=3 path count changed")
    edge_index = {e.key: i for i, e in enumerate(edges)}
    path_indices = [[edge_index[e.key] for e in path] for path in paths]
    policies = [policy_from_path(path, 8) for path in paths]
    def choose(rows):
        values = [statistics.mean(row[i] for row in rows) for i in range(len(edges))]
        costs = [sum(values[i] for i in idx) for idx in path_indices]
        winner = min(range(len(paths)), key=lambda i: (costs[i], policies[i].id))
        return winner, values, costs
    selected = {}
    for n in (12, 20):
        winner, values, costs = choose(deltas[:n])
        rng = random.Random(BOOTSTRAP_SEED + n)
        picks = Counter(choose([deltas[rng.randrange(n)] for _ in range(n)])[0]
                        for _ in range(BOOTSTRAPS))
        policy = policies[winner]
        selected[str(n)] = {
            "policy_id": policy.id, "steps": list(policy.resampling_steps),
            "taus": [[i, t] for i, t in policy.temperature_by_step],
            "predicted_additive_regret": costs[winner],
            "bootstrap_exact_path_frequency": picks[winner] / BOOTSTRAPS,
            "bootstrap_unique_paths": len(picks),
            "bootstrap_top2_frequency": sum(v for _, v in picks.most_common(2)) / BOOTSTRAPS,
        }
        if n == 20:
            edge_rows = [{"edge_key": edge.key, "src": edge.src, "dst": edge.dst,
                          "tau_dst": edge.tau_dst, "mean_delta": values[i],
                          "sem_delta": statistics.stdev(row[i] for row in deltas) / math.sqrt(n),
                          "n_prompt_seed_units": n} for i, edge in enumerate(edges)]
            edge_path = root / "graph/edge_values.csv"
            edge_path.parent.mkdir(parents=True, exist_ok=True)
            with edge_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(edge_rows[0]))
                writer.writeheader()
                writer.writerows(edge_rows)
    result = {"protocol": PROTOCOL, "plan_sha256": digest(root / "plan.json"),
              "reference_mean_ImageReward": statistics.mean(ref_scores),
              "reference_prompt_count": PROMPT_COUNT, "seed": SEED,
              "selected": selected, "integrity": state,
              "validation_touched": False, "test_touched": False,
              "policy_reward_audit": False}
    write_once(root / "graph/results.json", result)
    lines = ["# N=8 graph on 20 TRAIN prompts × one seed", "",
             "SD1.5 BF16, DDIM100 eta=1, N=8, Max+SSP, ImageReward.",
             "92 first-order edges; K=3 additive search over 945 paths.",
             "This is graph fitting only. The chosen policies were not reward-audited.",
             "Prompts have appeared in earlier TRAIN work; no TEST or VALIDATION was used.", "",
             f"Dense-reference mean IR: {result['reference_mean_ImageReward']:.6f}.", "",
             "| Fit prompts | Steps | Tau | Predicted regret | Exact bootstrap |",
             "| ---: | --- | --- | ---: | ---: |"]
    for n, item in selected.items():
        lines.append(f"| {n} | {item['steps']} | {item['taus']} | "
                     f"{item['predicted_additive_regret']:.6f} | "
                     f"{item['bootstrap_exact_path_frequency']:.1%} |")
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("plan", "reference", "edges", "status", "analyze"))
    parser.add_argument("--run-root", type=Path, default=REPO / "results/static-bfs-graph" / RUN_ID)
    parser.add_argument("--dcs-root", type=Path, default=Path.home() / "static-bfs-dcs")
    parser.add_argument("--chip", type=int)
    parser.add_argument("--max-new-units", type=int)
    args = parser.parse_args()
    root = args.run_root.resolve()
    if args.stage == "plan":
        result = freeze(root)
    elif args.stage in ("reference", "edges"):
        if args.chip is None:
            parser.error("workers need --chip 0..3")
        result = worker(root, args.dcs_root.resolve(), args.stage, args.chip, args.max_new_units)
    elif args.stage == "status":
        result = status(root)
    else:
        result = analyze(root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
