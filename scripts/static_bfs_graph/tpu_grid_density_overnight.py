"""Preregistered TRAIN reference-grid campaign using the audited TPU runtime.

No graph fitting, AER change, reward-dependent selection, or stopping occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time

from .adapter import RNG_PROTOCOL, event_seed
from .policy import StaticBFSPolicy
from .tpu_reference_compare import REPO, CONFIG, BASELINE, POOL, checked_inputs, digest, read, write_once

PROTOCOL = "STATIC_BFS_GRID_DENSITY_OVERNIGHT_TPU_V1"
SEEDS = (314159, 271828, 161803, 141421)
SMOKE_NAMES = ("Dense10", "M95", "Early14", "Mid12", "Late12", "Late18", "Dense20")
BASE = tuple(range(10, 100, 10))
MIDPOINTS = tuple(range(5, 100, 10))
PRIMARY = (("C1", "Dense20", "Dense10"), ("C2", "Late12", "Dense10"),
           ("C3", "Late12", "Dense20"), ("C4", "Late18", "Late12"),
           ("C5", "Early14", "Dense10"))


def policy_specs():
    requested = {"Dense10": [str(Decimal(i) / 100) for i in BASE]}
    for mid in MIDPOINTS:
        requested[f"M{mid:02d}"] = [str(Decimal(i) / 100) for i in sorted((*BASE, mid))]
    for name, added in (("Early14", (5, 15, 25, 35, 45)),
                        ("Mid12", (45, 55, 65)), ("Late12", (75, 85, 95))):
        requested[name] = [str(Decimal(i) / 100) for i in sorted(set(BASE) | set(added))]
    requested["Late18"] = [str(Decimal(i) / 100) for i in range(10, 70, 10)] + [
        "0.70", "0.725", "0.75", "0.775", "0.80", "0.825", "0.85", "0.875",
        "0.90", "0.925", "0.95", "0.975"]
    requested["Dense20"] = [str(Decimal(i) / 100) for i in range(5, 100, 5)]
    out = {}
    for name, progress in requested.items():
        indices = [int((Decimal(p) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
                   for p in progress]
        p = StaticBFSPolicy(tuple(indices), tuple((i, 8.0) for i in indices), 8)
        out[name] = {"policy_name": name, "requested_progress": list(map(float, progress)),
                     "actual_step_indices": indices, "actual_progress": [i / 100 for i in indices],
                     "num_search_events": len(indices),
                     "verifier_evaluations": (len(indices) + 1) * 8,
                     "policy_record": {"kind": "static", "id": p.id, **p.record()}}
    assert len(out) == 16 and len({v["policy_record"]["id"] for v in out.values()}) == 16
    assert out["Late18"]["actual_step_indices"] == [10,20,30,40,50,60,70,73,75,78,80,83,85,88,90,93,95,98]
    assert [out[n]["num_search_events"] for n in SMOKE_NAMES] == [9,10,14,12,12,18,19]
    for mid in MIDPOINTS:
        assert set(out[f"M{mid:02d}"]["actual_step_indices"]) - set(BASE) == {mid}
    for spec in out.values():
        assert spec["actual_step_indices"] == sorted(set(spec["actual_step_indices"]))
        assert all(0 < x < 100 for x in spec["actual_step_indices"])
        p = spec["policy_record"]
        assert p["particles"] == 8 and p["selection_mode"] == "raw_tau"
        assert all(t == 8.0 for _, t in p["temperature_by_step"])
    return out


def scientific_hashes():
    paths = {"campaign_driver": Path(__file__), "runner": REPO / "scripts/static_bfs_graph/runner.py",
             "orchestration": Path(__file__).with_name("run_grid_density_overnight_tpu.sh"),
             "input_checker": Path(__file__).with_name("tpu_reference_compare.py"),
             "adapter": REPO / "scripts/static_bfs_graph/adapter.py",
             "policy": REPO / "scripts/static_bfs_graph/policy.py",
             "baseline_runner": REPO / "scripts/diffusion_classical_search_t2i/tpu_runner.py",
             "config": CONFIG, "baseline_config": BASELINE, "train_pool": POOL}
    return {k: digest(p) for k, p in paths.items()}


def pilot_split(pilot: Path):
    pool, _, _ = checked_inputs()
    cfg, manifest = read(pilot / "config.json"), read(pilot / "manifest.json")
    if (digest(pilot / "config.json") != manifest["plan_sha256"] or cfg["seed"] != 42 or
        cfg["protocol"] != "STATIC_BFS_GRID_DENSITY_10BIN_TPU_V1"):
        raise RuntimeError("ambiguous or changed pilot config")
    old = cfg["prompt_indices"]
    rows = [json.loads(x) for x in (pilot / "dense10_records.jsonl").read_text().splitlines()]
    if (len(old) != 20 or len(set(old)) != 20 or len(rows) != 20 or
        sorted(r["prompt_id"] for r in rows) != sorted(old) or
        any(r["seed"] != 42 for r in rows)):
        raise RuntimeError("pilot prompt IDs not recovered unambiguously")
    train = pool["train_indices"]
    fresh = sorted(set(train) - set(old))
    if len(train) != 60 or len(fresh) != 40 or not set(old).issubset(train):
        raise RuntimeError("not a 40/20 partition of the exact 60 TRAIN prompts")
    return {"CONFIRMATORY": fresh, "REPLICATION": sorted(old),
            "pilot_seed": 42, "pilot_config_sha256": digest(pilot / "config.json"),
            "pilot_dense10_records_sha256": digest(pilot / "dense10_records.jsonl"),
            "train_pool_sha256": digest(POOL),
            "fresh_means": "not in previous midpoint pilot; previously explored TRAIN pool",
            "validation_touched": False, "test_touched": False, "external_pool_touched": False}


def freeze(root: Path, pilot: Path):
    _, _, baseline = checked_inputs()
    specs, split = policy_specs(), pilot_split(pilot)
    write_once(root / "preregistered_policies.json", specs)
    write_once(root / "prompt_split.json", split)
    plan = {"protocol": PROTOCOL, "seeds": list(SEEDS), "pilot_root": str(pilot),
            "policies_sha256": digest(root / "preregistered_policies.json"),
            "split_sha256": digest(root / "prompt_split.json"),
            "model": "SD1.5", "N": 8, "dtype": "bfloat16", "sampler": "DDIM100",
            "eta": 1.0, "scoring": "Max", "resampling": "SSP", "tau": 8.0,
            "verifier": "ImageReward-v1.0", "mapping": "Decimal ROUND_HALF_UP(progress * 100)",
            "event_semantics": "audited zero-based denoising-loop sampling_index; not scheduler timestep",
            "expected_logical_runs": 3840, "primary_runs": 2560, "replication_runs": 1280,
            "diffusion_NFE_per_run": 800, "total_diffusion_NFE": 3072000,
            "total_verifier_evaluations": sum(s["verifier_evaluations"] for s in specs.values()) * 240,
            "physical_chips": 4, "chip_assignment": "one preregistered seed per chip",
            "execution_order": "CONFIRMATORY IDs then REPLICATION IDs; fixed policy insertion order",
            "bootstrap_replicates": 50000, "sign_flips": 100000, "statistics_seed": 20261007,
            "primary_contrasts": [list(x) for x in PRIMARY], "primary_holm_family": 5,
            "midpoint_holm_family": 10, "permutation_alternative": "two-sided",
            "cluster_unit": "prompt after averaging four paired seed differences",
            "meaningful_gain": .01, "noninferiority_margin": .02,
            "noninferiority_CI": "5th percentile of prompt bootstrap (one-sided 95% lower)",
            "gain_capture_guard": "omit ratio if Dense20-Dense10 95% bootstrap CI includes zero",
            "decision_F_precision_guard": "F takes priority over A if Q1=NO and C1 CI includes both zero and +.01; otherwise literal A-E rules, unmatched patterns F",
            "Q4_interpretation": "literal protocol checklist; absence of significance is not proof of saturation",
            "power_method": "normal paired approximation, two-sided alpha .05, power .8; uncorrected planning",
            "reward_dependent_stopping": False, "policy_tuning": False,
            "validation_touched": False, "test_touched": False, "external_pool_touched": False}
    assert 42 not in SEEDS
    write_once(root / "campaign_config.json", plan)
    write_once(root / "manifest.json", {"config_sha256": digest(root / "campaign_config.json"),
        "scientific_source_sha256": scientific_hashes(),
        "analysis_sha256": digest(Path(__file__).with_name("grid_density_overnight_analysis.py")),
        "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "official_source_commit": baseline["official_source"]["commit"],
        "model_revision": baseline["models"]["sd15"]["revision"],
        "frozen_utc": datetime.now(timezone.utc).isoformat()})
    return {"config_sha256": digest(root / "campaign_config.json"), "runs": 3840,
            "prompt_split": {k: len(split[k]) for k in ("CONFIRMATORY", "REPLICATION")},
            "event_indices": {k: v["actual_step_indices"] for k,v in specs.items()}}


def verify(root: Path):
    pool, cfg, old = checked_inputs()
    plan, manifest = read(root / "campaign_config.json"), read(root / "manifest.json")
    specs, split = read(root / "preregistered_policies.json"), read(root / "prompt_split.json")
    if (plan["protocol"] != PROTOCOL or plan["seeds"] != list(SEEDS) or
        specs != policy_specs() or split != pilot_split(Path(plan["pilot_root"])) or
        digest(root / "campaign_config.json") != manifest["config_sha256"] or
        digest(root / "preregistered_policies.json") != plan["policies_sha256"] or
        digest(root / "prompt_split.json") != plan["split_sha256"] or
        scientific_hashes() != manifest["scientific_source_sha256"]):
        raise RuntimeError("frozen protocol, split, or scientific source changed")
    return pool, {**cfg, "particles": 8, "protocol_version": PROTOCOL}, old, plan, specs, split


def record_path(root: Path, seed: int, prompt: int, name: str):
    return root / "records" / f"seed{seed}" / f"p{prompt:03d}-{name}.json"


def validate_row(row, spec, seed, prompt):
    if (row["policy_id"] != spec["policy_record"]["id"] or row["trial_seed"] != seed or
        row["prompt_index"] != prompt or row["diffusion_NFE"] != 800 or
        row["execution_dtype"] != "torch.bfloat16" or row["rng_protocol"] != RNG_PROTOCOL or
        len(row["final_particle_scores"]) != 8 or
        row["verifier_calls"] != spec["verifier_evaluations"] or
        [e["sampling_index"] for e in row["resampling_events"]] != spec["actual_step_indices"] or
        not math.isfinite(row["selected_final_score"])):
        raise RuntimeError(f"invalid result {prompt}/{seed}/{spec['policy_name']}")
    for e in row["resampling_events"]:
        assert e["temperature"] == 8.0 and e["event_seed"] == event_seed(seed,prompt,e["sampling_index"])
        assert len(e["parent_indices"]) == 8 and all(0 <= i < 8 for i in e["parent_indices"])


def append_attempt(root: Path, row):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "attempts.jsonl").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(row, sort_keys=True) + "\n"); f.flush(); os.fsync(f.fileno())


def worker(root: Path, dcs: Path, chip: int, smoke: bool):
    pool, cfg, old, plan, specs, split = verify(root)
    if chip not in range(4) or (smoke and chip != 0): raise ValueError("invalid chip")
    if os.environ.get("PJRT_DEVICE") != "TPU": raise RuntimeError("PJRT_DEVICE=TPU required")
    official = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dcs/"official", text=True).strip()
    if official != old["official_source"]["commit"]: raise RuntimeError("official FKD source changed")
    if not smoke:
        gate = read(root / "smoke_gate.json")
        if not gate["passed"] or gate["config_sha256"] != digest(root/"campaign_config.json"):
            raise RuntimeError("smoke gate absent or changed")
    from .runner import Runtime
    seed = SEEDS[chip]
    prompt_records = [{"prompt": pool["prompts"].get(str(i))} for i in range(100)]
    runtime = Runtime(root, dcs, seed, chip, cfg, old, prompt_records)
    ref = StaticBFSPolicy(BASE, tuple((i,8.0) for i in BASE), 8)
    prompts = [split["REPLICATION"][0]] if smoke else split["CONFIRMATORY"]+split["REPLICATION"]
    names = SMOKE_NAMES if smoke else tuple(specs)
    for prompt in prompts:
        hashes = set()
        for name in names:
            spec = specs[name]; dest = record_path(root,seed,prompt,name)
            started = time.time()
            try:
                row = runtime.run("grid_overnight",spec["policy_record"],prompt,ref)
                validate_row(row,spec,seed,prompt)
                hashes.add(row["initial_generator_state_sha256"])
                compact = {"split": "CONFIRMATORY" if prompt in split["CONFIRMATORY"] else "REPLICATION",
                    "prompt_id": prompt, "seed": seed, "policy_name": name,
                    **{k: spec[k] for k in ("requested_progress","actual_step_indices","actual_progress","num_search_events")},
                    "ImageReward": row["selected_final_score"], "wallclock_seconds": row["elapsed_seconds"],
                    "diffusion_NFE": row["diffusion_NFE"], "verifier_evaluations": row["verifier_calls"],
                    "resampling_events": row["resampling_events"], "status": "success", "error_message": "",
                    "initial_generator_state_sha256": row["initial_generator_state_sha256"],
                    "prompt_seed": row["prompt_seed"], "raw_unit_hash": row["unit_hash"]}
                if not dest.exists():
                    write_once(dest,compact)
                    append_attempt(root,{"status":"success","prompt_id":prompt,"seed":seed,"policy_name":name,
                        "started_unix":started,"finished_unix":time.time(),"record_wallclock_seconds":row["elapsed_seconds"]})
                elif read(dest) != compact: raise RuntimeError("normalized raw record changed")
            except BaseException as exc:
                append_attempt(root,{"status":"failed","prompt_id":prompt,"seed":seed,"policy_name":name,
                    "started_unix":started,"finished_unix":time.time(),"error_message":repr(exc)})
                raise
        if len(hashes) != 1: raise RuntimeError("particle-noise pairing failed")
    if smoke:
        rows = {name:read(record_path(root,seed,prompts[0],name)) for name in names}
        seq = {n:r["actual_step_indices"] for n,r in rows.items()}
        write_once(root/"smoke_gate.json",{"passed":True,"config_sha256":digest(root/"campaign_config.json"),
            "prompt_id":prompts[0],"seed":seed,"event_sequences":seq,
            "initial_particle_noise_paired":len({r["initial_generator_state_sha256"] for r in rows.values()})==1,
            "non_temporal_settings_identical":True,"tau":8.0,"mapping":plan["mapping"],
            "requested_and_actual_progress":{n:{k:specs[n][k] for k in ("requested_progress","actual_progress")} for n in names}})
        print("COMPLETE_EVENT_SEQUENCES",json.dumps(seq),flush=True)
    return {"stage":"smoke" if smoke else "full","chip":chip,"seed":seed,"checked":len(prompts)*len(names)}


def status(root: Path):
    paths=list((root/"records").glob("seed*/*.json"))
    counts={"CONFIRMATORY":0,"REPLICATION":0}; seed_counts={str(s):0 for s in SEEDS}
    for path in paths:
        r=read(path); counts[r["split"]]+=1; seed_counts[str(r["seed"])]+=1
    attempts=[json.loads(x) for x in (root/"attempts.jsonl").read_text().splitlines()] if (root/"attempts.jsonl").exists() else []
    return {"successful_runs":len(paths),"expected":3840,"splits":counts,"seeds":seed_counts,
            "failed_attempts":sum(r["status"]=="failed" for r in attempts)}


def export(root: Path):
    _,_,_,plan,specs,split=verify(root)
    rows=[]
    for seed in SEEDS:
        for part in ("CONFIRMATORY","REPLICATION"):
            for p in split[part]:
                rr=[read(record_path(root,seed,p,n)) for n in specs]
                assert len({r["initial_generator_state_sha256"] for r in rr})==1
                assert len({r["prompt_seed"] for r in rr})==1
                for r in rr:
                    assert r["status"]=="success" and r["split"]==part and r["seed"]==seed and r["prompt_id"]==p
                    assert r["actual_step_indices"]==specs[r["policy_name"]]["actual_step_indices"]
                    assert r["diffusion_NFE"]==800 and r["verifier_evaluations"]==specs[r["policy_name"]]["verifier_evaluations"]
                    assert [e["sampling_index"] for e in r["resampling_events"]]==r["actual_step_indices"]
                    assert all(e["temperature"]==8 and e["event_seed"]==event_seed(seed,p,e["sampling_index"]) for e in r["resampling_events"])
                    assert math.isfinite(r["ImageReward"])
                rows.extend(rr)
    assert len(rows)==3840
    content="".join(json.dumps(r,sort_keys=True)+"\n" for r in rows)
    path=root/"raw_records.jsonl"
    if path.exists() and path.read_text()!=content: raise RuntimeError("raw export changed")
    path.write_text(content)
    return {"exported":len(rows),"raw_records_sha256":digest(path),"source_hashes_verified":True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("plan","smoke","full","status","export","analyze"))
    parser.add_argument("--run-root",type=Path,required=True)
    parser.add_argument("--pilot-root",type=Path)
    parser.add_argument("--dcs-root",type=Path,default=Path.home()/"static-bfs-dcs")
    parser.add_argument("--chip",type=int)
    args=parser.parse_args(); root=args.run_root.resolve()
    if args.stage=="plan":
        if args.pilot_root is None: parser.error("plan requires exact --pilot-root")
        result=freeze(root,args.pilot_root.resolve())
    elif args.stage in ("smoke","full"):
        result=worker(root,args.dcs_root.resolve(),args.chip,args.stage=="smoke")
    elif args.stage=="status": result=status(root)
    elif args.stage=="export": result=export(root)
    else:
        verify(root)
        from .grid_density_overnight_analysis import analyze
        result=analyze(root)
    print(json.dumps(result,indent=2,sort_keys=True),flush=True)


if __name__=="__main__": main()
