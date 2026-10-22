"""Small paired exploratory trial: resample after update 20 vs updates 1–20."""
from __future__ import annotations

import csv
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

from .policy import StaticBFSPolicy
from .adapter import event_seed
from .tpu_reference_compare import checked_inputs,REPO,digest,read,write_once

PROTOCOL="STATIC_BFS_EARLY20_QUICK_TPU_V1"
SEED=42
PROMPTS=(0,2,3,4,5)
SCHEDULES={"ONLY20":(19,),"EVERY1_TO20":tuple(range(20))}


class CapturePipeline:
    """Delegate unchanged inference and retain decoded endpoints for review."""
    def __init__(self,pipe):self.pipe=pipe;self.images=None
    def __getattr__(self,name):return getattr(self.pipe,name)
    def __call__(self,*args,**kwargs):
        result=self.pipe(*args,**kwargs);self.images=list(result.images);return result


def run(root:Path,dcs:Path):
    pool,cfg,baseline=checked_inputs()
    assert all(p in pool["train_indices"] for p in PROMPTS)
    assert os.environ.get("PJRT_DEVICE")=="TPU" and os.environ.get("TPU_VISIBLE_CHIPS")=="0"
    policies={n:StaticBFSPolicy(steps,tuple((i,8.) for i in steps),8) for n,steps in SCHEDULES.items()}
    plan={"protocol":PROTOCOL,"prompt_ids":list(PROMPTS),"prompts":{str(p):pool["prompts"][str(p)] for p in PROMPTS},
        "seed":SEED,"N":8,"sampler_steps":100,"eta":1.,"precision":"BF16","scoring":"Max",
        "resampling":"SSP","tau":8.,"verifier":"ImageReward-v1.0","split":"previously explored TRAIN",
        "step_definition":"one-based completed denoising update: update k maps to sampling_index k-1",
        "policies":{n:{"kind":"static","id":p.id,**p.record()} for n,p in policies.items()},
        "logical_runs":10,"diffusion_NFE_per_run":800,"expected_verifier_evaluations":920,
        "purpose":"exploratory image/reward comparison, no confirmatory inference","validation_or_test_used":False}
    write_once(root/"config.json",plan)
    paths=[Path(__file__),REPO/"scripts/static_bfs_graph/runner.py",REPO/"scripts/static_bfs_graph/adapter.py",
        REPO/"scripts/diffusion_classical_search_t2i/tpu_runner.py"]
    write_once(root/"manifest.json",{"config_sha256":digest(root/"config.json"),
        "source_sha256":{str(p.relative_to(REPO)):digest(p) for p in paths},
        "repo_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
        "model_revision":baseline["models"]["sd15"]["revision"]})
    cfg={**cfg,"particles":8,"protocol_version":PROTOCOL}
    from .runner import Runtime
    from .runner import static_record
    started=time.time()
    write_once(root/"started.json",{"utc":datetime.fromtimestamp(started,timezone.utc).isoformat()})
    runtime=Runtime(root,dcs,SEED,0,cfg,baseline,[{"prompt":pool["prompts"].get(str(i))} for i in range(100)])
    runtime.pipe=CapturePipeline(runtime.pipe)
    reference=policies["ONLY20"]
    (root/"images").mkdir(exist_ok=True)
    records=[]
    for prompt in PROMPTS:
        paired=[]
        for name,p in policies.items():
            runtime.pipe.images=None
            row=runtime.run("early20_quick",static_record(p),prompt,reference)
            assert row["diffusion_NFE"]==800 and row["verifier_calls"]==8*(len(p.resampling_steps)+1)
            assert [e["sampling_index"] for e in row["resampling_events"]]==list(p.resampling_steps)
            assert all(e["temperature"]==8. and e["event_seed"]==event_seed(SEED,prompt,e["sampling_index"]) for e in row["resampling_events"])
            image=root/"images"/f"p{prompt:03d}-{name}.png"
            if runtime.pipe.images is not None:
                runtime.pipe.images[row["selected_particle_index"]].save(image)
            if not image.exists():raise RuntimeError("resumed rollout has no saved image; preserve raw and rerun in new namespace")
            compact={"prompt_id":prompt,"prompt":pool["prompts"][str(prompt)],"seed":SEED,"method":name,
                "ImageReward":row["selected_final_score"],"diffusion_NFE":row["diffusion_NFE"],
                "verifier_evaluations":row["verifier_calls"],"runtime_seconds":row["elapsed_seconds"],
                "event_update_numbers":[e["sampling_index"]+1 for e in row["resampling_events"]],
                "resampling_events":row["resampling_events"],"initial_noise_hash":row["initial_generator_state_sha256"],
                "selected_rgb_sha256":row["selected_image_rgb_sha256"],"final_particle_scores":row["final_particle_scores"],
                "image":str(image.relative_to(root)),"status":"success"}
            write_once(root/"records"/f"p{prompt:03d}-{name}.json",compact)
            records.append(compact);paired.append(compact)
        assert paired[0]["initial_noise_hash"]==paired[1]["initial_noise_hash"]
    (root/"raw_records.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in records))
    table=[]
    for prompt in PROMPTS:
        a,b=[r for r in records if r["prompt_id"]==prompt]
        table.append({"prompt_id":prompt,"prompt":a["prompt"],"IR_ONLY20":a["ImageReward"],
            "IR_EVERY1_TO20":b["ImageReward"],"delta_every_minus_only":b["ImageReward"]-a["ImageReward"],
            "selected_image_bitwise_equal":a["selected_rgb_sha256"]==b["selected_rgb_sha256"],
            "all_particle_scores_equal":a["final_particle_scores"]==b["final_particle_scores"]})
    with (root/"comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    events=[e for r in records if r["method"]=="EVERY1_TO20" for e in r["resampling_events"]]
    uniform=sum(max(e["normalized_weights"])-min(e["normalized_weights"])<1e-12 for e in events)
    identity=sum(e["parent_indices"]==list(range(8)) for e in events)
    result={"successful_runs":10,"NFE":sum(r["diffusion_NFE"] for r in records),
        "verifier_evaluations":sum(r["verifier_evaluations"] for r in records),
        "mean_IR_ONLY20":statistics.mean(r["IR_ONLY20"] for r in table),
        "mean_IR_EVERY1_TO20":statistics.mean(r["IR_EVERY1_TO20"] for r in table),
        "mean_paired_delta":statistics.mean(r["delta_every_minus_only"] for r in table),
        "wins_ties_losses":[sum(r["delta_every_minus_only"]>0 for r in table),sum(r["delta_every_minus_only"]==0 for r in table),sum(r["delta_every_minus_only"]<0 for r in table)],
        "identical_selected_images":sum(r["selected_image_bitwise_equal"] for r in table),
        "every20_uniform_weight_events":uniform,"every20_identity_parent_events":identity,"every20_total_events":len(events),
        "mean_runtime_by_method":{n:statistics.mean(r["runtime_seconds"] for r in records if r["method"]==n) for n in policies},
        "wall_seconds_including_load_and_compile":time.time()-started,
        "note":"five previously explored TRAIN prompts, one seed; descriptive only"}
    write_once(root/"results.json",result)
    lines=["# Quick early-20 resampling trial","",
        "SD1.5 BF16, DDIM100 eta1, N8, Max+SSP, tau8, ImageReward; five TRAIN prompts, seed42.",
        "One-based update20 maps to sampling_index19. Updates1–20 map to indices0–19.",
        "Both methods denoise normally after update20. Same initial noise and 800 diffusion NFE per rollout.",
        "Verifier particle evaluations per rollout, including terminal scoring: ONLY20=16; EVERY1_TO20=168.","",
        "| Prompt | ONLY20 IR | EVERY1_TO20 IR | Delta | Selected image bitwise equal |",
        "| --- | ---: | ---: | ---: | --- |"]
    for r in table:lines.append(f"| {r['prompt_id']}: {r['prompt']} | {r['IR_ONLY20']:.6f} | {r['IR_EVERY1_TO20']:.6f} | {r['delta_every_minus_only']:+.6f} | {r['selected_image_bitwise_equal']} |")
    lines += ["",f"Mean paired delta: {result['mean_paired_delta']:+.6f}; W/T/L {result['wins_ties_losses']}.",
        f"Identical selected endpoint RGB images: {result['identical_selected_images']}/5.",
        f"EVERY1_TO20: uniform weights at {uniform}/{len(events)} events, identity parent allocation at {identity}/{len(events)} events.",
        "Audited Max scoring initializes its previous-score buffer at zero. Negative early rewards are floored at zero by Max; equal scores produce uniform weights.",
        f"Total wall-clock including load/compile: {result['wall_seconds_including_load_and_compile']:.1f} s.",
        "Exploratory sample only; not a statistical confirmation. Selected endpoint images are saved under images/.",""]
    (root/"REPORT.md").write_text("\n".join(lines))
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--run-root",type=Path,required=True);p.add_argument("--dcs-root",type=Path,required=True)
    a=p.parse_args();run(a.run_root.resolve(),a.dcs_root.resolve())
