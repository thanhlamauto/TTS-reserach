"""Prompt-cluster inference for the preregistered grid campaign; no policy search."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")


def write_csv(path, rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def holm(pvalues):
    ordered=sorted(pvalues,key=pvalues.get);m=len(ordered);out={};maximum=0.0
    for i,k in enumerate(ordered):
        maximum=max(maximum,(m-i)*pvalues[k]);out[k]=min(1.0,maximum)
    return out


def paired_stats(differences, draws, signs):
    d=np.asarray(differences,dtype=float);n=len(d)
    boots=d[draws].mean(axis=1)
    null=signs @ d/n
    observed=float(d.mean())
    p=float((1+np.count_nonzero(np.abs(null)>=abs(observed)-1e-14))/(len(null)+1))
    return {"n_prompts":n,"mean_delta":observed,"median_delta":float(np.median(d)),
        "prompt_SD":float(d.std(ddof=1)),"SE":float(d.std(ddof=1)/math.sqrt(n)),
        "ci95_low":float(np.quantile(boots,.025)),"ci95_high":float(np.quantile(boots,.975)),
        "one_sided95_lower":float(np.quantile(boots,.05)),
        "raw_p":p,"wins":int(np.count_nonzero(d>0)),"ties":int(np.count_nonzero(d==0)),
        "losses":int(np.count_nonzero(d<0))},boots


def decision(primary, midpoint):
    c={r["contrast"]:r for r in primary}
    positive=lambda r:r["mean_delta"]>=.01 and r["holm_p"]<.05
    q={"Q1":positive(c["C1"]),"Q2":positive(c["C2"]),
       "Q3":c["C3"]["one_sided95_lower"]>-.02,
       "Q4":c["C4"]["mean_delta"]<.01 and not(c["C4"]["mean_delta"]>0 and c["C4"]["holm_p"]<.05),
       "Q5":positive(c["C5"]),"Q6":any(positive(r) for r in midpoint)}
    if not q["Q1"]:
        uncertain=c["C1"]["ci95_low"]<=0 and c["C1"]["ci95_high"]>=.01
        case,next_ref=("F","unresolved") if uncertain else ("A","Dense10")
    elif q["Q2"] and q["Q3"] and q["Q4"]: case,next_ref="B","Late12 (next graph-reference candidate)"
    elif q["Q2"] and q["Q3"] and not q["Q4"]: case,next_ref="C","unresolved; Late18 diagnostic candidate"
    elif not q["Q3"] and q["Q5"]: case,next_ref="D","unresolved; multi-region reference needed"
    elif not q["Q3"] and not q["Q5"]: case,next_ref="E","Dense20 (next diagnostic candidate)"
    else: case,next_ref="F","unresolved"
    return {"case":case,"recommended_next_reference":next_ref,"checklist":q,
        "Q4_caveat":"Literal checklist rule; absence of corrected evidence does not establish saturation.",
        "noninferiority":"One-sided 95% bootstrap lower > -.02; not equivalence",
        "production_changed":False}


def analyze(root: Path):
    from .tpu_grid_density_overnight import SEEDS,PRIMARY,MIDPOINTS,verify,digest
    _,_,_,plan,specs,split=verify(root)
    manifest=json.loads((root/"manifest.json").read_text())
    if digest(Path(__file__))!=manifest["analysis_sha256"]:
        raise RuntimeError("analysis implementation changed; audit amendment required")
    rows=[json.loads(s) for s in (root/"raw_records.jsonl").read_text().splitlines()]
    names=list(specs);name_index={n:i for i,n in enumerate(names)}
    ids=sorted(split["CONFIRMATORY"]+split["REPLICATION"])
    by_key={(r["prompt_id"],r["seed"],r["policy_name"]):r for r in rows}
    assert len(rows)==len(by_key)==3840
    for p in ids:
        for s in SEEDS:
            paired=[by_key[p,s,n] for n in names]
            assert len({r["initial_generator_state_sha256"] for r in paired})==1
            assert all(r["status"]=="success" for r in paired)
    # Raw ledger exists and is fully verified before any aggregation.
    seed_rows=[{k:r[k] for k in ("split","prompt_id","seed","policy_name","ImageReward",
        "num_search_events","diffusion_NFE","verifier_evaluations","wallclock_seconds","status","error_message")}
        for r in rows]
    write_csv(root/"per_prompt_seed.csv",seed_rows)
    averaged=[];values={}
    for p in ids:
        part="CONFIRMATORY" if p in split["CONFIRMATORY"] else "REPLICATION"
        for n in names:
            rr=[by_key[p,s,n] for s in SEEDS];reward=float(np.mean([r["ImageReward"] for r in rr]))
            values[p,n]=reward
            averaged.append({"split":part,"prompt_id":p,"policy_name":n,"mean_ImageReward":reward,
                "mean_delta_vs_Dense10":reward-float(np.mean([by_key[p,s,"Dense10"]["ImageReward"] for s in SEEDS])),
                "seeds_averaged":4,"mean_wallclock_seconds":float(np.mean([r["wallclock_seconds"] for r in rr]))})
    write_csv(root/"per_prompt_averaged.csv",averaged)
    all_primary=[];all_midpoint=[];boot_result={};holm_result={};summaries=[];power=[]
    extra_effects={};cache={}
    z=NormalDist().inv_cdf(.975)+NormalDist().inv_cdf(.8)
    for split_index,(part,pids) in enumerate((("CONFIRMATORY",split["CONFIRMATORY"]),
                                             ("REPLICATION",split["REPLICATION"]),("POOLED",ids))):
        n=len(pids);rng=np.random.default_rng(plan["statistics_seed"]+split_index)
        draws=rng.integers(0,n,size=(plan["bootstrap_replicates"],n))
        signs=(rng.integers(0,2,size=(plan["sign_flips"],n),dtype=np.int8)*2-1).astype(float)
        matrix=np.array([[values[p,name] for name in names] for p in pids])
        def estimate(a,b):
            key=(a,b)
            if key not in cache.setdefault(part,{}):
                cache[part][key]=paired_stats(matrix[:,name_index[a]]-matrix[:,name_index[b]],draws,signs)
            return cache[part][key]
        primary=[]
        for label,a,b in PRIMARY:
            st,_=estimate(a,b);primary.append({"split":part,"contrast":label,"policy_a":a,"policy_b":b,**st})
        adjusted=holm({r["contrast"]:r["raw_p"] for r in primary})
        for r in primary:
            r["holm_p"]=adjusted[r["contrast"]]
            if part=="CONFIRMATORY":
                sd=r["prompt_SD"]
                for target in (.01,.02,abs(r["mean_delta"])):
                    required=max(2,math.ceil((z*sd/target)**2)) if target>0 else None
                    power.append({"contrast":r["contrast"],"observed_prompt_SD":sd,
                        "target_abs_IR_effect":target,"independent_prompts_for_80pct_power":required,
                        "alpha_two_sided":.05,"power":.8,"method":"normal paired approximation; no Holm inflation"})
        mids=[]
        for mid in MIDPOINTS:
            name=f"M{mid:02d}";st,_=estimate(name,"Dense10")
            mids.append({"split":part,"policy_name":name,"requested_midpoint":mid/100,**st})
        ma=holm({r["policy_name"]:r["raw_p"] for r in mids})
        for r in mids:
            r["holm_p"]=ma[r["policy_name"]]
            r["confirmed_sensitive"]=r["mean_delta"]>=.01 and r["holm_p"]<.05
        all_primary.extend(primary);all_midpoint.extend(mids)
        holm_result[part]={"primary_family5":adjusted,"midpoint_family10":ma}
        denominator,den_boot=estimate("Dense20","Dense10")
        numerator,num_boot=estimate("Late12","Dense10")
        capture={"extra_event_ratio":.3,"reported":False,"reason":"global-gain denominator unstable around zero"}
        if denominator["mean_delta"]>0 and denominator["ci95_low"]>0:
            ratios=num_boot/den_boot
            capture={"extra_event_ratio":.3,"reported":True,
                "capture_late12":numerator["mean_delta"]/denominator["mean_delta"],
                "ci95":np.quantile(ratios,[.025,.975]).tolist()}
        boot_result[part]={"primary_contrasts":primary,"midpoint_contrasts":mids,
            "capture_late12":capture,"bootstrap_replicates":plan["bootstrap_replicates"],
            "sign_flip_replicates":plan["sign_flips"],"statistical_unit":"prompt, averaged over four seeds"}
        extra_effects[part]={}
        dense_time=float(np.mean([by_key[p,s,"Dense10"]["wallclock_seconds"] for p in pids for s in SEEDS]))
        for name in names:
            rr=[by_key[p,s,name] for p in pids for s in SEEDS]
            st,_=estimate(name,"Dense10");extra=specs[name]["num_search_events"]-9
            wall=float(np.mean([r["wallclock_seconds"] for r in rr]))
            summaries.append({"split":part,"policy_name":name,"n_prompts":n,
                "mean_ImageReward":float(matrix[:,name_index[name]].mean()),
                "num_search_events":specs[name]["num_search_events"],
                "verifier_evaluations_per_run":specs[name]["verifier_evaluations"],
                "mean_wallclock_seconds":wall,"relative_wallclock_vs_Dense10":wall/dense_time,
                "gain_over_Dense10":st["mean_delta"],"extra_search_events":extra,
                "gain_per_extra_search_event":st["mean_delta"]/extra if extra else None})
            extra_effects[part][name]=st
    write_csv(root/"primary_contrasts.csv",all_primary)
    write_csv(root/"midpoint_contrasts.csv",all_midpoint)
    write_csv(root/"policy_summary.csv",summaries)
    write_csv(root/"power_analysis.csv",power)
    write_json(root/"holm_results.json",holm_result)
    write_json(root/"bootstrap_results.json",boot_result)
    main_primary=boot_result["CONFIRMATORY"]["primary_contrasts"]
    main_mids=boot_result["CONFIRMATORY"]["midpoint_contrasts"]
    result=decision(main_primary,main_mids)
    reversals=[]
    for name in names[1:]:
        fresh=extra_effects["CONFIRMATORY"][name]["mean_delta"]
        old=extra_effects["REPLICATION"][name]["mean_delta"]
        if fresh*old<0: reversals.append(name)
    # Also flag reversal for non-baseline primary contrasts C3 and C4.
    for fresh,old in zip(main_primary,boot_result["REPLICATION"]["primary_contrasts"]):
        if fresh["mean_delta"]*old["mean_delta"]<0: reversals.append(fresh["contrast"])
    result["effect_sign_reversals"]=reversals
    write_json(root/"decision.json",result)
    compute=account(root,rows,plan,specs)
    figures(root,split,values,summaries,boot_result,extra_effects)
    report(root,result,boot_result,extra_effects,power,compute,plan)
    return {"successful_runs":3840,"decision":result,"accounting":compute}


def account(root,rows,plan,specs):
    timing=json.loads((root/"timing.json").read_text())
    attempts=[json.loads(s) for s in (root/"attempts.jsonl").read_text().splitlines()]
    failed=[r for r in attempts if r["status"]=="failed"]
    failed_keys={(r["prompt_id"],r["seed"],r["policy_name"]) for r in failed}
    full_wall=timing["full_finished_unix"]-timing["full_started_unix"]
    smoke_wall=timing["smoke_finished_unix"]-timing["smoke_started_unix"]
    total_wall=timing["full_finished_unix"]-timing["smoke_started_unix"]
    busy_seconds=sum(r["wallclock_seconds"] for r in rows)
    busy_seconds+=sum(r["finished_unix"]-r["started_unix"] for r in failed)
    allocated=(4*full_wall+smoke_wall)/3600
    pilot=json.loads((Path(plan["pilot_root"])/"run_accounting.json").read_text())
    successful_verifier=sum(r["verifier_evaluations"] for r in rows)
    assert successful_verifier==plan["total_verifier_evaluations"]
    compute={"logical_policy_runs":3840,"successful_runs":len(rows),"failed_attempts":len(failed),
        "rerun_units":len(failed_keys),"total_executed_attempts":len(rows)+len(failed),
        "wall_seconds_including_smoke":total_wall,"full_stage_wall_seconds":full_wall,
        "smoke_stage_wall_seconds":smoke_wall,"physical_TPU_chips":4,"gpu_hours":0,
        "allocated_accelerator_hours_smoke1chip_plus_full4chips":allocated,
        "summed_rollout_accelerator_hours":busy_seconds/3600,
        "successful_diffusion_NFE":sum(r["diffusion_NFE"] for r in rows),
        "successful_verifier_evaluations":successful_verifier,
        "failed_attempt_actual_NFE_and_verifier_work":"unknown" if failed else 0,
        "mean_runtime_by_policy":{n:float(np.mean([r["wallclock_seconds"] for r in rows if r["policy_name"]==n])) for n in specs},
        "campaign_successful_runs_per_wall_hour":3840/(total_wall/3600),
        "pilot_successful_runs_per_wall_hour":240/pilot["wall_hours"],
        "illustrative_rate_usd_per_TPU_chip_hour":4.2,
        "illustrative_cost_usd":allocated*4.2,"actual_billed_cost_usd":None,
        "rate_note":"Illustrative public us-east1 reference rate, not actual us-central1/account billing; READY idle time outside job excluded",
        "rate_source":"https://cloud.google.com/tpu/pricing",
        "actual_work":"no shared-tree or snapshot reuse; seven smoke rows reused in full logical campaign"}
    write_json(root/"compute_accounting.json",compute)
    return compute


def figures(root,split,values,summaries,boots,effects):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def save(fig,name):fig.tight_layout();fig.savefig(root/name,dpi=180);plt.close(fig)
    main=boots["CONFIRMATORY"];mid=main["midpoint_contrasts"]
    fig,ax=plt.subplots(figsize=(8,4.8))
    for r in mid:
        y=r["mean_delta"];good=r["confirmed_sensitive"]
        ax.errorbar(r["requested_midpoint"],y,yerr=[[y-r["ci95_low"]],[r["ci95_high"]-y]],
            fmt="*" if good else "o",color="#b42318" if good else "#175cd3",capsize=3,markersize=10 if good else 5)
    ax.axhline(0,color="black",lw=1);ax.set(xlabel="Normalized midpoint progress",ylabel="Paired IR gain over Dense10",
        title="40 confirmatory TRAIN prompts; four seeds averaged per prompt")
    ax.text(.02,.98,"Star: Holm p < .05 and mean gain ≥ .01",transform=ax.transAxes,va="top",fontsize=9)
    save(fig,"midpoint_sensitivity.png")
    fig,ax=plt.subplots(figsize=(8,4.8))
    for r in summaries:
        if r["split"]!="CONFIRMATORY" or r["policy_name"] not in ("Dense10","Early14","Mid12","Late12","Late18","Dense20"):continue
        ax.scatter(r["num_search_events"],r["mean_ImageReward"])
        offset=(6,8) if r["policy_name"]!="Mid12" else (6,-15)
        ax.annotate(r["policy_name"],(r["num_search_events"],r["mean_ImageReward"]),xytext=offset,textcoords="offset points")
    ax.set(xlabel="Search/resampling events",ylabel="Mean terminal ImageReward",title="Reference quality versus events (confirmatory)")
    save(fig,"quality_vs_events.png")
    fig,ax=plt.subplots(figsize=(8,4.6));primary=main["primary_contrasts"]
    for i,r in enumerate(primary):
        y=r["mean_delta"]
        ax.errorbar(y,i,xerr=[[y-r["ci95_low"]],[r["ci95_high"]-y]],fmt="o",capsize=3,
                    color="#b42318" if r["holm_p"]<.05 else "#175cd3")
    ax.set(yticks=range(5),yticklabels=[r["policy_a"]+" − "+r["policy_b"] for r in primary],
           xlabel="Mean paired ImageReward difference",title="Primary contrasts: prompt bootstrap 95% CIs")
    ax.axvline(0,color="black",lw=1);ax.invert_yaxis();save(fig,"primary_contrasts.png")
    fig,ax=plt.subplots(figsize=(6,5))
    x=np.array([values[p,"Dense20"]-values[p,"Dense10"] for p in split["CONFIRMATORY"]])
    y=np.array([values[p,"Late12"]-values[p,"Dense10"] for p in split["CONFIRMATORY"]])
    ax.scatter(x,y,alpha=.8);lo=min(x.min(),y.min(),0);hi=max(x.max(),y.max(),0)
    ax.plot([lo,hi],[lo,hi],"--",color="gray");ax.axhline(0,color="black",lw=.6);ax.axvline(0,color="black",lw=.6)
    ax.set(xlabel="Dense20 − Dense10 (four-seed prompt mean)",ylabel="Late12 − Dense10 (four-seed prompt mean)",
           title="Targeted late versus global refinement");save(fig,"late12_vs_dense20.png")
    fig,ax=plt.subplots(figsize=(8,5.5));xs=[];ys=[]
    for name in effects["CONFIRMATORY"]:
        if name=="Dense10":continue
        x=effects["CONFIRMATORY"][name]["mean_delta"];y=effects["REPLICATION"][name]["mean_delta"]
        xs.append(x);ys.append(y);ax.scatter(x,y);ax.annotate(name,(x,y),xytext=(4,4),textcoords="offset points",fontsize=8)
    lo=min(xs+ys+[0]);hi=max(xs+ys+[0]);ax.plot([lo,hi],[lo,hi],"--",color="gray")
    ax.axhline(0,color="black",lw=.6);ax.axvline(0,color="black",lw=.6)
    ax.set(xlabel="Effect on 40 confirmatory prompts",ylabel="Effect on 20 replication prompts (fresh seeds)",
           title="Fresh-to-pilot versus replication effects over Dense10");save(fig,"fresh_vs_replication.png")


def report(root,result,boots,effects,power,compute,plan):
    labels=("Dense10 is too coarse globally","Late refinement is genuinely useful",
            "Late12 is non-inferior to Dense20","0.05 late spacing is already sufficiently dense",
            "Early refinement is also necessary","At least one individual interval is confirmed sensitive")
    checklist=["# Decision checklist",""]
    for i,label in enumerate(labels,1):
        q=f"Q{i}";checklist.append(f"[{'x' if result['checklist'][q] else ' '}] {q}. {label}")
    checklist += ["",result["Q4_caveat"],"All decisions use only the 40 confirmatory prompts."]
    (root/"CHECKLIST.md").write_text("\n".join(checklist)+"\n")
    lines=[f"DECISION: {result['case']}",""]
    summary_labels=("Dense10 too coarse","Late refinement useful","Late12 non-inferior to Dense20",
                    "late 0.05 spacing saturated","early refinement necessary","confirmed sensitive midpoint exists")
    for i,label in enumerate(summary_labels,1):
        lines.append(f"Q{i} {label}: {'YES' if result['checklist'][f'Q{i}'] else 'NO'}")
    lines += ["",f"Recommended next reference: {result['recommended_next_reference']}","",
        "# Preregistered TRAIN grid-resolution campaign", "",
        "SD1.5 BF16, DDIM100 eta=1, N=8, Max+SSP, ImageReward; tau=8 at every event.",
        "40 prompts not used in the previous midpoint pilot are the CONFIRMATORY subset.",
        "20 previous pilot prompts with four fresh seeds are REPLICATION only.",
        "The entire pool consists of previously explored TRAIN prompts; fresh means fresh to this midpoint pilot.",
        "VALIDATION and TEST remain untouched; no external evaluation pool was used.",
        "Earlier pilot policy comparisons motivate these hypotheses. The 40-prompt split supplies the actual fresh-to-pilot confirmation.",
        "No candidate changed after rewards; pooled results cannot rescue a failed confirmatory test.",
        "All inference averages the four paired seed differences within each prompt first.",
        f"{plan['bootstrap_replicates']:,} prompt-cluster bootstrap replicates, {plan['sign_flips']:,} two-sided Monte Carlo sign flips.",
        "Separate Holm families: five primary contrasts, ten midpoint contrasts; alpha=.05.",
        "Non-inferiority uses the one-sided 95% lower bound > -.02; this is not equivalence.",
        "Late18 requested .725/.775/... maps half-up to 73/78/83/88/93/98; actual progress is .73/.78/.../.98.","",
        "## CONFIRMATORY RESULTS",""]
    for part,heading in (("CONFIRMATORY",None),("REPLICATION","REPLICATION RESULTS"),("POOLED","POOLED DESCRIPTIVE RESULTS")):
        if heading: lines += ["",f"## {heading}",""]
        lines += ["| Contrast | Mean Δ | Median | SE | 95% CI | Raw p | Holm p | W/T/L |",
                  "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |"]
        for r in boots[part]["primary_contrasts"]:
            lines.append(f"| {r['policy_a']} − {r['policy_b']} | {r['mean_delta']:+.5f} | {r['median_delta']:+.5f} | {r['SE']:.5f} | [{r['ci95_low']:+.5f}, {r['ci95_high']:+.5f}] | {r['raw_p']:.5g} | {r['holm_p']:.5g} | {r['wins']}/{r['ties']}/{r['losses']} |")
        ni=boots[part]["primary_contrasts"][2]
        lines += ["",f"Late12 − Dense20 one-sided 95% lower bound: {ni['one_sided95_lower']:+.5f} (margin −.02)."]
        if part=="CONFIRMATORY":
            lines += ["", "| Midpoint | Mean Δ | SE | 95% CI | Raw p | Holm p | W/T/L | Sensitive |",
                      "| --- | ---: | ---: | --- | ---: | ---: | --- | --- |"]
            for r in boots[part]["midpoint_contrasts"]:
                lines.append(f"| {r['requested_midpoint']:.2f} | {r['mean_delta']:+.5f} | {r['SE']:.5f} | [{r['ci95_low']:+.5f}, {r['ci95_high']:+.5f}] | {r['raw_p']:.5g} | {r['holm_p']:.5g} | {r['wins']}/{r['ties']}/{r['losses']} | {'YES' if r['confirmed_sensitive'] else 'NO'} |")
    lines += ["", "## Sign consistency and quality–compute", "",
        "| Policy vs Dense10 | Confirmatory Δ | Replication Δ | Pooled Δ |", "| --- | ---: | ---: | ---: |"]
    for name in effects["CONFIRMATORY"]:
        if name=="Dense10":continue
        d=[effects[part][name]["mean_delta"] for part in ("CONFIRMATORY","REPLICATION","POOLED")]
        lines.append(f"| {name} | {d[0]:+.5f} | {d[1]:+.5f} | {d[2]:+.5f} |")
    lines += ["",f"Effect sign reversals: {', '.join(result['effect_sign_reversals']) or 'none'}.",
        "Late12 adds 3 events; Dense20 adds 10: extra-event ratio 30%.",
        "Verifier evaluations include eight particle scores per search event and eight terminal scores.",
        "Every logical rollout uses the same 800 diffusion NFE. Verifier budgets differ by design."]
    capture=boots["CONFIRMATORY"]["capture_late12"]
    if capture["reported"]: lines.append(f"Confirmatory gain capture: {capture['capture_late12']:.3f}, 95% prompt-bootstrap ratio CI {capture['ci95']}.")
    else: lines.append("Gain capture NOT REPORTED: Dense20−Dense10 denominator is unstable around zero.")
    lines += ["", "## Precision and planning", "", result["Q4_caveat"],
        "Decision F has precedence over A when C1's CI includes both zero and +.01, as frozen before rewards.",
        "Power estimates use the observed prompt SD, normal paired design, two-sided alpha=.05 and 80% power; they are planning approximations, without Holm inflation.",
        "No additional prompts are added automatically.","",
        "| Contrast | Prompt SD | CI width | Prompts for .01 | Prompts for .02 | Prompts for observed effect |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for i,r in enumerate(boots["CONFIRMATORY"]["primary_contrasts"]):
        pp=power[i*3:i*3+3]
        lines.append(f"| {r['contrast']} | {r['prompt_SD']:.5f} | {r['ci95_high']-r['ci95_low']:.5f} | {pp[0]['independent_prompts_for_80pct_power']} | {pp[1]['independent_prompts_for_80pct_power']} | {pp[2]['independent_prompts_for_80pct_power']} |")
    lines += ["", "## Compute accounting", "",
        f"Logical/successful runs: 3840/{compute['successful_runs']}; failed attempts: {compute['failed_attempts']}; rerun units: {compute['rerun_units']}.",
        f"Wall-clock including smoke: {compute['wall_seconds_including_smoke']/3600:.3f} h; full stage: {compute['full_stage_wall_seconds']/3600:.3f} h.",
        f"Allocated TPU-chip-hours (smoke one chip, full four): {compute['allocated_accelerator_hours_smoke1chip_plus_full4chips']:.3f}; GPU-hours: 0.",
        f"Successful diffusion NFE: {compute['successful_diffusion_NFE']:,}; verifier particle evaluations: {compute['successful_verifier_evaluations']:,}.",
        f"Throughput: {compute['campaign_successful_runs_per_wall_hour']:.1f} runs/h; previous pilot: {compute['pilot_successful_runs_per_wall_hour']:.1f} runs/h.",
        f"Illustrative cost at $4.20/chip-hour: ${compute['illustrative_cost_usd']:.2f}. Actual account billing rate is unknown; public us-east1 tariff is only a reference scenario. READY idle outside the job is excluded.",
        "Per-policy runtime, verifier count and gain per extra event are in policy_summary.csv.",
        "No shared-tree reuse; smoke results are reused as seven of the 3840 logical rows.","",
        "Production graph and compiled BFS schedule were not modified. Human review is required before adopting a reference candidate."]
    (root/"REPORT.md").write_text("\n".join(lines)+"\n")
