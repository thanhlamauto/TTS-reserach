"""Offline integrity, paired policy audit, honest work ledger and final plots."""
from pathlib import Path
import time, json, math
import numpy as np
from .adaptive_refinement import *

def finalize(root):
    cfg=read(root/'config.json');split=read(root/'data_split.json');policies=read(root/'frozen_policies.json')
    assert sha(root/'frozen_policies.json')==read(root/'policy_freeze_receipt.json')['sha256']
    graphs={n:read(root/f'{n}_graph.json') for n in ['coarse','refined','fine']};confirmed=read(root/'midpoint_confirmation.json');proposals=read(root/'frozen_proposals.json')['proposals']
    # All logical graph observations must exist, even if identical executions were reused.
    for g in graphs.values():
        for j in jobs_for(g['nodes']):
            for pid in split['A']:
                for seed in cfg['seeds']['A']:
                    row=read(unitpath(root,'A',pid,seed,j));assert row['job']==j and math.isfinite(row['regret'])
    assert read(root/'gates.json')['passed'];assert not list((root/'failures').glob('*.json'))
    for j in proposals:
        for pid in split['B']:
            for seed in cfg['seeds']['B']:
                for version in [job(j['src'],j['dst'],j['tau']),job(j['src'],j['dst'],j['tau'],j['mid'],j['mtau'])]:
                    row=read(unitpath(root,'B',pid,seed,version));assert row['job']==version and math.isfinite(row['reward'])
    for pid in split['C']:
        for seed in cfg['seeds']['C']:
            initial_hashes={read(root/'audit'/f'p{pid}-s{seed}-{p["id"]}.json')['initial_latents_sha256'] for p in policies.values()}
            assert len(initial_hashes)==1,'audit initial particle pairing differs'
    assert len(list((root/'audit').glob('*.json')))==len({p['id'] for p in policies.values()})*80
    rng=np.random.default_rng(cfg['stat_seed']+3);idx=rng.integers(0,20,size=(50000,20));reward={};per=[]
    for name,pol in policies.items():
        values=[]
        for pid in split['C']:
            units=[read(root/'audit'/f'p{pid}-s{s}-{pol["id"]}.json') for s in cfg['seeds']['C']]
            assert all(u['policy']==pol and len(u['events'])==3 and [e['sampling_index'] for e in u['events']]==pol['resampling_steps'] for u in units)
            value=float(np.mean([u['reward'] for u in units]));values.append(value);per.append({'policy':name,'prompt':pid,'mean_IR_over4seeds':value})
        reward[name]=np.array(values)
    pairs=[('P_ADAPTIVE','P_COARSE','primary'),('P_ADAPTIVE','P_FINE','primary'),('P_ADAPTIVE','P_MANUAL','primary'),('P_FINE','P_COARSE','secondary'),('P_FINE','P_MANUAL','secondary'),('P_ADAPTIVE','TRANSFER4_TO_8','secondary')];audit=[]
    for a,b,family in pairs:
        x=reward[a]-reward[b];lo,hi=ci(x,idx);bs=x[idx].mean(axis=1)
        audit.append({'A':a,'B':b,'family':family,'mean_IR_A':float(reward[a].mean()),'mean_IR_B':float(reward[b].mean()),'delta':float(x.mean()),'SE':float(x.std(ddof=1)/np.sqrt(20)),'CI_low':lo,'CI_high':hi,'one_sided95_lower':float(np.quantile(bs,.05)),'wins':int(sum(x>0)),'ties':int(sum(x==0)),'losses':int(sum(x<0))})
    csvout(root/'audit_per_prompt.csv',per);csvout(root/'audit_summary.csv',audit)
    ledgers=[read(p) for p in(root/'work_ledgers').glob('*.json')];stage_work={}
    for row in ledgers:
        totals=stage_work.setdefault(row['stage'],{'NFE':0,'verifier_candidate_scores':0,'verifier_batch_calls':0,'worker_chip_hours':0.,'reference_trajectories':0,'new_observations':0})
        for a,b in [('NFE','actual_denoising_NFE'),('verifier_candidate_scores','actual_verifier_candidate_scores'),('verifier_batch_calls','actual_verifier_batch_calls'),('reference_trajectories','reference_trajectories'),('new_observations','logical_observations_new')]:totals[a]+=row[b]
        totals['worker_chip_hours']+=row['wall_seconds_including_load']/3600
    a_split=len(split_jobs(root,graphs['coarse']))*20;b_jobs={jobid(j) for p in proposals for j in [job(p['src'],p['dst'],p['tau']),job(p['src'],p['dst'],p['tau'],p['mid'],p['mtau'])]};b_obs=len(b_jobs)*80
    coarse=51*20;new=(graphs['refined']['logical_edges']-51)*20;fine=145*20;allin=coarse+a_split+b_obs+new;graph_only=coarse+new
    accounting={'stage_actual_work':stage_work,'logical_coarse_edge_observations':coarse,'logical_discovery_split_observations':a_split,'logical_confirmation_coarse_and_split_observations':b_obs,'logical_new_refined_edge_observations':new,'logical_fine_edge_observations_standalone':fine,'adaptive_all_in_observation_ratio':allin/fine,'adaptive_graph_only_observation_ratio':graph_only/fine,'ratios_include_B80_and_A20_different_sample_counts':True,'unique_actual_executed_NFE':sum(r['actual_denoising_NFE'] for r in ledgers),'actual_verifier_candidate_scores':sum(r['actual_verifier_candidate_scores'] for r in ledgers),'actual_verifier_batch_calls':sum(r['actual_verifier_batch_calls'] for r in ledgers),'TPU_chip_hours_including_load':sum(r['wall_seconds_including_load'] for r in ledgers)/3600,'campaign_wall_seconds':time.time()-read(root/'driver.pid.json')['started_unix'],'actual_billing':'UNKNOWN','approximate_cost_scenario_historical_rate4_20_per_chip_hour':4.2*sum(r['wall_seconds_including_load'] for r in ledgers)/3600,'cost_scenario_note':'Historical illustrative rate only; not current quote or actual billing','audit_unique_policy_runs':len({p['id'] for p in policies.values()})*80,'audit_logical_policy_runs':len(policies)*80,'audit_unique_diffusion_NFE':len({p['id'] for p in policies.values()})*80*800,'fine_actual_incremental_work':stage_work.get('fine'),'baseline_standalone_actual_work':'not directly timed: cache reuse confounds wall-clock; logical standalone work reported, no speedup claim from incremental cost'}
    dump(root/'compute_accounting.json',accounting)
    comparisons=[{'graph':n,'nodes':len(g['nodes']),'logical_edges':g['logical_edges'],'logical_edge_observations':g['edge_observations'],'top1_bootstrap_frequency':g['bootstrap_exact_top1_frequency'],'candidate_entropy':g['candidate_entropy'],'logical_full_rollout_NFE_equivalent':g['edge_observations']*800,'actual_incremental_NFE':stage_work.get({'coarse':'coarse','refined':'refinement','fine':'fine'}[n],{}).get('NFE',0)} for n,g in graphs.items()];csvout(root/'graph_comparison.csv',comparisons)
    newnodes=sorted(set(graphs['refined']['nodes'])-set(G0));AF=next(a for a in audit if a['A']=='P_ADAPTIVE' and a['B']=='P_FINE');AM=next(a for a in audit if a['A']=='P_ADAPTIVE' and a['B']=='P_MANUAL')
    conditions={'A_confirmed_new_midpoint':bool(newnodes),'B_all_in_ratio_at_most_60pct':allin/fine<=.6,'C_descriptive_noninferiority_AF':AF['one_sided95_lower']>=-.02,'D_manual_competitive_point_delta_ge_minus02':AM['delta']>=-.02}
    # Predeclared diagnostic labels; no statistical stability/equivalence theorem.
    if graphs['fine']['bootstrap_exact_top1_frequency']<.5:outcome='E'
    elif not newnodes:outcome='A'
    elif AF['one_sided95_lower']<-.02:outcome='D'
    elif len(newnodes)>=3 and len(newnodes)>=len({p['mid'] for p in proposals if p['new_node']})/2:outcome='C'
    else:outcome='B'
    dump(root/'decision.json',{'outcome':outcome,'success_conditions':conditions,'supports_all_criteria':all(conditions.values()),'interpretation_rules_predeclared_in_report_source':True,'fine_instability_flag_threshold_exact_path_frequency':.5,'threshold_is_descriptive_not_test':True,'no_confirmation_not_proof_coarse_sufficient':True,'C_is_internal_not_TEST':True})
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5))
    for row in confirmed:
        color={'CONFIRMED':'green','BORDERLINE':'orange','NO':'gray'}[row['verdict']]
        ax.errorbar(row['mid'],row['mean_gain_B'],yerr=[[max(0,row['mean_gain_B']-row['CI_low'])],[max(0,row['CI_high']-row['mean_gain_B'])]],fmt='o',color=color,markersize=4+8*row['edge_bootstrap_appearance'])
        ax.annotate(f"{row['src']}→{row['dst']} τ={row['tau']}/{row['mtau']}",(row['mid'],row['mean_gain_B']),fontsize=6,xytext=(3,3),textcoords='offset points')
    if not confirmed:ax.text(.5,.5,'No proposed midpoint tests',transform=ax.transAxes,ha='center')
    ax.axhline(0,color='black',lw=.7);ax.axhline(.01,color='gray',ls='--');ax.set_xlabel('Midpoint sampling index');ax.set_ylabel('Paired split gain, StageB (95% prompt CI)');ax.set_title('Sensitivity map; size = top16 bootstrap edge appearance');fig.tight_layout();fig.savefig(root/'sensitivity_map.png',dpi=160);plt.close(fig)
    for name,g in graphs.items():
        fig,ax=plt.subplots(figsize=(12,4));nodes=[-1]+g['nodes']+[100];selected=g['top1']['policy'];es=g['top1']['edge_keys']
        ax.scatter(nodes,[0]*len(nodes),c=['orange' if t in newnodes and name=='refined' else 'steelblue' for t in nodes],s=90)
        for t in nodes:ax.text(t,-.12,'START' if t==-1 else 'END' if t==100 else str(t),ha='center')
        for e in g['edges']:
            if e['key'] in es:
                ax.annotate('',xy=(e['dst'],0),xytext=(e['src'],0),arrowprops={'arrowstyle':'->','color':'black'});ax.text((e['src']+e['dst'])/2,.12,f"τ={e['tau_dst']}\nw={e['mean_regret']:.3f}",ha='center',fontsize=8)
        ax.set_ylim(-.4,.55);ax.set_xlim(-8,108);ax.axis('off');ax.set_title(f"{name.upper()} nodes; selected K3 path: {selected['resampling_steps']}");fig.tight_layout();fig.savefig(root/({'refined':'adaptive'}.get(name,name)+'_graph.png'),dpi=160);plt.close(fig)
    lines=['# ADAPTIVE GRAPH REFINEMENT','',f'Confirmed midpoint nodes: {newnodes}',f'Coarse nodes: {list(G0)}',f'Final adaptive nodes: {graphs["refined"]["nodes"]}',f'G_FINE nodes: {list(FINE)}','']
    for n,g in graphs.items():lines +=[f'{n.upper()} top-1:',json.dumps(g['top1']['policy']), 'START -> '+' -> '.join(map(str,g['top1']['policy']['resampling_steps']))+' -> END']
    lines +=['',f'Profiling observations: Coarse={coarse}; Adaptive all-in={allin}; Fine={fine}',f'Adaptive/Fine all-in={allin/fine:.4f}; graph-only={graph_only/fine:.4f}','', 'Stage-C rewards:',json.dumps({n:float(v.mean()) for n,v in reward.items()},indent=2),'Adaptive - Fine:',json.dumps(AF,indent=2),'Adaptive - Manual:',json.dumps(AM,indent=2),'FINAL INTERPRETATION: '+outcome,json.dumps(conditions,indent=2),'','Internal/diagnostic TRAIN only. StageC overlaps StageB prompts, with fresh seeds. No TEST/generalization claim. No-confirmation does not prove coarse sufficient. Fine instability label uses predeclared descriptive exact-path frequency<0.5, not a hypothesis test. Endpoint/RNG confound of the existing backend remains; all policies use that same audited protocol. Fixed K3; local split diagnostics can have an extra event and do not change deployment K.','', 'Reference generation and real work:',json.dumps(accounting,indent=2)]
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    dump(root/'INTEGRITY.json',{'passed':True,'graph_observations_verified':True,'policy_hash_verified':True,'all_C_units_verified':True,'failure_count':0,'gates_passed':True})
    (root/'CHECKLIST.md').write_text('\n'.join(['# Final checklist','- [x] gates1–8 passed','- [x] TRAIN-only frozen data/seeds/reference','- [x] no production change','- [x] logical graph/cache observations complete','- [x] StageC freeze hash and paired units verified','- [x] required tables/plots/report generated','- [x] all-in adaptive costs include discovery and confirmation','- [x] no TEST generalization claim'])+'\n')
