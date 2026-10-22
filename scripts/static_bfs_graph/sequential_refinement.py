"""Prospective budgeted edge-split racing. Allocation intervals are heuristics."""
from .adaptive_refinement import *
from .graph_builder import path_edges
import os,time,subprocess,sys,fcntl,traceback,shutil
REPO=Path(__file__).resolve().parents[2]
SOURCES=['sequential_refinement.py','sequential_refinement_tpu.py','adaptive_refinement.py','adaptive_refinement_tpu.py','graph_builder.py','adapter.py','runner.py','policy.py','tpu_reference_compare.py']

def freeze(root):
    if (root/'manifest.json').exists():verify(root);return
    root.mkdir(parents=True,exist_ok=True);pool,cfg,old=checked_inputs();ids=pool['train_indices']
    assert len(ids)==60 and len(jobs_for(G0))==51 and len(jobs_for(FINE))==145
    ref=records()['DENSE_REFERENCE'];assert ref['resampling_steps']==[10,20,30,40,60,80,90]
    config={'protocol':'BUDGETED_SEQUENTIAL_REFINEMENT_V1','seeds':{'A':[2026100811],'B':[2026100812,2026100813],'C':[2026100812,2026100813]},'snapshots':list(range(5,100,5)),'N':8,'K':3,'steps':100,'eta':1.,'dtype':'BF16','taus':list(TAUS),'max_observations':2030,'preferred_observations':1740,'checkpoints':[5,10,15,20],'graph_bootstraps':20000,'audit_bootstraps':50000,'stat_seed':2026100814}
    # Rules fixed before any new generation. Old audit rewards never enter allocator.
    prereg={'candidate_cap':12,'candidate_retention':'point leader plus largest joint-bootstrap winner frequencies, deterministic ID ties','candidate_mass_target':.9,'midpoint_location_cap':4,'split_probe':'one highest importance*SEM interval per location; all three tau_mid on first5; best mean tau retained if mean>=.01 OR bootstrap P(gain>0)>=.8','replacement_paths':'replace one existing event by midpoint in every retained coarse candidate; tau_mid from split screen; retain other temperatures; at most12 proposed paths chosen by split-gain minus source path cost, deterministic ties','eliminate':'paired probability(cost_candidate-cost_leader>.01)>=.95 AND winner_probability<.01; incumbent and leader protected','stable':'same leader at two adjacent looks, winner_probability>=.8 AND paired bootstrap P(cost_leader-cost_each_competitor<.02)>=.95','underidentified':'no stable leader by n20, or next comparable paired batch exceeds hard budget; fallback transfer','budget_unit':'each newly generated calibration intervention JSON including failed probes; correctness gate observations separate calibration-external mandatory preflight','allocation':'joint common first-n trajectory IDs for entire live candidate edge union; batch reservation before execute; uncertainty edge SEM and pair covariance reported; eliminated candidates retain earlier estimates only and never compared at incompatible n','midpoints':'rounded5 midpoint withSTART treated0; width>=10; exclude existingG0nodes; no historical node preference','audit':'40 disjoint-within-campaign TRAIN prompts*2freshseeds; historical exposure disclosed; freeze all policies before access;50k paired prompt bootstrap','NI_margin':-.02,'no_independent_TEST_claim':True,'no_formal_sequential_guarantees':True,'full_fine_rebuild':False,'ablations':'historical fixed+fine actual records; retrospective sequential-all-expansion and lazy counterfactual edge-union cost estimates, no wall-clock or prospective causal saving claim','no_production_update':True}
    split={'A':ids[:20],'B':ids[20:],'C':ids[20:],'TRAIN_prompts':pool['prompts'],'historically_explored':True,'compiler_audit_prompt_disjoint':True}
    for n,d in [('config.json',config),('preregistration.json',prereg),('data_split.json',split),('dataset_split.json',split),('reference_schedule.json',ref)]:dump(root/n,d)
    source={str((REPO/'scripts/static_bfs_graph'/n).relative_to(REPO)):sha(REPO/'scripts/static_bfs_graph'/n) for n in SOURCES}
    for n in ['configs/static_bfs_graph_t2i.json','configs/diffusion_classical_search_t2i_table12.json','modal_n8_gpu/train_pool.json','scripts/diffusion_classical_search_t2i/tpu_runner.py']:source[n]=sha(REPO/n)
    dump(root/'manifest.json',{'sources':source,'frozen':{n:sha(root/n) for n in ['config.json','preregistration.json','dataset_split.json','data_split.json','reference_schedule.json']},'model_revision':old['models']['sd15']['revision']})
    (root/'commands.md').write_text('cd /home/thanhlamtba31/TTS-reserach\n/home/thanhlamtba31/static-bfs-dcs/venv/bin/python -m scripts.static_bfs_graph.sequential_refinement --run-root '+str(root)+'\n')

def verify(root):
    m=read(root/'manifest.json');assert all(sha(REPO/n)==h for n,h in m['sources'].items())
    assert all(sha(root/n)==h for n,h in m['frozen'].items())

def count(root):return len(list((root/'observations'/'A').glob('*/*.json')))
def profile(root,label,jobs,n):
    cfg=read(root/'config.json');ids=read(root/'data_split.json')['A'][:n];seed=cfg['seeds']['A'][0]
    jobs=list({jobid(j):j for j in jobs}.values());missing=sum(not unitpath(root,'A',p,seed,j).exists() for p in ids for j in jobs)
    if count(root)+missing>cfg['max_observations']:return False
    dump(root/'budget_reservation.json',{'stage':label,'used':count(root),'reserved_new':missing,'total_max':count(root)+missing,'cap':2030})
    stage(root,label,'A',jobs,ids);assert count(root)<=2030;return True

def stage(root,name,cohort=None,jobs=None,ids=None):
    verify(root)
    if (root/'stage_ledger'/f'{name}.json').exists():return
    assert shutil.disk_usage(root).free>5*1024**3,'disk resource stop'
    if cohort:dump(root/'tasks'/f'{name}.json',{'cohort':cohort,'jobs':jobs,'ids':ids})
    dump(root/'status.json',{'stage':name,'status':'RUNNING','start_unix':time.time()})
    chips=[0] if name=='gates' else list(range(4));procs=[];logs=[];start=time.time()
    try:
        for chip in chips:
            p=root/'logs'/f'{name}-chip{chip}.log';p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a');logs.append(f)
            env={**os.environ,'PJRT_DEVICE':'TPU','TPU_VISIBLE_CHIPS':str(chip),'PYTHONUNBUFFERED':'1','STATIC_BFS_CACHE_ROOT':f'/dev/shm/static-bfs-sequential-v1/chip{chip}/cache'}
            procs.append(subprocess.Popen([sys.executable,'-m','scripts.static_bfs_graph.sequential_refinement_tpu','--run-root',str(root),'--stage',name,'--chip',str(chip)],env=env,stdout=f,stderr=subprocess.STDOUT))
        dump(root/'active_workers.json',{'stage':name,'PIDs':[p.pid for p in procs]})
        while True:
            states=[p.poll() for p in procs]
            if any(x not in (None,0) for x in states):raise RuntimeError(f'{name}: {states}')
            if all(x==0 for x in states):break
            time.sleep(5)
        for chip in chips:assert read(root/'workers'/name/f'chip{chip}.done.json')['success']
        dump(root/'stage_ledger'/f'{name}.json',{'complete':True,'wall_seconds':time.time()-start,'chips':chips})
    finally:
        for p in procs:
            if p.poll() is None:p.terminate()
        for p in procs:
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for f in logs:f.close()

def edges_of(rec):return [job(e.src,e.dst,e.tau_dst) for e in path_edges(tuple(rec['resampling_steps']),tuple(t for s,t in rec['temperature_by_step']))]
def union(candidates):return list({jobid(j):j for p in candidates for j in edges_of(p)}.values())
def values(root,j,n):
    c=read(root/'config.json');ids=read(root/'data_split.json')['A'][:n]
    return np.array([read(unitpath(root,'A',p,c['seeds']['A'][0],j))['regret'] for p in ids])
def rank_arrays(x,seed,reps=20000):
    rng=np.random.default_rng(seed);idx=rng.integers(0,len(x),size=(reps,len(x)));b=x[idx].mean(axis=1);w=np.bincount(np.argmin(b,axis=1),minlength=x.shape[1])/reps
    return x.mean(axis=0),b,w

def evaluate(root,candidates,n):
    candidates=sorted({p['id']:p for p in candidates}.values(),key=lambda p:p['id'])
    x=np.stack([sum((values(root,j,n) for j in edges_of(p)),np.zeros(n)) for p in candidates],axis=1)
    mean,b,prob=rank_arrays(x,read(root/'config.json')['stat_seed']+n)
    order=np.argsort(mean,kind='stable');rows=[{'policy':p,'predicted_regret':float(mean[i]),'winner_probability':float(prob[i]),'SE':float(x[:,i].std(ddof=1)/np.sqrt(n))} for i,p in enumerate(candidates)]
    return candidates,x,b,prob,order,rows

def shortlist(root):
    paths=[static_record(policy_from_path(es,8)) for es in all_paths(G0,TAUS,3)]
    p,x,b,prob,order,rows=evaluate(root,paths,5);selected=[int(order[0])]
    selected+= [i for i in sorted(range(len(p)),key=lambda i:(-prob[i],p[i]['id'])) if i not in selected][:11]
    kept=[p[i] for i in selected];inc=records()['TRANSFER4_TO_8'];kept=list({q['id']:q for q in kept+[inc]}.values())
    mass=float(sum(prob[i] for i in selected));dump(root/'coarse_graph.json',{'nodes':G0,'logical_edges':51,'samples':5,'path_count':270,'top1':rows[int(order[0])],'candidate_mass':mass,'mass_under90':mass<.9})
    dump(root/'coarse_candidates.json',kept);dump(root/'coarse_policy.json',p[int(order[0])]);return kept,mass

def screening(root,candidates):
    p,x,b,prob,order,rows=evaluate(root,candidates,5);importance={}
    for i,q in enumerate(p):
        for j in edges_of(q):importance[jobid(j)]=importance.get(jobid(j),0)+prob[i]
    options=[]
    for j in union(p):
        m=mid(j['src'],j['dst'])
        if m is None or m in G0 or j['dst']-max(j['src'],0)<10:continue
        sem=float(values(root,j,5).std(ddof=1)/np.sqrt(5));score=importance[jobid(j)]*(sem+.000001)
        options.append((score,jobid(j),m,j))
    options.sort(key=lambda t:(-t[0],t[1]));chosen=[];seen=set()
    for _,_,m,j in options:
        if m not in seen:seen.add(m);chosen.append(j|{'mid':m})
        if len(seen)==4:break
    probes=[job(j['src'],j['dst'],j['tau'],j['mid'],t) for j in chosen for t in TAUS]
    assert profile(root,'split5',probes,5)
    results=[];promising=[];rng=np.random.default_rng(read(root/'config.json')['stat_seed']);idx=rng.integers(0,5,size=(20000,5))
    for j in chosen:
        versions=[]
        for t in TAUS:
            z=job(j['src'],j['dst'],j['tau'],j['mid'],t);gain=values(root,job(j['src'],j['dst'],j['tau']),5)-values(root,z,5)
            versions.append((float(gain.mean()),t,z,gain));results.append(z|{'mean_gain':float(gain.mean()),'P_positive':float(np.mean(gain[idx].mean(axis=1)>0))})
        mean,t,z,gain=sorted(versions,key=lambda v:(-v[0],v[1]))[0]
        if mean>=.01 or np.mean(gain[idx].mean(axis=1)>0)>=.8:promising.append(z|{'mean_gain':mean})
    csvout(root/'midpoint_proposals.csv',results,fields=['src','dst','tau','mid','mtau','mean_gain','P_positive'])
    dump(root/'split_screening.json',{'proposed_locations':sorted(seen),'results':results,'promising':promising,'local_extra_event_not_K3_proof':True})
    proposals=[]
    cost={q['policy']['id']:q['predicted_regret'] for q in rows}
    for z in promising:
        for q in candidates:
            for removed in q['resampling_steps']:
                temperatures=dict(q['temperature_by_step']);temperatures.pop(removed);temperatures[z['mid']]=z['mtau']
                if len(temperatures)!=3:continue
                pp=static_record(StaticBFSPolicy(tuple(sorted(temperatures)),tuple(sorted(temperatures.items())),8))
                proposals.append((cost[q['id']]-z['mean_gain'],pp['id'],pp))
    proposals.sort(key=lambda z:(z[0],z[1]));unique={}
    for _,k,q in proposals:
        if k not in unique:unique[k]=q
        if len(unique)==12:break
    dump(root/'path_proposals.json',list(unique.values()));return list(unique.values())

def race(root,candidates,mass):
    prev=None;rounds=[];inc=records()['TRANSFER4_TO_8'];reason='UNDERIDENTIFIED';selected=False;leader=None
    for n in [5,10,15,20]:
        if not profile(root,f'race{n}',union(candidates),n):reason='BUDGET_PREVENTS_NEXT_PAIRED_LOOK';break
        candidates,x,b,prob,order,rows=evaluate(root,candidates,n);k=int(order[0]);leader=candidates[k]
        gaps=b-b[:,k,None];live=[];details=[]
        for i,p in enumerate(candidates):
            worse=float(np.mean(gaps[:,i]>.01));drop=bool(i!=k and p['id']!=inc['id'] and worse>=.95 and prob[i]<.01)
            if not drop:live.append(p)
            dif=x[:,i]-x[:,k];details.append({'policy_id':p['id'],'leader_id':leader['id'],'mean_gap':float(dif.mean()),'paired_SE':float(dif.std(ddof=1)/np.sqrt(n)),'P_worse_by01':worse,'winner_probability':float(prob[i]),'eliminated':drop,'n':n})
        stable=prev==leader['id'] and prob[k]>=.8 and all(float(np.mean(b[:,k]-b[:,i]<.02))>=.95 for i in range(len(candidates)) if i!=k)
        edge_uncertainty=[{'edge_id':jobid(j),'job':j,'SE':float(values(root,j,n).std(ddof=1)/np.sqrt(n))} for j in union(live)]
        row={'n':n,'live_candidates':len(live),'active_midpoints':sorted({s for p in live for s in p['resampling_steps'] if s not in G0}),'entropy':float(-sum(v*np.log(v) for v in prob if v>0)),'incumbent_rank':next(int(np.where(order==i)[0][0]+1) for i,p in enumerate(candidates) if p['id']==inc['id']),'observations':count(root),'remaining_budget':2030-count(root),'leader':leader,'stable':bool(stable),'candidate_mass_initial':mass,'undercoverage_initial':mass<.9,'details':details,'uncertainty_edges':sorted(edge_uncertainty,key=lambda v:-v['SE'])}
        rounds.append(row);dump(root/'racing_state.json',{'rounds':rounds});(root/'racing_rounds.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in rounds))
        dump(root/'candidate_paths.json',{'live':live,'last_evaluated':rows,'n':n,'actively_profiled_candidate_set_not_global_expanded_optimum':True})
        if stable:selected=True;reason='STABLE_CANDIDATE';break
        prev=leader['id'];candidates=live
    assert leader is not None
    dump(root/'selection.json',{'leading_proposed_policy':leader,'deployed_policy':leader if selected else inc,'selected_by_new_graph':selected,'fallback_to_incumbent':not selected,'reason':reason,'observations':count(root),'initial_candidate_mass':mass})
    return leader,rounds

def bootstrap_diff(x,seed):
    rng=np.random.default_rng(seed);means=x[rng.integers(0,len(x),size=(50000,len(x)))].mean(axis=1)
    return {'delta':float(x.mean()),'SE':float(x.std(ddof=1)/np.sqrt(len(x))),'CI_low':float(np.quantile(means,.025)),'CI_high':float(np.quantile(means,.975)),'one_sided95_lower':float(np.quantile(means,.05)),'wins':int(sum(x>0)),'ties':int(sum(x==0)),'losses':int(sum(x<0))}

def finalize(root,rounds):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cfg=read(root/'config.json');split=read(root/'data_split.json');pol=read(root/'frozen_policies.json');selection=read(root/'selection.json')
    audit=[read(p) for p in (root/'audit').glob('*.json')];expected=len({p['id'] for p in pol.values()})*80;assert len(audit)==expected
    assert not list((root/'failures').glob('*'));lookup={(r['prompt'],r['seed'],r['policy']['id']):r for r in audit}
    summary=[];vectors={}
    for name,p in pol.items():
        rows=[lookup[(pid,s,p['id'])] for pid in split['C'] for s in cfg['seeds']['C']]
        assert all(len(r['events'])==3 for r in rows)
        vectors[name]=np.array([np.mean([lookup[(pid,s,p['id'])]['reward'] for s in cfg['seeds']['C']]) for pid in split['C']])
        summary.append({'policy':name,'mean_IR':float(vectors[name].mean()),'sec_per_run':float(np.mean([r['runtime_seconds'] for r in rows])),'NFE_per_run':800,'intermediate_verifier_scores':24,'terminal_verifier_scores':8,'events':3})
    csvout(root/'audit_summary.csv',summary);(root/'audit_raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in sorted(audit,key=lambda r:(r['prompt'],r['seed'],r['policy']['id']))))
    comparisons=[{'A':'ADAPTIVE_NEW','B':b,**bootstrap_diff(vectors['ADAPTIVE_NEW']-vectors[b],cfg['stat_seed']+i)} for i,b in enumerate(['P_FINE','P_COARSE','P_TRANSFER','P_MANUAL'])];csvout(root/'paired_comparisons.csv',comparisons)
    obs=[read(p) for p in (root/'observations'/'A').glob('*/*.json')];assert len(obs)<=2030
    csvout(root/'coarse_edge_samples.csv',[{'prompt':r['prompt'],'seed':r['seed'],'edge_id':jobid(r['job']),'regret':r['regret']} for r in obs if r['job']['mid'] is None and r['job'] in jobs_for(G0)])
    allocation=[{'prompt':r['prompt'],'seed':r['seed'],'edge_id':jobid(r['job']),'src':r['job']['src'],'dst':r['job']['dst'],'mid':r['job']['mid'],'regret':r['regret']} for r in obs];csvout(root/'sample_allocation.csv',allocation)
    lazy=[r for r in allocation if r['mid'] is None and (r['src'] not in [-1,*G0] or r['dst'] not in [*G0,100])];csvout(root/'lazy_profiled_edges.csv',lazy,fields=list(allocation[0]))
    csvout(root/'racing_survivors.csv',[d for r in rounds for d in r['details']]);csvout(root/'racing_budget.csv',[{k:r[k] for k in ['n','live_candidates','entropy','observations','remaining_budget']} for r in rounds])
    dump(root/'final_candidate_graph.json',{'nodes':sorted({s for p in read(root/'candidate_paths.json')['live'] for s in p['resampling_steps']}),'measured_jobs':list({jobid(r['job']):r['job'] for r in obs if r['job']['mid'] is None}.values()),'missing_edges_not_assigned_zero':True,'solver_scope':'actively profiled path set'})
    ledgers=[read(p) for p in (root/'work_ledgers').glob('*.json')];work={}
    for l in ledgers:
        group='audit' if l['stage']=='audit' else 'correctness' if l['stage']=='gates' else 'compilation'
        w=work.setdefault(group,{k:0 for k in ['NFE','verifier_scores','verifier_calls','references','chip_hours']})
        for out,inp in [('NFE','actual_denoising_NFE'),('verifier_scores','actual_verifier_candidate_scores'),('verifier_calls','actual_verifier_batch_calls'),('references','reference_trajectories')]:w[out]+=l[inp]
        w['chip_hours']+=l['wall_seconds_including_load']/3600
    mids=sorted({s for p in read(root/'candidate_paths.json')['live'] for s in p['resampling_steps'] if s not in G0})
    # Cost-only ablations disclose counterfactual unmatched work. No policy rewards invented.
    all_expansion_edges=len(jobs_for(tuple(sorted(set(G0)|set(mids)))))-51
    ablations=[{'arm':'Full fine historical','observations':2900,'evidence':'historical executed logical observations; different grid'}, {'arm':'Fixed adaptive v1 historical','observations':3800,'evidence':'historical executed discovery+confirmation+expansion'}, {'arm':'Sequential+lazy prospective','observations':len(obs),'evidence':'new executed calibration observations'}, {'arm':'Sequential all-edge expansion counterfactual','observations':len(obs)-len(lazy)+all_expansion_edges*rounds[-1]['n'],'evidence':'counterfactual expansion cost at final live-node set; not a separately run allocator'}]
    csvout(root/'ablations.csv',ablations)
    accounting={'compilation_observations':len(obs),'historical_fine_observations':2900,'ratio':len(obs)/2900,'preferred_met':len(obs)<=1740,'work':work,'wall_seconds':time.time()-read(root/'driver.pid.json')['started_unix'],'actual_billing':'UNKNOWN','no_measured_speedup_claim':True,'correctness_gate_observations_separate':len(list((root/'observations'/'gate').glob('*/*.json'))),'full_fine_standalone_not_executed':True};dump(root/'cost_accounting.json',accounting)
    ni=comparisons[0]['one_sided95_lower']>-.02;newnode=any(s not in G0 for s in pol['ADAPTIVE_NEW']['resampling_steps']);success={'efficiency':len(obs)<=2030,'new_policy_quality':bool(ni and selection['selected_by_new_graph']),'proposed_policy_NI':bool(ni),'adaptive_refinement':bool(selection['selected_by_new_graph'] and newnode),'conservative_fallback':selection['fallback_to_incumbent']};dump(root/'decision.json',success)
    dump(root/'INTEGRITY.json',{'passed':True,'audit_unique_rows':expected,'prompt_count':40,'paired_seeds':2,'compiler_audit_disjoint':not bool(set(split['A'])&set(split['C'])),'observations':len(obs),'cap':2030,'gates':read(root/'gates.json')['passed'],'policy_hash_verified':sha(root/'frozen_policies.json')==read(root/'policy_freeze_receipt.json')['sha256']})
    # Six standalone diagnostic plots.
    for field,out,ylabel in [('live_candidates','budget_vs_live_candidates.png','Live candidates'),('entropy','budget_vs_candidate_entropy.png','Winner entropy')]:
        fig,ax=plt.subplots();ax.plot([r['observations'] for r in rounds],[r[field] for r in rounds],'o-');ax.axvline(2030,color='red',ls='--');ax.set(xlabel='All-in cumulative calibration observations',ylabel=ylabel);fig.tight_layout();fig.savefig(root/out);plt.close(fig)
    screen=read(root/'split_screening.json')['results'];fig,ax=plt.subplots();ax.bar(range(len(screen)),[r['mean_gain'] for r in screen]);ax.set_xticks(range(len(screen)),[f"{r['mid']}:tau{r['mtau']:g}" for r in screen],rotation=60);ax.set_ylabel('Paired split gain, first5 (screening only)');fig.tight_layout();fig.savefig(root/'midpoint_split_gain.png');plt.close(fig)
    fig,ax=plt.subplots();ax.scatter(G0,[0]*len(G0),label='Coarse');nodes=sorted(set(G0)|set(mids));ax.scatter(nodes,[1]*len(nodes),label='Live candidate grid');ax.set(yticks=[0,1],yticklabels=['Coarse','Adaptive'],xlabel='Sampling index');fig.tight_layout();fig.savefig(root/'coarse_vs_adaptive_graph.png');plt.close(fig)
    fig,ax=plt.subplots();ax.scatter([len(obs),255,2900],[vectors['ADAPTIVE_NEW'].mean(),vectors['P_COARSE'].mean(),vectors['P_FINE'].mean()]);ax.set(xlabel='Profiling observations (Fine historical normalized)',ylabel='Audit mean IR');fig.tight_layout();fig.savefig(root/'quality_vs_compilation_cost.png');plt.close(fig)
    fig,ax=plt.subplots();ax.scatter(vectors['P_FINE'],vectors['ADAPTIVE_NEW']);a=min(vectors['P_FINE'].min(),vectors['ADAPTIVE_NEW'].min());z=max(vectors['P_FINE'].max(),vectors['ADAPTIVE_NEW'].max());ax.plot([a,z],[a,z],'--');ax.set(xlabel='Fine prompt mean IR',ylabel='Proposed adaptive prompt mean IR');fig.tight_layout();fig.savefig(root/'adaptive_vs_fine_paired.png');plt.close(fig)
    lines=['# BUDGETED SEQUENTIAL GRAPH REFINEMENT','Correctness gates: PASS',f'All-in profiling observations: {len(obs)}','Historical G_FINE observations: 2900',f'Adaptive / Fine ratio: {len(obs)/2900:.4f}','Target <=0.70: '+('PASS' if len(obs)<=2030 else 'FAIL'),f'Actual profiling NFE: {work["compilation"]["NFE"]}',f'Actual profiling TPU chip-hours: {work["compilation"]["chip_hours"]:.4f}', 'Midpoint locations proposed: '+str(read(root/'split_screening.json')['proposed_locations']),'Midpoint locations retained in live paths: '+str(mids),'Fallback used: '+str(selection['fallback_to_incumbent']), 'Stopping reason: '+selection['reason']]
    for name,p in pol.items():lines.append(name+': steps='+str(p['resampling_steps'])+'; temperatures='+str(p.get('temperature_by_step',p.get('tempering')))+'; meanIR='+str(float(vectors[name].mean())))
    lines+=['\nPaired comparisons:\n```json\n'+json.dumps(comparisons,indent=2)+'\n```','Success criteria:\n```json\n'+json.dumps(success,indent=2)+'\n```','FINAL INTERPRETATION: '+('New candidate identified; assess the independent-within-campaign audit above.' if selection['selected_by_new_graph'] else 'Underidentified within the preregistered allocation procedure; retain Transfer. A proposed-policy audit is reported separately and cannot turn fallback into adaptive discovery.'),'Initial cap winner mass: '+str(selection['initial_candidate_mass'])+'; under90 coverage must not be described as all plausible winners.','Compiler and audit prompts are disjoint in this campaign, but all60 TRAIN prompts were historically explored. No TEST/generalization claim. Endpoint effects remain execution-protocol dependent. Ordinary repeated bootstrap intervals are allocation heuristics, not sequential confidence guarantees. Fine ends at90; adaptive permits95, so grids are not nested.','Where budget was spent: see sample_allocation.csv and racing_budget.csv; all local probes and unsuccessful candidates included. Shared-edge cancellation is preserved in paired candidate differences. Difficult comparisons and contributing SEMs are recorded at each look in racing_rounds.jsonl. Active candidate optimization is not global expanded-DAG optimization.','Ablations: historical fine/fixed are executed old evidence; sequential-all-expansion is a cost-only counterfactual at the same live nodes. It cannot establish prospective separate causal effects of sequential allocation versus lazy profiling.','Cost ledgers include reference regeneration, failed recorded work, VAE/reward replay overhead; audit and correctness separate. No actual standalone fine runtime comparator, hence no runtime speedup claim.']
    (root/'REPORT.md').write_text('\n\n'.join(lines));(root/'CHECKLIST.md').write_text('\n'.join('- [x] '+s for s in ['gates passed','TRAIN-only frozen split/algorithm/seeds','compilation cap verified','compiler decisions frozen before audit','40prompt paired inference 50k bootstrap','fallback not counted as new discovery','historical versus counterfactual costs labeled','all required artifacts generated','production unchanged']))

def run(root):
    root.mkdir(parents=True,exist_ok=True);lock=(root/'driver.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'status.json').exists() and read(root/'status.json').get('stage')=='DONE':return
    if not (root/'driver.pid.json').exists():dump(root/'driver.pid.json',{'pid':os.getpid(),'started_unix':time.time()})
    try:
        freeze(root);stage(root,'gates');assert read(root/'gates.json')['passed'];shutil.copy2(root/'gates.json',root/'smoke_gates.json')
        assert profile(root,'coarse5',jobs_for(G0),5);candidates,mass=shortlist(root);new=screening(root,candidates)
        leader,rounds=race(root,list({p['id']:p for p in candidates+new}.values()),mass)
        policies={'ADAPTIVE_NEW':leader,'P_COARSE':read(root/'coarse_policy.json'),'P_FINE':static_record(StaticBFSPolicy((50,60,80),((50,8.),(60,2.),(80,32.)),8)),'P_TRANSFER':records()['TRANSFER4_TO_8'],'P_MANUAL':records()['MANUAL8'],'DEPLOYED':read(root/'selection.json')['deployed_policy']}
        assert all(len(p['resampling_steps'])==3 for p in policies.values());dump(root/'frozen_policies.json',policies);dump(root/'policy_freeze_receipt.json',{'sha256':sha(root/'frozen_policies.json'),'before_audit':True,'timestamp':time.time()})
        stage(root,'audit');finalize(root,rounds);verify(root)
        dump(root/'status.json',{'stage':'DONE','all_stages_complete':True,'end_unix':time.time()})
    except BaseException as e:
        dump(root/'status.json',{'stage':'ABORTED','exception':repr(e),'traceback':traceback.format_exc(),'end_unix':time.time()});raise
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args();run(args.run_root.resolve())
