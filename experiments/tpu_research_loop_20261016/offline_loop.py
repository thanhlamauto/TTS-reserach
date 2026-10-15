"""Read-only historical diagnostics + CPU synthetic checks. Never launches models."""
from pathlib import Path
import argparse,ast,collections,csv,hashlib,json,math,statistics,time
import numpy as np
HERE=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):
 p=HERE/n;p.parent.mkdir(parents=True,exist_ok=True)
 raw=json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n'
 if p.exists():assert p.read_text()==raw,f'immutable output differs: {p}; use a new round'
 else:p.write_text(raw)
def csvout(n,rows):
 p=HERE/n;p.parent.mkdir(parents=True,exist_ok=True)
 import io
 s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 if p.exists():assert p.read_text()==s.getvalue().replace('\r\n','\n')
 else:p.write_text(s.getvalue())
def append_event(x):
 p=HERE/'cost_ledger.jsonl';old=[json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
 if any(a['event_id']==x['event_id'] for a in old):return
 with p.open('a') as f:f.write(json.dumps(x,sort_keys=True)+'\n')
def canonical(p):
 steps=p['resampling_steps'];temps=[10*((1+.024)**i-1) for i in steps] if p['kind']=='manual' else [dict(p['temperature_by_step'])[i] for i in steps]
 return {'particles':p['particles'],'steps':steps,'score_multipliers':temps,'scoring':'max_with_zero_initial_history','resampling':'SSP','selection':'all_endpoints_then_first_argmax','DDIM':100,'eta':1,'precision':'BF16','event_index':'zero-based-after-denoise'}
def block_variance(d):
 P,S=d.shape;W=float(np.var(d,axis=1,ddof=1).mean());Braw=float(np.var(d.mean(1),ddof=1)-W/S)
 return {'P':P,'S':S,'within_prompt_seed_variance_W':W,'between_prompt_variance_B_untruncated':Braw,'between_prompt_variance_B_forecast':max(0.,Braw),'boundary_rule':'retain raw negative estimate; truncate to0 only for nonnegative planning forecast','observed_SE':float(d.mean(1).std(ddof=1)/np.sqrt(P))}
def history(work,cfg):
 root=work/'artifacts/static-bfs-graph/policy_verification_compiler_v1';rows=[json.loads(s) for s in (root/'audit_records.jsonl').read_text().splitlines()]
 roles=read(root/'frozen_audit_policies.json');unique={p['id']:p for p in roles.values()};prompts=sorted({r['prompt'] for r in rows});seeds=sorted({r['seed'] for r in rows})
 assert len(rows)==576 and len(unique)==6 and len(prompts)==24 and len(seeds)==4
 by={(r['prompt'],r['seed'],r['policy']['id']):r for r in rows};assert len(by)==len(rows)
 for p in prompts:
  for s in seeds:
   block=[by[p,s,i] for i in unique];assert len({r['initial_latents_sha256'] for r in block})==1
   for r in block:
    assert r['NFE']==800 and r['verifier_calls']==4 and r['verifier_scores']==32 and math.isfinite(r['reward'])
    assert len(r['events'])==3 and [e['sampling_index'] for e in r['events']]==r['policy']['resampling_steps']
    assert r['reward']==max(r['scores'])
 matrices={role:np.array([[by[p,s,rec['id']]['reward'] for s in seeds] for p in prompts]) for role,rec in roles.items()}
 input_manifest={'scope':'historical DEVELOPMENT only, not new repeats or confirmation','inputs':{str(p.relative_to(work)):{'sha256':h(p),'bytes':p.stat().st_size} for p in [root/'audit_records.jsonl',root/'frozen_audit_policies.json',root/'data_split.json',root/'REPORT.md',work/'experiments/pointmaze_ultra_density_pilot_v1/REVIEW_SUMMARY.md']},'unique_rollouts':576,'prompts':prompts,'seeds':seeds,'role_count':len(roles),'unique_policies':len(unique)}
 write('evidence/input_manifest.json',input_manifest)
 canon={role:canonical(p) for role,p in roles.items()}
 candidate={}
 for role,p in roles.items():
  c=canon[role];ident=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
  if ident not in candidate:candidate[ident]={'semantic_sha256':ident,'semantics':c,'roles':[],'source_policy_ids':[]}
  candidate[ident]['roles'].append(role)
  if p['id'] not in candidate[ident]['source_policy_ids']:candidate[ident]['source_policy_ids'].append(p['id'])
 write('immutable_policy_hashes.json',{'candidates':list(candidate.values()),'node50_present':True,'exact_Manual_in_class':True,'exact_Fine_in_class':True,'notes':'Freeze this covered finite bank for future graph/direct/verification comparisons; no class-global optimum claim. Historical Fine+Manual were audit-only: future baseline access is a new development design.'})
 rng=np.random.default_rng(2026101631);ix=rng.integers(0,len(prompts),(cfg['bootstrap_replicates'],len(prompts)))
 stat=[];plans=[]
 for a,b in cfg['historical_contrasts']:
  A,B=matrices[a],matrices[b];d=A-B;v=block_variance(d);means=d.mean(1);lo,hi=np.quantile(means[ix].mean(1),[.025,.975]);cov=float(np.mean([np.cov(x,y,ddof=1)[0,1] for x,y in zip(A,B)]))
  sd=float(means.std(ddof=1));z=statistics.NormalDist().inv_cdf(1-cfg['alpha_family']/(2*len(cfg['historical_contrasts'])));zp=statistics.NormalDist().inv_cdf(.8)
  n=math.ceil(((z+zp)*sd/cfg['epsilon_IR'])**2)
  row={'A':a,'B':b,**v,'mean_delta':float(means.mean()),'descriptive_CI95_low':float(lo),'descriptive_CI95_high':float(hi),'within_prompt_policy_covariance':cov,'within_prompt_unpaired_variance':float(np.var(A,axis=1,ddof=1).mean()+np.var(B,axis=1,ddof=1).mean()),'planned_epsilon':cfg['epsilon_IR'],'approx_prompts_80pct_power_alpha_family_0p05_three_contrasts':n,'precision_note':'normal approximation from historical 24-prompt pilot, not a guarantee or IR tail bound'}
  stat.append(row)
  for trajectories in [24,48,96,192]:
   for S in [1,2,4]:
    P=trajectories//S;var=v['between_prompt_variance_B_forecast']/P+v['within_prompt_seed_variance_W']/(P*S)
    plans.append({'A':a,'B':b,'paired_trajectories':trajectories,'prompts':P,'seeds_per_prompt':S,'full_policy_rollouts_two_policies':trajectories*2,'NFE_two_policies':trajectories*1600,'forecast_SE':math.sqrt(var),'forecast_MDE_80pct_power':(z+zp)*math.sqrt(var),'approx_CI95_halfwidth':1.96*math.sqrt(var)})
 csvout('variance_components.csv',stat);csvout('precision_plan.csv',plans)
 write('split_manifest.json',{'DEVELOPMENT':{'historic_policy_audit':{'prompts':prompts,'seeds':seeds,'source_sha256':h(root/'audit_records.jsonl')},'all_old_graphs_and_reports':'previously inspected TRAIN; fresh seeds do not make prompts unseen'},'SELECTION':{'status':'NOT_OPENED','requires':'separate fresh prompt inventory and frozen sampling distribution; no reuse of history as independent selection'},'CONFIRMATION':{'status':'NOT_OPENED','budget_reserved_fraction':.4,'requires':'one fixed-sample opening after selector and candidate rule freeze; independent compilation replicates when feasible'},'unused_reserved_128_prefix_prompts':'not read, not used','new_observations':0})
 # Deduplicate by content, not number of copies. Path names are provenance only.
 hashes=collections.defaultdict(list)
 for p in list((work/'artifacts/static-bfs-graph').glob('*/REPORT.md'))+[work/'experiments/pointmaze_ultra_density_pilot_v1/REPORT.md',work/'experiments/pointmaze_ultra_density_pilot_v1/remote_artifacts/pointmaze_ultra_density_pilot_v1/REPORT.md']:
  if p.exists():hashes[h(p)].append(str(p.relative_to(work)))
 write('evidence/report_deduplication.json',{'content_hash_groups':dict(hashes),'independent_compilation_replicates_in_this_campaign':0,'warning':'different reports sharing source run/data are also not independent even when bytes differ'})
 return stat

def geometry(work,cfg):
 old=work/'experiments/pointmaze_ultra_density_pilot_v1/remote_artifacts/pointmaze_ultra_density_pilot_v1'
 asset=read(old/'asset_status.json');maze=np.array(asset['maze_map']);cells=set(map(tuple,np.argwhere(maze==0)))
 adj={a:[b for b in [(a[0]+1,a[1]),(a[0]-1,a[1]),(a[0],a[1]+1),(a[0],a[1]-1)] if b in cells] for a in cells}
 # Tarjan articulation points (geometry bottleneck proxy).
 disc={};low={};parent={};art=set();tick=[0]
 def dfs(u):
  tick[0]+=1;disc[u]=low[u]=tick[0];children=0
  for v in adj[u]:
   if v not in disc:
    parent[v]=u;children+=1;dfs(v);low[u]=min(low[u],low[v])
    if u not in parent and children>1:art.add(u)
    if u in parent and low[v]>=disc[u]:art.add(u)
   elif parent.get(u)!=v:low[u]=min(low[u],disc[v])
 for a in sorted(cells):
  if a not in disc:dfs(a)
 oldrows=list(csv.DictReader((old/'geometry_inventory.csv').open()));assert len(oldrows)==len(cells)*(len(cells)-1)//2
 audited=[]
 for row in oldrows:
  a,b=tuple(ast.literal_eval(row['start_ij'])),tuple(ast.literal_eval(row['goal_ij']));dist={a:0};q=[a]
  for x in q:
   if x==b:break
   for y in adj[x]:
    if y not in dist:dist[y]=dist[x]+1;q.append(y)
  assert dist[b]==int(row['shortest_grid_distance'])
  # Expected articulation points over uniform shortest paths, via dynamic programming.
  count={a:1};bottle={a:float(a in art)}
  for x in q[1:]:
   preds=[y for y in adj[x] if dist.get(y)==dist[x]-1];n=sum(count[y] for y in preds)
   count[x]=n;bottle[x]=float(x in art)+sum(count[y]*bottle[y] for y in preds)/n
  d=dist[b]
  audited.append({**row,'expected_articulation_points':bottle[b],'geodesic_world_distance':d*4.,'world_length_per_sampling_horizon':d*4/2800,'world_length_per_training_horizon':d*4/2000,'reachability':'GRID_CONNECTED_ONLY; dynamical/controller feasibility not certified','training_coverage':'UNVERIFIED: exact official trajectory distribution not available locally'})
 csvout('ultra_geometry_audit.csv',audited)
 lengths=np.array([int(r['shortest_grid_distance']) for r in audited]);qs=np.quantile(lengths,[0,.25,.5,.75,1]);all_old=read(old/'frozen_task_manifest.json')['cases'];used={tuple(sorted((tuple(c['start_ij']),tuple(c['goal_ij'])))) for c in all_old}
 for t in asset['task_infos']:used.add(tuple(sorted((tuple(t['init_ij']),tuple(t['goal_ij'])))))
 proposals=[]
 for label,lower,upper in [('MIDDLE_LOW',qs[1],qs[2]),('MIDDLE_HIGH',qs[2],qs[3])]:
  available=[r for r in audited if lower<=int(r['shortest_grid_distance'])<upper and tuple(sorted((tuple(ast.literal_eval(r['start_ij'])),tuple(ast.literal_eval(r['goal_ij']))))) not in used]
  by=collections.defaultdict(list)
  for r in available:by[int(r['shortest_grid_distance'])].append(r)
  target=(lower+upper)/2
  eligible=[]
  for d,rr in by.items():
   lo,hi=np.quantile([float(r['detour_ratio']) for r in rr],[1/3,2/3])
   lowr=[r for r in rr if float(r['detour_ratio'])<=lo];highr=[r for r in rr if float(r['detour_ratio'])>=hi]
   if hi>lo and len(lowr)>=2 and len(highr)>=2:eligible.append((abs(d-target),d,lo,hi,lowr,highr))
  assert eligible
  _,distance,lo,hi,lr,hr=min(eligible,key=lambda x:(x[0],x[1]))
  for tier,rr in [('LOW_DETOUR',lr),('HIGH_DETOUR',hr)]:
   ranked=sorted(rr,key=lambda r:hashlib.sha256(json.dumps([r['start_ij'],r['goal_ij'],20261016]).encode()).hexdigest())
   for r in ranked[:cfg['geometry_dev_pairs_per_bin_tier']]:
    a,b=ast.literal_eval(r['start_ij']),ast.literal_eval(r['goal_ij']);k=len(proposals);info={'task_name':f'ultra_dev20261016_{k:02d}','init_ij':a,'goal_ij':b,'init_xy':[4*a[1]-4,4*a[0]-4],'goal_xy':[4*b[1]-4,4*b[0]-4]}
    proposals.append({'id':f'ultra_dev{k:02d}','task_info':info,'distance_bin':label,'detour_tier':tier,'shortest_grid_distance':distance,'detour_ratio':float(r['detour_ratio']),'expected_turns':float(r['expected_shortest_path_turns']),'expected_bottlenecks':float(r['expected_articulation_points']),'degree2_corridor_fraction':float(r['degree2_corridor_fraction']),'frozen_bin_detour_thresholds':[float(lo),float(hi)]})
 assert len(proposals)==8 and len({tuple(sorted((tuple(c['task_info']['init_ij']),tuple(c['task_info']['goal_ij'])))) for c in proposals})==8
 policies=read(old/'frozen_policy_manifest.json')
 write('ultra_development_manifest.json',{'status':'GEOMETRY_ONLY_FROZEN_NO_INFERENCE','cases':proposals,'seeds':cfg['geometry_dev_seeds'],'policies':policies,'source_policy_manifest_sha256':h(old/'frozen_policy_manifest.json'),'inference_steps':16,'N':16,'horizon':2800,'recur_steps':6,'source_training_horizon':2000,'checkpoint_archive_sha256':'c7f20e023444e1d417be33e854cb6e53b0d0428a712dfb2b034731e6b3ed24b6','all_outcomes_unopened':True,'DEVELOPMENT_only':True,'old_pairs_and_official_task_pairs_excluded':True,'one_map_only':True,'requires_TPU_port_and_budget':True,'success_primary':'environment success','success_20_80pct':'heuristic only; policy discordance/precision gate also required'})
 write('geometry_summary.json',{'free_cells':len(cells),'pairs_verified':len(audited),'geodesic_quantiles_0_25_50_75_100':qs.tolist(),'articulation_cells':len(art),'narrowness_proxy':'fraction of degree2 cells is a topology proxy, not physical clearance; Ultra wall expansion .7 must be retained in collision checks','world_length_horizon_ratio':'unitful geometric load proxy, not reachability or calibrated training coverage','old_tasks':len(all_old),'new_development_tasks':len(proposals),'new_grid_distances':sorted(set(c['shortest_grid_distance'] for c in proposals)),'old_source_sha256':h(old/'geometry_inventory.csv')})
 # Revalidate old Ultra handoff from original rows, including paired outcome identity.
 rr=[json.loads(s) for s in (old/'raw_policy_runs.jsonl').read_text().splitlines()];assert len(rr)==96
 counts=collections.Counter(r['policy_name'] for r in rr if r['success']);assert counts=={k:1 for k in policies}
 outcomes={k:{(r['case_id'],r['seed']) for r in rr if r['policy_name']==k and r['success']} for k in policies};assert len({tuple(sorted(v)) for v in outcomes.values()})==1
 write('evidence/ultra_raw_audit.json',{'passed':True,'rows':96,'successes':dict(counts),'same_success_case_seed':True,'raw_sha256':h(old/'raw_policy_runs.jsonl'),'interpretation':'floor effect; identical observed differences are not population equivalence'})

def synthetic(cfg):
 rng=np.random.default_rng(2026101633);tests=[]
 def softlog(x):
  x=x-x.max();return x-np.log(np.exp(x).sum())
 for _ in range(1000):
  r=rng.normal(size=8);eta=float(rng.uniform(.001,.5));error=rng.uniform(-eta,eta,8);beta=float(rng.choice([0,1,8,32]));ratio=softlog(beta*(r+error))-softlog(beta*r)
  assert np.max(np.abs(ratio))<=2*beta*eta+1e-10
 tests.append({'name':'temperature_ratio_bound','cases':1000,'passed':True,'terminal_reward_gain_implied':False})
 for _ in range(1000):
  J=rng.normal(size=7);C=np.arange(5);em=float(rng.uniform(0,.2));ee=float(rng.uniform(0,.2));g=J+rng.uniform(-em,em,7);gh=g+rng.uniform(-ee,ee,7);pick=int(C[np.argmax(gh[C])]);gap=J.max()-J[C].max()
  assert J.max()-J[pick]<=gap+2*em+2*ee+1e-12
 tests.append({'name':'surrogate_regret_decomposition','cases':1000,'passed':True})
 scenarios=[];K=4;alpha=.05;epsilon=.02;looks=[16,64,256];M=K*(K-1)//2;z=statistics.NormalDist().inv_cdf(1-alpha/(2*M*len(looks)));reps=500
 for label,base,interaction,sigma,omitted in [('additive_high_noise',np.array([.4,.42,.44,.46]),np.zeros(4),.8,None),('interaction',np.array([.4,.42,.44,.46]),np.array([.3,0,0,0]),.08,None),('missing_optimum',np.array([.4,.42,.44,.46]),np.zeros(4),.08,.9)]:
  J=base+interaction;false_cert=0;certs=0;graph_regrets=[];direct_regrets=[];residuals=[]
  # Shared Gaussian block factor makes arm observations dependent; exact paired variance known here only.
  for _ in range(reps):
   common=rng.normal(0,sigma*.5,(max(looks),1));idnoise=rng.normal(0,sigma,(max(looks),K));X=J+common+idnoise;graph_regrets.append(float(J.max()-J[np.argmax(base)]))
   picked=None;cert=False
   for n in looks:
    means=X[:n].mean(0);picked=int(np.argmax(means));radius=z*math.sqrt(2)*sigma/math.sqrt(n)
    if max(means[j]-means[picked]+radius for j in range(K) if j!=picked)<=epsilon:cert=True;break
   certs+=cert;false_cert+=cert and J.max()-J[picked]>epsilon;direct_regrets.append(float(J.max()-J[picked]));residuals.append((X[-64:]-base).mean(0).tolist())
  assert false_cert/reps<=alpha+3*math.sqrt(alpha*(1-alpha)/reps)
  res=np.mean(residuals,axis=0);observed_class='COMPOSITION_PRESENT' if np.max(abs(res))>.1 else 'NO_STRUCTURAL_INTERACTION_DETECTED'
  if omitted is not None:observed_class='COVERAGE_FAILURE_KNOWN_ONLY_BECAUSE_SYNTHETIC_TRUTH_AVAILABLE'
  scenarios.append({'scenario':label,'repetitions':reps,'true_model_residual':interaction.tolist(),'estimated_model_residual':res.tolist(),'diagnostic':observed_class,'certificate_rate':certs/reps,'false_epsilon_certificate_rate':false_cert/reps,'mean_graph_within_class_regret':float(np.mean(graph_regrets)),'mean_direct_within_class_regret':float(np.mean(direct_regrets)),'known_synthetic_coverage_gap':None if omitted is None else float(omitted-J.max()),'guarantee_scope':'finite-look Gaussian paired known-variance toy only; not ImageReward; shared block correlation retained'})
 assert scenarios[0]['certificate_rate']<.1 and scenarios[1]['diagnostic']=='COMPOSITION_PRESENT' and scenarios[2]['known_synthetic_coverage_gap']>0
 write('synthetic_tests.json',{'passed':True,'algebra_tests':tests,'scenarios':scenarios,'no_diffusion_claim':True,'no_new_method':True})

def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();t=time.monotonic();cfg=read(HERE/'preregistered_config.json')
 stats=history(a.workspace.resolve(),cfg);geometry(a.workspace.resolve(),cfg);synthetic(cfg)
 write('diagnostic_decision.json',{'status':['BUDGET_UNVERIFIED','UNDERIDENTIFIED'],'diagnostic':'paired prompt/seed variance and precision planning','contrasts':stats,'next_step':'TPU-only ImageReward gate, then one small multiseed DEVELOPMENT pilot if budget authorized; no interaction objective until reproducible residual evidence','direct_search_required':'exact shared candidate bank and total all-in profiling cost; NOT RUN, no equal-budget winner claim','final_confirmation_opened':False,'new_compilation_replicates':0})
 append_event({'event_id':'offline-diagnostic-1','execution':'local CPU existing raw analysis + synthetic tests','elapsed_CPU_wall_seconds':time.monotonic()-t,'TPU_chip_hours':0,'GPU_hours':0,'neural_model_forwards':0,'new_diffusion_rollouts':0,'new_verifier_evaluations':0,'billing_USD':'no provider compute launched; existing READY slice may accrue separately'})
 print(json.dumps({'stage':'OFFLINE_DIAGNOSTIC_DONE','status':'BUDGET_UNVERIFIED','new_neural_compute':0,'outputs':str(HERE),'variance_diagnostic':[{k:s[k] for k in ['A','B','between_prompt_variance_B_untruncated','within_prompt_seed_variance_W','approx_prompts_80pct_power_alpha_family_0p05_three_contrasts']} for s in stats]},indent=2))
if __name__=='__main__':main()
