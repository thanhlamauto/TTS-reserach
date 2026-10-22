"""Offline-only forensics. Imports no diffusion, Torch, or reward implementation."""
import argparse,json,csv,hashlib,itertools,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def table(p,rows,fields=None):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fields or list(rows[0]));w.writeheader();w.writerows(rows)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(j):return (j['src'],j['dst'],j['tau'],j.get('mid'),j.get('mtau'))
def edges(p):
 if p['kind']!='static':return None
 prev=-1;out=[]
 for s,t in p['temperature_by_step']:out.append((prev,s,t,None,None));prev=s
 return out+[(prev,100,None,None,None)]
def policy(steps,taus):
 # Retain exact historical identities: policy.id uses canonical policy record.
 items=list(zip(steps,taus));rec={'resampling_steps':list(steps),'temperature_by_step':[[s,float(t)] for s,t in items],'particles':8,'selection_mode':'raw_tau'}
 from_record=json.dumps(rec,sort_keys=True,separators=(',',':'))
 return rec

def boot(x,seed,reps=50000):
 rng=np.random.default_rng(seed);idx=rng.integers(0,len(x),(reps,len(x)));m=x[idx].mean(axis=1)
 return {'delta':float(x.mean()),'SE':float(x.std(ddof=1)/np.sqrt(len(x))),'CI_low':float(np.quantile(m,.025)),'CI_high':float(np.quantile(m,.975)),'one_sided95_lower':float(np.quantile(m,.05)),'wins':int(sum(x>0)),'ties':int(sum(x==0)),'losses':int(sum(x<0))}
def joint(x,seed,reps=20000):
 idx=np.random.default_rng(seed).integers(0,len(x),(reps,len(x)));b=x[idx].mean(axis=1);prob=np.bincount(np.argmin(b,axis=1),minlength=x.shape[1])/reps
 return b,prob

def run(src,out,hist,retuned):
 out.mkdir(parents=True,exist_ok=True)
 names=['config.json','preregistration.json','dataset_split.json','coarse_graph.json','coarse_edge_samples.csv','candidate_paths.json','midpoint_proposals.csv','racing_rounds.jsonl','racing_survivors.csv','racing_budget.csv','sample_allocation.csv','lazy_profiled_edges.csv','final_candidate_graph.json','frozen_policies.json','audit_raw.jsonl','audit_summary.csv','paired_comparisons.csv','cost_accounting.json','REPORT.md']
 inventory={'latest':{n:{'exists':(src/n).exists(),'path':str(src/n),'sha256':digest(src/n) if (src/n).exists() else None} for n in names},'raw_edges':{'path':str(out/'observations/A'),'records':len(list((out/'observations/A').glob('*/*.json')))},'historical_adaptive':{'path':str(hist),'fine_graph':(hist/'fine_graph.json').exists(),'audit_per_prompt':(hist/'audit_per_prompt.csv').exists()},'historical_n8_retuning':{'report':str(retuned),'report_exists':retuned.exists(),'raw_records_available':False,'limitation':'report only; raw48prompt paired rewards not locally available'},'runtime':'local CPU; no inference; no validation/test access'};write(out/'source_inventory.json',inventory)
 for n in names:assert (src/n).exists(),n
 cfg=read(src/'config.json');split=read(src/'dataset_split.json');pol=read(src/'frozen_policies.json');logs=[json.loads(l) for l in (src/'racing_rounds.jsonl').read_text().splitlines()];select=read(src/'selection.json');ids=split['A'];seeds=cfg['seeds']['C'];audit=[json.loads(l) for l in (src/'audit_raw.jsonl').read_text().splitlines()];raw=[read(p) for p in (out/'observations/A').glob('*/*.json')]
 obs={}
 for r in raw:
  k=(r['prompt'],r['seed'],key(r['job']));assert k not in obs;obs[k]=r['regret'];assert np.isfinite(r['regret']) and abs(r['regret']-(r['reference_reward']-r['reward']))<1e-12
  assert r['seed']==cfg['seeds']['A'][0] and r['prompt'] in ids
  assert r['reference_sha256']==digest(src/'reference_schedule.json')
 assert len(raw)==675 and len(audit)==400
 lookup={(r['prompt'],r['seed'],r['policy']['id']):r for r in audit};assert len(lookup)==400
 initial_hashes={}
 for r in audit:
  assert abs(r['reward']-max(r['scores']))<1e-12 and len(r['events'])==3
  u=(r['prompt'],r['seed']);initial_hashes.setdefault(u,set()).add(r['initial_latents_sha256'])
 assert len(initial_hashes)==80 and all(len(v)==1 for v in initial_hashes.values())
 R={name:np.array([np.mean([lookup[(pid,s,p['id'])]['reward'] for s in seeds]) for pid in split['C']]) for name,p in pol.items()}
 expected={'ADAPTIVE_NEW':.9659451401326805,'P_COARSE':.9924954587826506,'P_FINE':1.0078347500413656,'P_MANUAL':1.0068782581249252,'P_TRANSFER':.9491690368857235}
 assert all(abs(R[k].mean()-v)<1e-12 for k,v in expected.items())
 finegraph=read(hist/'fine_graph.json');assert finegraph['logical_edges']==145 and len(finegraph['edges'])==145 and finegraph['edge_observations']==2900
 assert select['reason']=='UNDERIDENTIFIED' and select['fallback_to_incumbent'] and not select['selected_by_new_graph']
 assert pol['ADAPTIVE_NEW']['resampling_steps']==[40,80,90] and pol['ADAPTIVE_NEW']['temperature_by_step']==[[40,8.],[80,32.],[90,32.]]
 assert set(split['A']).isdisjoint(split['C']) and len(split['C'])==40
 write(out/'reproduction_check.json',{'passed':True,'observations':len(raw),'baseline_edges':145,'historical_fine_baseline':145*20,'ratio':len(raw)/2900,'means':{k:float(R[k].mean()) for k in expected},'prompt_count':40,'paired_seed_count':2,'audit_unique_rows':400,'identical_initial_particles_all80units':True,'stop':select['reason'],'fallback':select['deployed_policy'],'reward_aggregation':'max8 endpoint particles, mean2seeds withinprompt then mean40prompts'})
 def vector(p,n):
  ee=edges(p)
  if ee is None:return None
  try:return np.array([sum(obs[(pid,cfg['seeds']['A'][0],e)] for e in ee) for pid in ids[:n]])
  except KeyError:return None
 def available_n(p):return max([n for n in [5,10,15,20] if vector(p,n) is not None] or [0])
 initial=read(src/'coarse_candidates.json');proposal=read(src/'path_proposals.json');allc=sorted({p['id']:p for p in initial+proposal}.values(),key=lambda p:p['id']);pmap={p['id']:p for p in allc};live=allc;roundrows=[];history=[];replay=[]
 for q in logs:
  n=q['n'];x=np.stack([vector(p,n) for p in live],axis=1);b,prob=joint(x,cfg['stat_seed']+n);order=np.argsort(x.mean(axis=0),kind='stable');leader=live[int(order[0])];assert leader['id']==q['leader']['id'];assert len(live)==len(q['details'])
  recorded={d['policy_id']:d for d in q['details']};survivors=[]
  for i,p in enumerate(live):
   d=x[:,i]-x[:,order[0]];worse=float(np.mean(b[:,i]-b[:,order[0]]>.01));drop=bool(i!=order[0] and p['id']!=pol['P_TRANSFER']['id'] and worse>=.95 and prob[i]<.01)
   z=recorded[p['id']];assert abs(worse-z['P_worse_by01'])<1e-12 and abs(prob[i]-z['winner_probability'])<1e-12 and drop==z['eliminated'];assert abs(d.std(ddof=1)/np.sqrt(n)-z['paired_SE'])<1e-12
   # Verify covariance and exact common-edge cancellation explicitly.
   e1=set(edges(p));e2=set(edges(leader));cancel=np.array([sum(obs[(pid,cfg['seeds']['A'][0],e)] for e in e1-e2)-sum(obs[(pid,cfg['seeds']['A'][0],e)] for e in e2-e1) for pid in ids[:n]]);assert np.allclose(cancel,d,atol=1e-12)
   roundrows.append({'n':n,'policy_id':p['id'],'steps':str(p['resampling_steps']),'tau':str(p['temperature_by_step']),'candidate_count_before':len(live),'candidate_count_after':q['live_candidates'],'cost':float(x[:,i].mean()),'cost_SE':float(x[:,i].std(ddof=1)/np.sqrt(n)),'leader_id':leader['id'],'paired_gap':float(d.mean()),'paired_SE':float(d.std(ddof=1)/np.sqrt(n)),'winner_frequency':float(prob[i]),'P_gap_gt01':worse,'eliminated':drop,'incumbent_rank':q['incumbent_rank'],'cumulative_observations':q['observations'],'shared_edge_count':len(e1&e2),'common_n':n,'stable':q['stable']})
   history.append({'n':n,'policy_id':p['id'],'before':True,'after':not drop,'eliminated':drop})
   if not drop:survivors.append(p)
  assert len(survivors)==q['live_candidates'];replay.append({'n':n,'exact_decisions_match':True});live=survivors
 table(out/'round_reconstruction.csv',roundrows);table(out/'candidate_stage_history.csv',history);write(out/'round_replay_check.json',replay)
 active={p['id'] for p in live};proposed={p['id'] for p in allc};initialset={p['id'] for p in initial};elim={r['policy_id']:r['n'] for r in history if r['eliminated']};G0=set(read(src/'coarse_graph.json')['nodes']);expanded=G0|{s for p in allc for s in p['resampling_steps']}
 coverage=[]
 for name,p in pol.items():
  steps=set(p['resampling_steps']);temps=([e['temperature'] for e in lookup[(split['C'][0],seeds[0],p['id'])]['events']] if p['kind']=='manual' else [t for s,t in p['temperature_by_step']]);action=all(t in [2,8,32] for t in temps);represent=steps<=G0 and action;rep_exp=steps<=expanded and action
  status='OUT_OF_GRID' if not steps<=expanded else 'OUT_OF_ACTION_GRID' if not action else 'ACTIVE_AT_TERMINATION' if p['id'] in active else 'PROPOSED_AND_ELIMINATED' if p['id'] in elim else 'PROPOSED_NOT_FULLY_PROFILED' if p['id'] in proposed else 'REPRESENTABLE_NOT_PROPOSED'
  coverage.append({'policy':name,'id':p['id'],'coarse_support':'EXACTLY_REPRESENTABLE' if represent else 'OUT_OF_GRID' if not steps<=G0 else 'OUT_OF_ACTION_GRID','expanded_support':'EXACTLY_REPRESENTABLE' if rep_exp else 'OUT_OF_GRID' if not steps<=expanded else 'OUT_OF_ACTION_GRID','initially_proposed':p['id'] in initialset,'ever_proposed':p['id'] in proposed,'fully_observed_n':available_n(p),'eliminated_n':elim.get(p['id'],''),'alive_at_end':p['id'] in active,'never_considered':p['id'] not in proposed,'classification':status,'actual_temperatures':str(temps)})
 table(out/'policy_coverage.csv',coverage)
 write(out/'candidate_coverage_summary.json',{'initial_count_including_transfer':len(initial),'initial_winner_mass':read(src/'coarse_graph.json')['candidate_mass'],'Transfer_present':pol['P_TRANSFER']['id'] in initialset,'distinct_timesteps':sorted({s for p in initial for s in p['resampling_steps']}),'distinct_edges':len({e for p in initial for e in edges(p)}),'expanded_nodes':sorted(expanded),'mass_interpretation':'conditional fitted-graph bootstrap; not true reward-optimal coverage'})
 stability=[];optimism=[];loo=[];sensitivity=[]
 for label,group,n in [('initial_all_proposals',allc,5),('fully_observed20',[p for p in allc if available_n(p)==20],20)]:
  group=sorted(group,key=lambda p:p['id']);x=np.stack([vector(p,n) for p in group],axis=1);b,prob=joint(x,2026100901+n);ranks=np.argsort(np.argsort(b,axis=1,kind='stable'),axis=1,kind='stable')+1
  for i,p in enumerate(group):stability.append({'cohort':label,'n':n,'candidate_count':len(group),'policy_id':p['id'],'cost':float(x[:,i].mean()),'cost_SE':float(x[:,i].std(ddof=1)/np.sqrt(n)),'winner_probability':float(prob[i]),'rank_median':float(np.median(ranks[:,i])),'rank_q025':float(np.quantile(ranks[:,i],.025)),'rank_q975':float(np.quantile(ranks[:,i],.975))})
  loo_means=np.stack([np.delete(x,h,axis=0).mean(axis=0) for h in range(n)]);loo_ranks=np.argsort(np.argsort(loo_means,axis=1,kind='stable'),axis=1,kind='stable')+1
  for i,p in enumerate(group):sensitivity.append({'cohort':label,'policy_id':p['id'],'n':n,'leave1_cost_min':float(loo_means[:,i].min()),'leave1_cost_max':float(loo_means[:,i].max()),'leave1_rank_min':int(loo_ranks[:,i].min()),'leave1_rank_max':int(loo_ranks[:,i].max()),'leave1_winner_count':int(sum(loo_ranks[:,i]==1))})
  for held in range(n):
   train=np.delete(x,held,axis=0);k=int(np.argmin(train.mean(axis=0)));gap=float(x[held,k]-train[:,k].mean());optimism.append({'cohort':label,'n':n,'held_prompt':ids[held],'selected_policy':group[k]['id'],'training_cost':float(train[:,k].mean()),'heldout_edge_cost':float(x[held,k]),'optimism_held_minus_train':gap,'interpretation':'graph-score optimism, not fullpolicy reward'})
   loo.append({'cohort':label,'held_prompt':ids[held],'leader':group[k]['id']})
 table(out/'bootstrap_path_stability.csv',stability);table(out/'selection_optimism.csv',optimism);table(out/'leave_one_out_leaders.csv',loo);table(out/'candidate_leave_one_out_sensitivity.csv',sensitivity)
 conditional=[]
 for label,group,n in [('initial_all_proposals',allc,5),('fully_observed20',[p for p in allc if available_n(p)==20],20)]:
  group=sorted(group,key=lambda p:p['id']);x=np.stack([vector(p,n) for p in group],axis=1);rng=np.random.default_rng(2026100990+n);gaps={p['id']:[] for p in group}
  for _ in range(2000):
   sampled=rng.integers(0,n,n);oob=np.array([i for i in range(n) if i not in set(sampled)])
   if not len(oob):continue
   k=int(np.argmin(x[sampled].mean(axis=0)));gaps[group[k]['id']].append(float(x[oob,k].mean()-x[sampled,k].mean()))
  for p in group:
   g=gaps[p['id']];conditional.append({'cohort':label,'n':n,'policy_id':p['id'],'selected_resamples':len(g),'conditional_graph_optimism':float(np.mean(g)) if g else None,'support':'DESCRIPTIVE' if len(g)>=20 else 'INSUFFICIENT_SELECTED_RESAMPLES','not_policy_reward':True})
 table(out/'candidate_selection_optimism.csv',conditional)
 finalx=np.stack([vector(p,20) for p in live],axis=1);finalb,finalprob=joint(finalx,cfg['stat_seed']+20);unresolved=[]
 for i,j in itertools.combinations(range(len(live)),2):
  d=finalx[:,i]-finalx[:,j];bd=finalb[:,i]-finalb[:,j];lo,hi=np.quantile(bd,[.025,.975]);unresolved.append({'A':live[i]['id'],'B':live[j]['id'],'n':20,'cost_gap':float(d.mean()),'paired_SE':float(d.std(ddof=1)/np.sqrt(20)),'CI_low':float(lo),'CI_high':float(hi),'unresolved_at_margin01':bool(lo<=.01 and hi>=-.01)})
 table(out/'final_candidate_paired_uncertainty.csv',unresolved)
 pairrows=[];uncertainty=[]
 for i,(a,bname) in enumerate([('ADAPTIVE_NEW','P_COARSE'),('ADAPTIVE_NEW','P_TRANSFER'),('P_COARSE','P_TRANSFER')]):
  n=min(available_n(pol[a]),available_n(pol[bname]));pred=-(vector(pol[a],n)-vector(pol[bname],n));cal=boot(pred,2026100920+i);actual=boot(R[a]-R[bname],2026100930+i)
  pairrows.append({'A':a,'B':bname,'common_calibration_n':n,'predicted_delta':cal['delta'],'predicted_CI_low':cal['CI_low'],'predicted_CI_high':cal['CI_high'],'predicted_SE':cal['SE'],'actual_delta':actual['delta'],'actual_CI_low':actual['CI_low'],'actual_CI_high':actual['CI_high'],'actual_SE':actual['SE'],'sign_agreement':bool(np.sign(cal['delta'])==np.sign(actual['delta'])),'not_same_trajectory_residual':True})
  e1=set(edges(pol[a]));e2=set(edges(pol[bname]));cols=[];signed=[]
  for e in sorted(e1^e2,key=str):
   sign=-1 if e in e1 else 1;v=np.array([obs[(pid,cfg['seeds']['A'][0],e)]*sign for pid in ids[:n]]);cols.append(v);signed.append(e)
  mx=np.stack(cols,axis=1);total=mx.sum(axis=1)
  for j,e in enumerate(signed):uncertainty.append({'A':a,'B':bname,'n':n,'edge':str(e),'edge_SE':float(mx[:,j].std(ddof=1)/np.sqrt(n)),'cov_contribution_to_mean_variance':float(np.cov(mx[:,j],total,ddof=1)[0,1]/n),'note':'contributions include covariance and can be negative'})
 table(out/'predicted_vs_actual.csv',pairrows);table(out/'edge_uncertainty_contributions.csv',uncertainty)
 fall=[];heter=[]
 for i,bname in enumerate(['P_MANUAL','P_COARSE','P_FINE']):
  d=R['P_TRANSFER']-R[bname];z=boot(d,2026100940+i);perseed=[]
  for seed in seeds:
   v=np.array([lookup[(pid,seed,pol['P_TRANSFER']['id'])]['reward']-lookup[(pid,seed,pol[bname]['id'])]['reward'] for pid in split['C']]);perseed.append(v)
  agreement=int(sum(np.sign(perseed[0])==np.sign(perseed[1])));largest=np.argsort(np.abs(d))[-3:];keep=np.delete(d,largest)
  fall.append({'A':'P_TRANSFER','B':bname,**z,'seed1_mean':float(perseed[0].mean()),'seed2_mean':float(perseed[1].mean()),'seed_sign_agree_prompts':agreement,'leave1_mean_min':float(min(np.delete(d,j).mean() for j in range(40))),'leave1_mean_max':float(max(np.delete(d,j).mean() for j in range(40))),'remove3_largest_abs_delta':float(keep.mean()),'removed3_prompts':str([split['C'][j] for j in largest])})
  for j,pid in enumerate(split['C']):heter.append({'B':bname,'prompt':pid,'mean_delta':float(d[j]),'seed1_delta':float(perseed[0][j]),'seed2_delta':float(perseed[1][j])})
 table(out/'fallback_paired_analysis.csv',fall);table(out/'fallback_prompt_differences.csv',heter)
 # Counterfactuals use ONLY measured calibration costs, never audit-based selection.
 ablations=[]
 def ablate(name,group,n,rule,lam=0.,ident='SUPPORTED_RESTRICTED_LOGGED_SET'):
  x=np.stack([vector(p,n) for p in group],axis=1);m=x.mean(axis=0);sem=x.std(axis=0,ddof=1)/np.sqrt(n);k=int(np.argmin(m+lam*sem));p=group[k];reward=next((float(R[nm].mean()) for nm,q in pol.items() if q['id']==p['id']),None)
  required=len({e for p in group for e in edges(p)})*n
  ablations.append({'arm':name,'rule':rule,'n':n,'candidate_count':len(group),'selected_id':p['id'],'selected_steps':str(p['resampling_steps']),'observations_required_for_edge_union':required,'observations_available_total':675,'underidentified':True,'offline_identifiability':ident,'hindsight_audit_IR':reward,'audit_not_used_to_select':True})
 # Exact original already reconstructed above.
 ablations.append({'arm':'A Original sequential+lazy','rule':'exact four-round replay; abstain toTransfer','n':20,'candidate_count':len(live),'selected_id':pol['P_TRANSFER']['id'],'selected_steps':str(pol['P_TRANSFER']['resampling_steps']),'observations_required_for_edge_union':675,'observations_available_total':675,'underidentified':True,'offline_identifiability':'EXACT_REPLAY','hindsight_audit_IR':float(R['P_TRANSFER'].mean()),'audit_not_used_to_select':True})
 supported=[];coarseall=[]
 for steps in itertools.combinations(sorted(G0),3):
  for taus in itertools.product([2.,8.,32.],repeat=3):
   p={'kind':'static','resampling_steps':list(steps),'temperature_by_step':[[s,t] for s,t in zip(steps,taus)]}
   matched=next((q for q in allc if q['resampling_steps']==p['resampling_steps'] and q['temperature_by_step']==p['temperature_by_step']),None);p['id']=matched['id'] if matched else 'static_'+hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!='kind'},sort_keys=True,separators=(',',':')).encode()).hexdigest()[:16]
   coarseall.append(p)
   if vector(p,20) is not None:supported.append(p)
 assert len(coarseall)==270
 ablate('B No cap, all270 at common5',coarseall,5,'unrestricted coarse rank at observed5; continuation unsupported')
 ablate('B Measured coarse subset20',supported,20,'rank only fully observed paths; not no-cap continuation')
 ablations.append({'arm':'B Full no-cap sequential continuation','rule':'missing backfill for unproposed paths','n':20,'candidate_count':270,'selected_id':'NOT_IDENTIFIABLE_FROM_LOGGED_DATA','selected_steps':'','observations_required_for_edge_union':51*20,'observations_available_total':675,'underidentified':True,'offline_identifiability':'NOT_IDENTIFIABLE_FROM_LOGGED_DATA','hindsight_audit_IR':None,'audit_not_used_to_select':True})
 ablations.append({'arm':'C Incumbent always eligible','rule':'identical original: Transfer protected at every elimination','n':20,'candidate_count':len(live),'selected_id':pol['P_TRANSFER']['id'],'selected_steps':str(pol['P_TRANSFER']['resampling_steps']),'observations_required_for_edge_union':675,'observations_available_total':675,'underidentified':True,'offline_identifiability':'EXACT_ALREADY_IMPLEMENTED','hindsight_audit_IR':float(R['P_TRANSFER'].mean()),'audit_not_used_to_select':True})
 # Prespecified small grid; reference is fixed incumbent, cancellation for UCB gaps.
 full20=sorted([p for p in allc if vector(p,20) is not None],key=lambda p:p['id']);x=np.stack([vector(p,20)-vector(pol['P_TRANSFER'],20) for p in full20],axis=1)
 for lam in [0.,1.,2.]:
  k=int(np.argmin(x.mean(axis=0)+lam*x.std(axis=0,ddof=1)/np.sqrt(20)));p=full20[k];reward=next((float(R[nm].mean()) for nm,q in pol.items() if q['id']==p['id']),None)
  ablations.append({'arm':f'D Paired UCB lambda{lam:g}','rule':'cost difference toTransfer + lambda*pairedSEM; diagnostic,not method','n':20,'candidate_count':len(full20),'selected_id':p['id'],'selected_steps':str(p['resampling_steps']),'observations_required_for_edge_union':len({e for q in full20 for e in edges(q)})*20,'observations_available_total':675,'underidentified':True,'offline_identifiability':'SUPPORTED_RESTRICTED_LOGGED_SET','hindsight_audit_IR':reward,'audit_not_used_to_select':True})
 ablate('E Best measured coarse20',supported,20,'prefer best fully observed coarse at20; originalCoarse remains eligible')
 ablations.append({'arm':'E Freeze initial coarse onabstain','rule':'retain recorded initialn5 coarse independently of laterelimination','n':5,'candidate_count':len(initial),'selected_id':pol['P_COARSE']['id'],'selected_steps':str(pol['P_COARSE']['resampling_steps']),'observations_required_for_edge_union':255,'observations_available_total':675,'underidentified':True,'offline_identifiability':'HINDSIGHT_DIAGNOSTIC_RULE','hindsight_audit_IR':float(R['P_COARSE'].mean()),'audit_not_used_to_select':True})
 table(out/'counterfactual_ablation.csv',ablations)
 # Plots use measured / reconstructed evidence only.
 fig,ax=plt.subplots(figsize=(7,4));ax.plot([q['n'] for q in logs],[len(q['details']) for q in logs],'o-',label='before');ax.plot([q['n'] for q in logs],[q['live_candidates'] for q in logs],'o-',label='after');ax.set(xlabel='Calibration prompts',ylabel='Candidates');ax.legend();fig.tight_layout();fig.savefig(out/'candidate_survival.png');plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4));labels=[r['A'].replace('ADAPTIVE_NEW','New')+' vs '+r['B'].replace('P_','') for r in pairrows];y=[r['predicted_delta'] for r in pairrows];ax.errorbar(range(3),y,yerr=[[v-r['predicted_CI_low'] for v,r in zip(y,pairrows)],[r['predicted_CI_high']-v for v,r in zip(y,pairrows)]],fmt='o');ax.axhline(0,c='gray');ax.set_xticks(range(3),labels);ax.set_ylabel('Predicted reward delta (paired calibration CI)');fig.tight_layout();fig.savefig(out/'path_cost_uncertainty.png');plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4));ax.bar(np.arange(3)-.18,[r['predicted_delta'] for r in pairrows],.36,label='graph predicted');ax.bar(np.arange(3)+.18,[r['actual_delta'] for r in pairrows],.36,label='audit realized');ax.set_xticks(range(3),labels);ax.axhline(0,c='gray');ax.legend();fig.tight_layout();fig.savefig(out/'ranking_disagreements.png');plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4));h=[r for r in heter if r['B']=='P_MANUAL'];h.sort(key=lambda r:r['mean_delta']);ax.bar(range(40),[r['mean_delta'] for r in h]);ax.plot(range(40),[r['seed1_delta'] for r in h],'.',label='seed1');ax.plot(range(40),[r['seed2_delta'] for r in h],'.',label='seed2');ax.axhline(0,c='gray');ax.set(xlabel='Prompts ordered by mean delta',ylabel='Transfer - Manual IR');ax.legend();fig.tight_layout();fig.savefig(out/'fallback_prompt_heterogeneity.png');plt.close(fig)
 # Detailed findings and supported confidence classes generated after reproduction.
 coarseelim=elim.get(pol['P_COARSE']['id']);newlead=[q['n'] for q in logs if q['leader']['id']==pol['ADAPTIVE_NEW']['id']];opt={label:float(np.mean([r['optimism_held_minus_train'] for r in optimism if r['cohort']==label])) for label in ['initial_all_proposals','fully_observed20']}
 diagnosis={
 'A':{'class':'Support limitation','confidence':'HIGH','supporting':'Fine timestep50 absent; exactManual increase taus outside{2,8,32}, both audit strong','against':'Coarse is exactly representable and has higher audit pointmean thanNew','missing':'fresh qualified expanded candidate audit'},
 'B':{'class':'Proposal limitation','confidence':'LOW','supporting':'candidate cap/exploration not global; unproposed paths lacklonger measurements','against':'initial mass96.98%; audited Coarse/New/Transfer allproposed','missing':'unproposed fullpolicy rewards and missingedge samples'},
 'C':{'class':'Selection noise / elimination','confidence':'MEDIUM','supporting':f'Coarse survives allrounds with20 pairedrows; jointbootstrap remains underidentified; LOOCV graphoptimism {opt}','against':'audit gapsNew-Coarse include0; noise alone cannot be isolated','missing':'independent same-trajectory estimates'},
 'D':{'class':'Edge-model mismatch','confidence':'LOW','supporting':'New-Coarse predicted andactual sign disagree','against':'differentcalibration and audit prompts; wideuncertainty; no same-row residual','missing':'paired edges and fullpolicy utility on samefreshpromptseed baseline'},
 'E':{'class':'Fallback mismatch','confidence':'HIGH' if fall[0]['CI_high']<0 else 'MEDIUM','supporting':f'Transfer-Manual delta{fall[0]["delta"]},CI[{fall[0]["CI_low"]},{fall[0]["CI_high"]}]; incumbent protected byrule','against':'priorTRAIN evaluations favoredTransfer; current40 internalprompts not finalTEST','missing':'independentpreregistered fallbackqualification'},
 'F':{'class':'Underidentified explanations','confidence':'HIGH','supporting':'cannot separate estimationnoise,promptshift,composition with differentcohorts; only3 graphscored auditedpolicies','against':'support exclusion andcurrentfallback mismatch identifiable','missing':'same-trajectory targeted diagnostic'}}
 write(out/'failure_diagnosis.json',{'classes':diagnosis,'dominant':'MULTIPLE: A andE directly demonstrated; C plausible; Dnot isolated','original_coarse_eliminated_n':coarseelim,'new_first_leads_n':newlead[0] if newlead else None,'LOOCV_graph_optimism':opt,'no_production_change':True})
 manualtemps=next(q['actual_temperatures'] for q in coverage if q['policy']=='P_MANUAL')
 texts={
 'coverage_diagnosis.md':f'Fine OUT_OF_GRID (50 neveradded). Manual OUT_OF_ACTION_GRID, actualeventtemperatures {manualtemps}. Neitherwas eliminated. Allthree graphscored auditedpolicies proposed. Initialcap mass96.98% is conditionalfittedgraph uncertainty, not rewardcoverage.',
 'racing_diagnosis.md':f'All four rounds independently reproduced exactly, including shared-edge cancellation. Coarse survived all rounds; New first leads n={newlead[0] if newlead else None}. Transfer protectedat everyround irrespective of cost, not restoredonly at end. See pairedgaps andcovariance contributions. Eliminationheuristics are not sequentiallyvalid guarantees.',
 'noise_diagnosis.md':f'LOOCV optimism heldcost-trainingcost: {opt}. This is graph-score optimism, not fullpolicyperformance. Fullyobserved20 subset is selected by originalallocator, so survivor conditioning limitsinterpretation. Missing common rows are notimputed. Bootstraprank andLOO frequencies show remainingselectionuncertainty.',
 'ranking_diagnosis.md':'OnlyNew/Coarse/Transfer have comparablecurrentcalibrationedges. Fine andManual have no validgraphcost here. Sign disagreement is observed, but not an additivecompositionresidual: calibrationandaudit cohorts differ. Rankcorrelation over3policies would onlybe descriptive.',
 'fallback_diagnosis.md':f'Transfer-Manual: {fall[0]}. Priorretuning report had +.0311CI[.0096,.0535] over48TRAIN; currentpairedresult reverses sign. Historicalraw unavailable, cohorts/seedsand executionversions differ, so this is not a paired cross-run causalreversal test. Statisticalabstention doesnot guaranteeperformance safety; futurefallback qualification shouldbe preregistered. Incumbentunchanged.',
 'ablation_identifiability.md':'Exact originalandprotectedincumbent replay supported. Fullno-cap continuation NOT_IDENTIFIABLE_FROM_LOGGED_DATA. First5 all270coarse paths supported, lateronly fullyobserved subsets. FixedUCB diagnosticgrid lambda0,1,2 was declared before computingcomparison, not optimizedagainst audit. Audit displayed onlyaftercalibrationselectionand labeledhindsight. Coarse-first alternatives are notvalidated on thissameaudit.'}
 for n,t in texts.items():(out/n).write_text(t+'\n')
 # ONE recommendation; not executed. Costs from existing measured profiler, no pricing.
 nexttext='''# Minimal next experiment (recommendation only; NOT RUN)
Freeze three policies: initial Coarse [20,40,80], tau[8,8,32]; proposed New [40,80,90], tau[8,32,32]; Transfer [40,60,90], tau[8,8,32]. On 12 fresh TRAIN prompt-seed trajectories, measure the union of their reference-conditioned edges and allthree fullpolicies against the same dense-reference terminalutility. Keep audited RNG/protocol; effects execution-protocol dependent. Do not use this forensic audit to tune objective or incumbent.
This directly separates same-row additive discrepancy from the calibration/audit promptshift, while estimating edge noise. It does not solve unsupportedFine/Manualcoverage or qualify a productionfallback.
'''
 unionedges=set(e for name in ['P_COARSE','ADAPTIVE_NEW','P_TRANSFER'] for e in edges(pol[name]));edgeNFE=sum((100-max(e[0]+1,0))*8 for e in unionedges)*12;referenceNFE=12*800;fullNFE=12*3*800;total=edgeNFE+referenceNFE+fullNFE
 chip_est=total/(269040/5.190002306088638)
 nexttext+=f'\nFresh trajectories:12. Extraedge observations:{len(unionedges)}×12={len(unionedges)*12}. Fullpolicyrollouts:36 plus12reference. Nominal no-sharing NFE:edge{edgeNFE}+reference{referenceNFE}+full{fullNFE}={total}. Allow extra official-loop replayoverhead. Approx accelerator-chip-hours:{chip_est:.2f}, range{chip_est*.7:.2f}–{chip_est*1.5:.2f} using current profilingthroughput; on4chips roughly{chip_est/4*60:.0f}minutes beforecoldstart. Actualcostunknown. Reference generationsharedoncepertrajectory; profilerfusion andrewardCPUtime meanNFE isnot a runtimeguarantee.\n'
 (out/'minimal_next_experiment.md').write_text(nexttext)
 fig,ax=plt.subplots(figsize=(9,4));ax.barh(['A Support exclusion','B Proposal limitation','C Selection noise','D Composition mismatch','E Fallback mismatch','F Remaining underidentification'],[3,1,2,1,3,3],color=['#d95f02','#aaa','#e6ab02','#aaa','#d95f02','#7570b3']);ax.set_xticks([1,2,3],['LOW','MEDIUM','HIGH']);ax.set_xlabel('Qualitative evidence confidence (not probability)');fig.tight_layout();fig.savefig(out/'failure_mode_summary.png');plt.close(fig)
 report=['# GRAPH SELECTION FAILURE FORENSICS','Raw-data integrity: PASS (historicalretuningraw unavailable)','Original report reproduced: YES','Strong-policy support: Fine and exactManual unsupported','Important candidate missing from graph: Fine timestep50; Manualtemperatureactions','Important candidate excluded during proposal: not demonstrated among auditedpolicies','Evidence of premature elimination: not demonstrated; Coarse survives all rounds, paired audit gap still includes0','Evidence of winner\'s curse: graph-score LOOCVoptimism '+str(opt)+'; descriptive, not proof of solefailurecause','Evidence of additive ranking mismatch: New/Coarse sign reversal; compositioncausalitynotidentifiable','Fallback Transfer vs Manual: DeltaIR='+str(fall[0]['delta'])+';95%CI=['+str(fall[0]['CI_low'])+','+str(fall[0]['CI_high'])+']','Dominant failure class: MULTIPLE — support limitation(A), fallback mismatch(E); selectionnoise(C)plausible; modelmismatch(D)unresolved','Can the problem be diagnosed offline? PARTIAL','Recommended minimal next experiment: same12freshTRAINtrajectories, threefrozenpolicies andtheirunionedges against sameReference','Estimated additional compute: '+str(len(unionedges)*12)+'edgeobservations;36fullpolicyrollouts+12reference; nominal'+str(total)+'NFE; approximate'+str(round(chip_est,2))+'TPU-chip-hours;NOT RUN','## Policy coverage\n'+json.dumps(coverage,indent=2),'## Predicted versusactual\n'+json.dumps(pairrows,indent=2),'## Fallbackanalysis\n'+json.dumps(fall,indent=2),'## Findings\n'+json.dumps(diagnosis,indent=2),'## Limits\nAll latest40prompt audit statistics average two pairedseeds before50k promptbootstrap. Retuninghistory report only; no historicalraw invented. Candidate selection/counterfactualrules use calibrationonly; audit comparisonhindsight, not validation. No newrollout/verifierinference, no productionchanges. Rawedge archive contains all675records; sourceinventory and reproductionchecks saved.','## Technical conclusion\nThe compiler could not represent the two strongest auditedbaselines. Coarse was fully measured on20rows and survived every round, so premature elimination does not explain this audited comparison. Its finalgraphleadingpolicy ranks aboveCoarse oncalibration butbelowit onaudit. This supports selectionfragility, while differentcohorts preventisolating reference-conditionedcompositionfailure. The protectedhistoricalTransfer fallback underperformsManual in thisaudit: abstention was faithful toprotocol butnot a performance-safety guarantee.']
 (out/'REPORT.md').write_text('\n\n'.join(report)+'\n')
 # Executive summary uses only findings already reproduced above.
 finalstats={r['policy_id']:r for r in roundrows if r['n']==20};newfreq=finalstats[pol['ADAPTIVE_NEW']['id']]['winner_frequency']
 unresolved_count=sum(r['unresolved_at_margin01'] for r in unresolved)
 tm=fall[0];tf=fall[2]
 sections=[
 '# GRAPH SELECTION FAILURE FORENSICS',
 'Raw-data integrity: PASS for the latest campaign; PARTIAL for historical context (N8 retuning raw records unavailable).',
 'Original report reproduced: YES — 675 observations, 2,900 historical Fine observations, ratio 0.2327586, all five reported audit means, 400 unique audit records and identical initial particles for all 80 paired units.',
 'Strong-policy support: Fine and exact Manual are not representable. Fine requires node 50; Manual uses event temperatures '+manualtemps+' rather than {2, 8, 32}.',
 'Important candidate missing from graph: Fine. Important candidate excluded during proposal: not demonstrated among the audited representable policies.',
 'Evidence of premature elimination: NO for the audited comparison. Coarse survived every round and all three graph-scored audited policies have 20 common calibration observations.',
 'Evidence of winner\'s curse: DESCRIPTIVE. Leave-one-out held-out minus selection-training graph cost is '+str(round(opt['initial_all_proposals'],6))+' at n=5 and '+str(round(opt['fully_observed20'],6))+' at n=20. These are graph-score diagnostics, not full-policy reward estimates; the n=20 set is conditioned on the original allocator.',
 'Evidence of additive ranking mismatch: YES as a point-estimate sign reversal; NO as an isolated composition-error diagnosis. Calibration and audit prompts differ and both New-versus-Coarse intervals cross zero.',
 f'Fallback Transfer vs Manual: Delta IR = {tm["delta"]:+.6f}; 95% prompt-bootstrap CI = [{tm["CI_low"]:+.6f}, {tm["CI_high"]:+.6f}]. Transfer vs Fine = {tf["delta"]:+.6f}, CI [{tf["CI_low"]:+.6f}, {tf["CI_high"]:+.6f}].',
 'Dominant failure class: MULTIPLE — A support limitation and E fallback mismatch are directly supported. C selection noise is plausible; D reference-conditioned model mismatch is not isolated. Can the problem be diagnosed offline? PARTIAL.',
 'Recommended minimal next experiment: same-trajectory edge-versus-full-policy diagnostic for New, Coarse and Transfer, on 12 fresh TRAIN prompt–seed trajectories. '+str(len(unionedges)*12)+' extra edge observations, 36 full-policy runs plus 12 references; nominal '+str(total)+' NFE; approximately '+str(round(chip_est,2))+' TPU-chip-hours before cold-start overhead. NOT RUN.',
 '## 1. Exact reproduction and provenance\nRaw regrets equal reference reward minus intervention reward. Audit utility equals the maximum endpoint score among eight particles. Every paired difference averages the two seeds within each of 40 prompts before bootstrap. Compiler and audit prompt sets are disjoint within the campaign; all prompts were historically explored TRAIN prompts. Inventory records real paths, missing historical raw data and hashes. No VALIDATION, TEST or external pool was loaded.',
 '## 2. Coverage\nThe initial set contains '+str(len(initial))+' candidates including Transfer and covers 96.98% of the fitted coarse-graph bootstrap winner mass. That coverage is conditional on this estimator, not coverage of genuinely best full-policy reward. The expanded candidate grid is '+str(sorted(expanded))+'. Fine and Manual were never candidates, not eliminated candidates. New, Coarse and Transfer were proposed initially and remained alive at termination. See policy_coverage.csv and candidate_stage_history.csv.',
 '## 3. Racing decisions\nAll four rounds reproduce the recorded bootstrap frequencies, paired standard errors and elimination decisions exactly. Shared edges cancel correctly on common rows. Coarse led at n=5. New first led at n=15 and led again at n=20. Its final winner frequency was '+str(round(newfreq*100,3))+'%, below the preregistered 80% stability threshold. Transfer was always protected, including n=5 when its rank was 23 and winner frequency zero; it ranked second at n=20 but had only about 5.7% winner frequency. Retention followed the incumbent rule, not demonstrated reward safety. '+str(unresolved_count)+' of '+str(len(unresolved))+' final paired candidate comparisons remain unresolved at the practical 0.01 margin using descriptive repeated-bootstrap intervals. See round_reconstruction.csv, final_candidate_paired_uncertainty.csv and edge_uncertainty_contributions.csv.',
 '## 4. Noise and ranking\nNew-minus-Coarse predicted reward is +0.027442, calibration CI approximately [-0.0515, +0.1132]. Actual audit reward is -0.026550, CI approximately [-0.0770, +0.0174]. Their sign disagreement is real at the point-estimate level but neither margin is precise. Coarse-minus-Transfer also reverses sign, while New-minus-Transfer agrees in sign. Only three distinct frozen audited policies have comparable graph scores; Fine and Manual graph scores are unavailable and were not invented. Bootstrap rank ranges, leave-one-out sensitivity and conditional out-of-bag selection optimism are saved. Different prompt sets prevent calling these discrepancies additive composition residuals.',
 f'## 5. Fallback\nTransfer beats Manual on {tm["wins"]}/40 prompt means. Both seed-average differences are negative ({tm["seed1_mean"]:+.6f}, {tm["seed2_mean"]:+.6f}), but their signs agree within only {tm["seed_sign_agree_prompts"]}/40 individual prompts. Removing the three largest absolute prompt differences reduces the point gap to {tm["remove3_largest_abs_delta"]:+.6f}; every leave-one-prompt-out mean remains negative. Large prompts affect magnitude but a single prompt does not determine the sign. Previous retuning reported Transfer–Manual +0.0311, CI [+0.0096, +0.0535]; the latest internal run shows the opposite direction with a negative interval. Historical raw records are missing and cohorts, seeds and execution versions differ, so no paired cross-run causal reversal test is claimed. These are exploratory forensic intervals, not a multiplicity-adjusted confirmatory family. Statistical abstention is faithful to the rule; it is not performance safety.',
 '## 6. Counterfactuals\nOriginal and incumbent-always-retained replay are exactly supported; the latter was already implemented. All 270 coarse paths can be scored at n=5; this still picks Coarse. Full no-cap continuation is NOT_IDENTIFIABLE_FROM_LOGGED_DATA because later edge rows are missing. At n=20, 29 coarse paths have complete measurements; their point leader is still New, so removing the cap within that observed subset does not fix the ranking. Paired uncertainty penalty lambda 0 or 1 selects New; lambda 2 selects Transfer. This fixed diagnostic grid was not fitted to audit reward, and its outcomes do not validate a new method. Freezing initial Coarse on abstention would have audit mean 0.992495, but that is hindsight evidence only. No production incumbent or schedule was changed.',
 '## 7. Diagnosis and next action\nSupport exclusion explains why the compiler could not discover the two strongest audited baselines. It does not explain all of the New-versus-Coarse ranking discrepancy, since Coarse was fully observed and survived. Available evidence cannot distinguish calibration noise, prompt shift and reference-conditioned edge interactions. One targeted same-row test is recommended in minimal_next_experiment.md; no additional inference has been launched.',
 '## Technical conclusion\nThis was not a demonstrated premature-elimination failure for Coarse. The directly established failures are limited support for strong baselines and an incumbent fallback that underperformed on this audit distribution. Remaining additive ranking uncertainty is substantial; composition failure is a hypothesis, not a proven cause.'
 ]
 (out/'REPORT.md').write_text('\n\n'.join(sections)+'\n')
 (out/'racing_diagnosis.md').write_text(next(t for t in sections if t.startswith('## 3.'))+'\n')
 (out/'coverage_diagnosis.md').write_text(next(t for t in sections if t.startswith('## 2.'))+'\n')
 (out/'fallback_diagnosis.md').write_text(next(t for t in sections if t.startswith('## 5.'))+'\n')
 (out/'noise_diagnosis.md').write_text(sections[6]+'\n\n'+next(t for t in sections if t.startswith('## 4.'))+'\n')
 (out/'ranking_diagnosis.md').write_text(next(t for t in sections if t.startswith('## 4.'))+'\n')
 (out/'ablation_identifiability.md').write_text(next(t for t in sections if t.startswith('## 6.'))+'\n')
 write(out/'analysis_integrity.json',{'passed':True,'four_rounds_exact':True,'shared_edges_cancel':True,'report_means_exact':True,'no_inference':True,'current_edge_records':675,'audit_rows':400,'script_sha256':digest(Path(__file__))})
 print(json.dumps({'coverage':coverage,'coarse_eliminated_n':coarseelim,'new_first_leads_n':newlead,'fallback':fall,'optimism':opt,'pairs':pairrows,'next_NFE':total,'next_edges':len(unionedges)},indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--source',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--history',type=Path,required=True);a.add_argument('--retuning-report',type=Path,required=True);x=a.parse_args();run(x.source.resolve(),x.output.resolve(),x.history.resolve(),x.retuning_report.resolve())
