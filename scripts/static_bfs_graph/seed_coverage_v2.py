"""Prospective TRAIN-only equal-trajectory allocation and coverage diagnostic."""
from .adaptive_refinement import read,dump,sha,csvout,job,jobid,unitpath,TAUS,G0
from .tpu_reference_compare import checked_inputs,records
from .policy import StaticBFSPolicy
from .runner import static_record
from .graph_builder import path_edges
from pathlib import Path
import numpy as np,itertools,time,os,subprocess,fcntl,sys,shutil,traceback,json
REPO=Path(__file__).resolve().parents[2]
SEEDS=[2026100902,2026100903]
SHAPES=[(20,40,80),(20,60,80),(40,60,90),(40,80,90),(50,60,80),(30,50,80),(20,50,90),(40,50,80)]
PATTERNS=[(8.,8.,32.),(8.,2.,32.),(2.,8.,32.)]
def edges(p):return [job(e.src,e.dst,e.tau_dst) for e in path_edges(tuple(p['resampling_steps']),tuple(t for s,t in p['temperature_by_step']))]
def union(ps):return list({jobid(j):j for p in ps for j in edges(p)}.values())
def verify(root):
 m=read(root/'manifest.json');assert all(sha(REPO/n)==h for n,h in m['sources'].items());assert all(sha(root/n)==h for n,h in m['frozen'].items())
def freeze(root):
 if (root/'manifest.json').exists():verify(root);return
 root.mkdir(parents=True,exist_ok=True);pool,cfg,base=checked_inputs();ids=pool['train_indices'];ps=[static_record(StaticBFSPolicy(s,tuple(zip(s,t)),8)) for s in SHAPES for t in PATTERNS];ps.append(static_record(StaticBFSPolicy((40,80,90),((40,8.),(80,32.),(90,32.)),8)));assert len(ps)==25 and len(union(ps))==32
 config={'protocol':'SEED_COVERAGE_COMPILER_V2_TRAIN_DIAGNOSTIC','seeds':{'A':SEEDS,'B':[],'C':[2026100904,2026100905]},'snapshots':list(range(5,100,5)),'N':8,'K':3,'steps':100,'eta':1.,'dtype':'BF16','checkpoints_trajectories':[10,20],'bootstrap_replicates':20000,'audit_bootstrap_replicates':50000,'stat_seed':2026100920,'max_observations':2030,'nominal_max_unique_compiler_observations':960,'reference_tau':8.,'historically_explored_TRAIN_only':True}
 prereg={'candidate_bank':'25 prespecified paths; eight temporal shapes x3temperaturepatterns pluspreviousNew; not globalDAG optimization','coverage_quota':'four coarse andfour fine-like shapes, chosen independently of current graph scores; timestep50 alwaysrepresented; temperature supportunchanged','shapes':SHAPES,'temperature_patterns':PATTERNS,'parent_selection_bias':'Fine/New shapes were historically examined; freshseeds onhistorically exploredTRAIN, not independentconfirmation','arms':{'SINGLE':'first20TRAIN xseed1','MULTI':'first10TRAIN xseeds1,2'},'same_budget':'10 then20trajectories peredge for botharms; samecandidateedgeunion acquisition; common10promptseed1 reused; maximum960newobservations total','allocation_confounds':'promptcount20vs10 changes withseedcount; matchedfirst10singlevsaverage2 additionaldiagnostic hasdifferentcost andcannotisolatepureseedbenefit','AER':'paired leadercomparison covariance; eliminate whenP(cost_gap>.01)>=.95 andwinnerfreq<.01; Transfer protected; nextbatch orderedbylargest pair-covariance contribution; acquireonly unionof remainingpaths acrossbotharms, commonrows complete','stable':'samepointleader two looks, winnerfreq>=.8 andallpairP(leader_gap<.02)>=.95; allocationheuristics notsequentialguarantees','fallback':'preselectedTransfer unchanged; recordabstention separately fromaudit performance; noautoqualification','audit':'last40TRAIN disjointfromcompiler20, twofreshseeds; freezeleaders fromfull/coarse-restrictedbanks beforeinference; includeFine,Manual,Transfer;50k promptpairedbootstrap','primary':'MULTI broadleader minusSINGLE broadleader auditIR; reportCI andNI loweragainstFine margin-.02; no productionchange','counterfactual':'coarse-restricted samecalibration measuredpaths only; fixed coverage comparison iscandidatebank ablation, not fullpreviouscompiler rerun','optional_newexploration':False,'stop':'twofixedlooksandfrozenaudit, no reward-dependentextension'}
 split={'A':ids[:20],'B':ids[20:],'C':ids[20:],'TRAIN_prompts':pool['prompts'],'MULTI_prompts':ids[:10],'compiler_audit_disjoint':True}
 for n,x in [('config.json',config),('preregistration.json',prereg),('data_split.json',split),('dataset_split.json',split),('candidate_bank.json',ps),('reference_schedule.json',records()['DENSE_REFERENCE'])]:dump(root/n,x)
 names=['seed_coverage_v2.py','seed_coverage_v2_tpu.py','adaptive_refinement.py','adaptive_refinement_tpu.py','policy.py','graph_builder.py','adapter.py','runner.py','tpu_reference_compare.py']
 sources={str((REPO/'scripts/static_bfs_graph'/n).relative_to(REPO)):sha(REPO/'scripts/static_bfs_graph'/n) for n in names}
 for n in ['scripts/diffusion_classical_search_t2i/tpu_runner.py','modal_n8_gpu/train_pool.json','configs/static_bfs_graph_t2i.json','configs/diffusion_classical_search_t2i_table12.json']:sources[n]=sha(REPO/n)
 dump(root/'manifest.json',{'sources':sources,'frozen':{n:sha(root/n) for n in ['config.json','preregistration.json','data_split.json','dataset_split.json','candidate_bank.json','reference_schedule.json']},'model_revision':base['models']['sd15']['revision']})
 for n in sources:
  p=root/'sources'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/n,p)
 (root/'commands.md').write_text('cd '+str(REPO)+'\n'+sys.executable+' -m scripts.static_bfs_graph.seed_coverage_v2 --run-root '+str(root)+'\n')
def stage(root,name,units=None,jobs=None):
 verify(root)
 if (root/'stage_ledger'/f'{name}.json').exists():return
 if units is not None:dump(root/'tasks'/f'{name}.json',{'units':units,'jobs':jobs})
 start=time.time();dump(root/'status.json',{'stage':name,'status':'RUNNING','start_unix':start});chips=[0] if name=='gates' else [0,1,2,3];procs=[];logs=[]
 assert shutil.disk_usage(root).free>5*1024**3
 try:
  for chip in chips:
   p=root/'logs'/f'{name}-chip{chip}.log';p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a');logs.append(f)
   env={**os.environ,'PJRT_DEVICE':'TPU','TPU_VISIBLE_CHIPS':str(chip),'PYTHONUNBUFFERED':'1','STATIC_BFS_CACHE_ROOT':f'/dev/shm/static-bfs-seed-coverage-v2/chip{chip}/cache'}
   procs.append(subprocess.Popen([sys.executable,'-m','scripts.static_bfs_graph.seed_coverage_v2_tpu','--run-root',str(root),'--stage',name,'--chip',str(chip)],env=env,stdout=f,stderr=subprocess.STDOUT))
  dump(root/'active_workers.json',{'stage':name,'PIDs':[p.pid for p in procs]})
  while True:
   states=[p.poll() for p in procs]
   if any(s not in (None,0) for s in states):raise RuntimeError(f'{name} failed {states}')
   if all(s==0 for s in states):break
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
def count(root):return len(list((root/'observations/A').glob('*/*.json')))
def units(root,n,arm):
 ids=read(root/'data_split.json')['A']
 return [(p,SEEDS[0]) for p in ids[:n]] if arm=='SINGLE' else list(itertools.product(ids[:n//2],SEEDS))
def matrix(root,ps,n,arm):
 us=units(root,n,arm);cols=[]
 for p in ps:
  v=np.array([sum(read(unitpath(root,'A',pid,s,j))['regret'] for j in edges(p)) for pid,s in us])
  cols.append(v if arm=='SINGLE' else v.reshape(n//2,2).mean(axis=1))
 return np.stack(cols,axis=1)
def stats(root,ps,n,arm):
 ps=sorted(ps,key=lambda p:p['id']);x=matrix(root,ps,n,arm);cfg=read(root/'config.json');idx=np.random.default_rng(cfg['stat_seed']+n+(1 if arm=='MULTI' else 0)).integers(0,len(x),(20000,len(x)));b=x[idx].mean(axis=1);prob=np.bincount(np.argmin(b,axis=1),minlength=len(ps))/20000;leader=int(np.argmin(x.mean(axis=0)));inc=records()['TRANSFER4_TO_8']['id'];details=[];live=[]
 for i,p in enumerate(ps):
  d=x[:,i]-x[:,leader];pr=float(np.mean(b[:,i]-b[:,leader]>.01));drop=bool(i!=leader and p['id']!=inc and pr>=.95 and prob[i]<.01)
  details.append({'policy':p,'cost':float(x[:,i].mean()),'winner_probability':float(prob[i]),'paired_gap_to_leader':float(d.mean()),'paired_SE':float(d.std(ddof=1)/np.sqrt(len(x))),'P_worse_gt01':pr,'eliminated':drop})
  if not drop:live.append(p)
 uncertainty=[]
 for j in union(live):
  es=np.array([read(unitpath(root,'A',pid,s,j))['regret'] for pid,s in units(root,n,arm)])
  if arm=='MULTI':es=es.reshape(n//2,2).mean(axis=1)
  contributions=[]
  for i,p in enumerate(ps):
   if p['id'] not in {q['id'] for q in live}:continue
   weight=int(j in edges(p))-int(j in edges(ps[leader]));d=x[:,i]-x[:,leader]
   contributions.append(float(weight*np.cov(es,d,ddof=1)[0,1]/len(x)))
  uncertainty.append({'edge':j,'pair_variance_priority':max([abs(v) for v in contributions] or [0])})
 stable_precision=prob[leader]>=.8 and all(float(np.mean(b[:,leader]-b[:,i]<.02))>=.95 for i in range(len(ps)) if i!=leader)
 coarse=[p for p in ps if set(p['resampling_steps'])<=set(G0)];cx=matrix(root,coarse,n,arm);coarseleader=coarse[int(np.argmin(cx.mean(axis=0)))]
 return {'arm':arm,'trajectory_count':n,'prompt_count':len(x),'candidate_count':len(ps),'leader':ps[leader],'coarse_restricted_leader':coarseleader,'winner_probability':float(prob[leader]),'entropy':float(-sum(v*np.log(v) for v in prob if v>0)),'precision_stable':bool(stable_precision),'details':details,'live':live,'uncertainty_edges':sorted(uncertainty,key=lambda v:-v['pair_variance_priority'])}
def profile(root,name,js,us):
 missing=sum(not unitpath(root,'A',p,s,j).exists() for p,s in us for j in js);assert count(root)+missing<=2030,'hardbudgetstop'
 dump(root/'budget_reservation.json',{'stage':name,'used':count(root),'reserved':missing,'total_max':count(root)+missing,'cap':2030});stage(root,name,us,js);assert count(root)<=2030

def bootstrap(x,seed):
 idx=np.random.default_rng(seed).integers(0,len(x),(50000,len(x)));z=x[idx].mean(axis=1)
 return {'delta':float(x.mean()),'SE':float(x.std(ddof=1)/np.sqrt(len(x))),'CI_low':float(np.quantile(z,.025)),'CI_high':float(np.quantile(z,.975)),'lower_one_sided95':float(np.quantile(z,.05)),'wins':int(sum(x>0)),'ties':int(sum(x==0)),'losses':int(sum(x<0))}
def finalize(root,rounds):
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 cfg=read(root/'config.json');split=read(root/'data_split.json');pol=read(root/'frozen_policies.json');raw=[read(p) for p in (root/'audit').glob('*.json')];unique=len({p['id'] for p in pol.values()});assert len(raw)==80*unique;look={(r['prompt'],r['seed'],r['policy']['id']):r for r in raw};R={n:np.array([np.mean([look[(pid,s,p['id'])]['reward'] for s in cfg['seeds']['C']]) for pid in split['C']]) for n,p in pol.items()}
 initial={}
 for r in raw:initial.setdefault((r['prompt'],r['seed']),set()).add(r['initial_latents_sha256']);assert len(r['events'])==3
 assert len(initial)==80 and all(len(v)==1 for v in initial.values())
 summary=[{'policy':n,'steps':str(pol[n]['resampling_steps']),'tau':str(pol[n].get('temperature_by_step',pol[n].get('tempering'))),'mean_IR':float(v.mean()),'NFE':800,'intermediate_scores':24} for n,v in R.items()];csvout(root/'audit_summary.csv',summary)
 pairs=[('MULTI_BROAD','SINGLE_BROAD'),('MULTI_BROAD','FINE'),('SINGLE_BROAD','FINE'),('MULTI_BROAD','MANUAL'),('SINGLE_BROAD','MANUAL'),('MULTI_BROAD','TRANSFER'),('SINGLE_BROAD','TRANSFER'),('MULTI_BROAD','MULTI_COARSE'),('SINGLE_BROAD','SINGLE_COARSE')]
 contrasts=[{'A':a,'B':b,**bootstrap(R[a]-R[b],cfg['stat_seed']+i)} for i,(a,b) in enumerate(pairs)];csvout(root/'paired_comparisons.csv',contrasts)
 variance=[];ps=read(root/'candidate_bank.json')
 for p in ps:
  vals=[]
  for pid in split['A'][:10]:
   try:vals.append([sum(read(unitpath(root,'A',pid,s,j))['regret'] for j in edges(p)) for s in SEEDS])
   except FileNotFoundError:vals=[];break
  if not vals:continue
  vals=np.array(vals);variance.append({'id':p['id'],'common_prompts':10,'seed1_mean':float(vals[:,0].mean()),'seed2_mean':float(vals[:,1].mean()),'mean2_mean':float(vals.mean()),'seed_disagreement_SD':float((vals[:,0]-vals[:,1]).std(ddof=1)),'prompt_SE_single10':float(vals[:,0].std(ddof=1)/np.sqrt(10)),'prompt_SE_mean2_10':float(vals.mean(axis=1).std(ddof=1)/np.sqrt(10))})
 csvout(root/'seed_variance_diagnostic.csv',variance)
 stages=[read(p) for p in (root/'work_ledgers').glob('*.json')];work={}
 for d in stages:
  kind='audit' if d['stage']=='audit' else 'correctness' if d['stage']=='gates' else 'compiler';w=work.setdefault(kind,{'NFE':0,'verifier_scores':0,'verifier_calls':0,'chip_hours':0,'references':0})
  for a,b in [('NFE','actual_denoising_NFE'),('verifier_scores','actual_verifier_candidate_scores'),('verifier_calls','actual_verifier_batch_calls'),('references','reference_trajectories')]:w[a]+=d[b]
  w['chip_hours']+=d['wall_seconds_including_load']/3600
 logical=[]
 for arm in ['SINGLE','MULTI']:
  us=set(units(root,20,arm));logical.append({'arm':arm,'unique_measured_edge_trajectory_rows':sum((read(p)['prompt'],read(p)['seed']) in us for p in (root/'observations/A').glob('*/*.json')),'final_unit_budget':20})
 csvout(root/'arm_cost_coverage.csv',logical);dump(root/'compute_accounting.json',{'unique_new_observations':count(root),'cap':2030,'nominal_maximum_fixed_bank':960,'logical_arms':logical,'shared_rows_not_double_counted':True,'work':work,'wall_seconds':time.time()-read(root/'driver.pid.json')['started_unix'],'actual_billing':'UNKNOWN','sameedge_plan_marginal_measurements_shared_between_arms':True})
 decision={}
 for arm in ['SINGLE','MULTI']:
  rows=[r for r in rounds if r['arm']==arm];stable=rows[-1]['leader']['id']==rows[0]['leader']['id'] and rows[-1]['precision_stable'];decision[arm]={'stable_new_graph_choice':bool(stable),'leading_policy':pol[arm+'_BROAD'],'abstain':not bool(stable),'fallback':records()['TRANSFER4_TO_8'] if not stable else None}
 decision['no_production_change']=True;decision['historically_explored_TRAIN_only']=True;dump(root/'decision.json',decision)
 fig,ax=plt.subplots();
 for arm in ['SINGLE','MULTI']:
  rr=[r for r in rounds if r['arm']==arm];ax.plot([r['trajectory_count'] for r in rr],[r['winner_probability'] for r in rr],'o-',label=arm)
 ax.axhline(.8,ls='--',c='gray');ax.set(xlabel='Trajectories peredge',ylabel='Bootstrap winner frequency');ax.legend();fig.tight_layout();fig.savefig(root/'stability_by_allocation.png');plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,4));ax.bar(range(len(summary)),[r['mean_IR'] for r in summary]);ax.set_xticks(range(len(summary)),[r['policy'] for r in summary],rotation=45,ha='right');ax.set_ylabel('Audit meanIR,40TRAINpromptmeans');fig.tight_layout();fig.savefig(root/'audit_policy_comparison.png');plt.close(fig)
 report=['# SEED-AWARE AND COVERAGE-AWARE COMPILER V2','Correctness gates: PASS. Fixed25candidate bank,32unionedges. 20prompt×1seed versus10prompt×2seeds at equal20trajectory peredge finalbudget. Candidates include node50 independentlyof initialgraph winners. No globalexpandedDAG optimum claim.',f'Unique compilation observations: {count(root)}; cap2030; maximum fixedbank960. Sharing overlap avoids duplicate execution. Logical costsperarm are reported separately.', '## Calibration rounds\n```json\n'+json.dumps([{k:v for k,v in r.items() if k not in ['details','live','uncertainty_edges']} for r in rounds],indent=2)+'\n```','## Frozen-policy audit\n```json\n'+json.dumps(summary,indent=2)+'\n```','## Paired comparisons\n```json\n'+json.dumps(contrasts,indent=2)+'\n```','## Decisions\n```json\n'+json.dumps(decision,indent=2)+'\n```','## Interpretation\nThis compares budget allocation across20promptsone seed vs10promptstwo seeds, so promptcountchanges togetherwithseedcount. Matching first10seed1 versusmean2 isadditionaldiagnostic withdifferentcost, not a causal isolationofseedcount. Botharms usecommoncandidate support; coarse-restricted leaders are calibration-only ablations onobservedpaths, not a replayof oldcompiler. Candidatebank contains previouslyexaminedFine/New schedules, so freshseed internalTRAINaudit isnot independentconfirmation. Mean2 seedbootstrap resamplesprompts jointly,not20independentseedrows. AER priorities arecovariance-aware allocationheuristics; nextstage measurescommoncomparable rows andnever assignszeroto missingedge. ProtectedTransfer hasno automaticperformance safetyqualification. Keep productionpolicy unchanged.']
 (root/'REPORT.md').write_text('\n\n'.join(report));dump(root/'INTEGRITY.json',{'passed':True,'gates':True,'cap_verified':count(root)<=2030,'audit_rows':len(raw),'audit_prompts':40,'audit_seeds':2,'identical_initial_particles':True,'compiler_audit_disjoint':not bool(set(split['A'])&set(split['C'])),'policy_hash':sha(root/'frozen_policies.json')})
def run(root):
 root.mkdir(parents=True,exist_ok=True);lock=(root/'driver.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (root/'status.json').exists() and read(root/'status.json').get('stage')=='DONE':return
 if not (root/'driver.pid.json').exists():dump(root/'driver.pid.json',{'pid':os.getpid(),'started_unix':time.time()})
 try:
  freeze(root);stage(root,'gates');assert read(root/'gates.json')['passed'];bank=read(root/'candidate_bank.json');live={'SINGLE':bank,'MULTI':bank};rounds=[]
  for n in [10,20]:
   pool={p['id']:p for ps in live.values() for p in ps};js=union(list(pool.values()));us=sorted(set(units(root,n,'SINGLE')+units(root,n,'MULTI')))
   if rounds:
    priority={}
    for rr in rounds[-2:]:
     for e in rr['uncertainty_edges']:priority[jobid(e['edge'])]=max(priority.get(jobid(e['edge']),0),e['pair_variance_priority'])
    js.sort(key=lambda j:(-priority.get(jobid(j),0),jobid(j)))
   profile(root,f'profile{n}',js,us)
   for arm in ['SINGLE','MULTI']:
    rr=stats(root,live[arm],n,arm);rr['unique_observations']=count(root);rounds.append(rr);live[arm]=rr['live']
   dump(root/'racing_rounds.json',rounds)
  last={r['arm']:r for r in rounds[-2:]};fine=next(p for p in bank if p['resampling_steps']==[50,60,80] and p['temperature_by_step']==[[50,8.],[60,2.],[80,32.]])
  pol={'SINGLE_BROAD':last['SINGLE']['leader'],'MULTI_BROAD':last['MULTI']['leader'],'SINGLE_COARSE':last['SINGLE']['coarse_restricted_leader'],'MULTI_COARSE':last['MULTI']['coarse_restricted_leader'],'FINE':fine,'MANUAL':records()['MANUAL8'],'TRANSFER':records()['TRANSFER4_TO_8']};dump(root/'frozen_policies.json',pol);dump(root/'freeze_receipt.json',{'sha256':sha(root/'frozen_policies.json'),'before_audit':True,'timestamp':time.time()});stage(root,'audit');finalize(root,rounds);verify(root);dump(root/'status.json',{'stage':'DONE','all_stages_complete':True,'end_unix':time.time()})
 except BaseException as e:dump(root/'status.json',{'stage':'ABORTED','exception':repr(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();run(a.run_root.resolve())
