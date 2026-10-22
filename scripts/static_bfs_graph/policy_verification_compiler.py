"""Frozen TRAIN-only graph proposal -> direct verification -> disjoint audit."""
from pathlib import Path
import argparse,fcntl,hashlib,itertools,json,os,shutil,subprocess,sys,time,traceback
import numpy as np
from .adaptive_refinement import read,dump,sha,csvout,unitpath
from . import seed_coverage_v2 as v2
from .tpu_reference_compare import checked_inputs,records
REPO=Path(__file__).resolve().parents[2]
SEEDS={'A':[2026100911],'B':[2026100912,2026100913],'C':[2026100914,2026100915,2026100916,2026100917]}

def verify(root):
 m=read(root/'manifest.json')
 assert all(sha(REPO/n)==h for n,h in m['sources'].items()),'source drift'
 assert all(sha(root/n)==h for n,h in m['frozen'].items()),'frozen plan drift'

def freeze(root,request):
 root.mkdir(parents=True,exist_ok=True)
 if (root/'manifest.json').exists():verify(root);return
 pool,cfg,old=checked_inputs();ids=pool['train_indices'];assert len(ids)==60
 ids=sorted(ids,key=lambda i:(hashlib.sha256(f'policy_verification_compiler_v1:{i}'.encode()).hexdigest(),i))
 bank=[v2.static_record(v2.StaticBFSPolicy(s,tuple(zip(s,t)),8)) for s in v2.SHAPES for t in v2.PATTERNS]
 bank.append(v2.static_record(v2.StaticBFSPolicy((40,80,90),((40,8.),(80,32.),(90,32.)),8)))
 # Read only the frozen proposal identities, never previous audit rewards.
 previous=read(REPO/'../artifacts/static-bfs-graph/seed_coverage_compiler_v2/candidate_bank.json') if (REPO/'../artifacts/static-bfs-graph/seed_coverage_compiler_v2/candidate_bank.json').exists() else read(Path('/home/thanhlamtba31/static-bfs-dcs/experiments/seed_coverage_compiler_v2/candidate_bank.json'))
 assert {p['id'] for p in bank}=={p['id'] for p in previous};assert len(bank)==len({p['id'] for p in bank})==25 and len(v2.union(bank))==32
 ref=records()['DENSE_REFERENCE'];assert ref['resampling_steps']==[10,20,30,40,60,80,90] and all(t==8 for _,t in ref['temperature_by_step'])
 split={'A':ids[:20],'B':ids[20:36],'C':ids[36:],'TRAIN_prompts':pool['prompts'],'allocation':'ascending SHA256(policy_verification_compiler_v1:prompt_id), reward independent'}
 config={'protocol':'POLICY_VERIFICATION_COMPILER_V1','N':8,'K':3,'steps':100,'eta':1.,'dtype':'BF16','Max':True,'SSP':True,'verifier':'ImageReward','taus':[2,8,32],'seeds':SEEDS,'snapshots':list(range(5,100,5)),'graph_bootstraps':20000,'paired_bootstraps':50000,'certificate_bootstraps':200000,'stat_seed':2026100930,'epsilon':.02,'tpu_chips':4,'graph_observation_cap':640,'verification_rollout_cap':160,'audit_rollout_cap':672}
 prereg={'internal_TRAIN_only':True,'historically_explored_prompts':True,'disjoint_campaign_cohorts':[20,16,24],'fresh_stage_seeds':SEEDS,'reference':ref,'bank':'original V2 25 identities, 32 edge union; no prior audit scores loaded','shortlist_order':['GRAPH_MEAN','GRAPH_BOOTSTRAP','GRAPH_ROBUST','COVERAGE','INCUMBENT'],'duplicates':'next unused candidate by criterion, tie by ID; incumbent is fixed Transfer, append reason if already included rather than invent substitute','robust':'mean additive cost + paired path SE (across prompts)','certificate':'Bonferroni one-sided percentile prompt bootstrap upper bounds, alpha=.05/(k*(k-1)), 200000 resamples, all ordered candidate pairs protects data-selected winner; approximate bootstrap coverage, not finite-sample guarantee','winner':'highest Stage B mean, ties by policy ID; UNDERIDENTIFIED retains tentative winner, no Transfer fallback','primary':'verified minus graph-only on 24 audit prompt means of four seeds','audit':'entire shortlist + genuine Manual8 + Fine, exact dedup; freeze before audit opens','oracle':'hindsight shortlist best only, descriptive point regret without unadjusted oracle CI','learning_curve':'fixed Stage B prefix4/8/12/16, retrospective only','cost':'actual ledger Stage A+B compiler; C + correctness separately; no unmatched fine-walltime claim','RNG_limitation':'event-keyed resampling RNG and paired initial particles; divergent ancestry changes subsequent trajectory coupling; no stronger pathwise causal claim','production_change':False,'failure':'abort on gate/hash/integrity/resource failure; no automatic scientific retune'}
 for n,x in [('config.json',config),('preregistration.json',prereg),('prompt_split.json',split),('data_split.json',split),('candidate_bank.json',bank),('reference_schedule.json',ref)]:dump(root/n,x)
 shutil.copy2(request,root/'PROTOCOL_REQUEST.txt')
 names=['policy_verification_compiler.py','policy_verification_compiler_tpu.py','seed_coverage_v2.py','adaptive_refinement.py','adaptive_refinement_tpu.py','adapter.py','runner.py','graph_builder.py','policy.py','tpu_reference_compare.py']
 sources={str(Path('scripts/static_bfs_graph')/n):sha(REPO/'scripts/static_bfs_graph'/n) for n in names}
 for n in ['scripts/diffusion_classical_search_t2i/tpu_runner.py','modal_n8_gpu/train_pool.json','configs/static_bfs_graph_t2i.json','configs/diffusion_classical_search_t2i_table12.json']:sources[n]=sha(REPO/n)
 frozen={n:sha(root/n) for n in ['config.json','preregistration.json','prompt_split.json','data_split.json','candidate_bank.json','reference_schedule.json','PROTOCOL_REQUEST.txt']}
 dump(root/'manifest.json',{'sources':sources,'frozen':frozen,'model_revision':old['models']['sd15']['revision'],'parent_bank_ids_verified':True})
 for n in sources:
  p=root/'sources'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/n,p)
 static_gates(root)

def static_gates(root):
 verify(root);c=read(root/'config.json');s=read(root/'data_split.json');bank=read(root/'candidate_bank.json');manual=records()['MANUAL8']
 assert [len(s[k]) for k in 'ABC']==[20,16,24] and len(set(sum([s[k] for k in 'ABC'],[])))==60
 assert set(sum([s[k] for k in 'ABC'],[]))==set(map(int,s['TRAIN_prompts']))
 assert len(set(sum(c['seeds'].values(),[])))==7 and 42 not in sum(c['seeds'].values(),[])
 assert len(bank)==25 and len(v2.union(bank))==32 and any(50 in p['resampling_steps'] for p in bank)
 for p in bank:assert len(p['resampling_steps'])==3 and p['particles']==8 and {t for _,t in p['temperature_by_step']}<=set([2.,8.,32.])
 assert manual=={'kind':'manual','id':'manual_bfs_g0p024_n8','resampling_steps':[20,40,80],'base_temperature':10.,'tempering':'increase','gamma':.024,'particles':8,'selection_mode':'manual'}
 dump(root/'static_gates.json',{'passed':True,'bank25_edges32':True,'original_V2_ids':True,'disjoint20_16_24':True,'fresh_seeds':True,'genuine_manual':manual,'RNG_limitation_documented':True})

def stage(root,name):
 verify(root)
 if (root/'stage_ledger'/f'{name}.json').exists():return
 assert shutil.disk_usage(root).free>5*1024**3
 chips=[0] if name=='gates' else [0,1,2,3];ps=[];fs=[];start=time.time()
 dump(root/'status.json',{'stage':name,'status':'RUNNING','start_unix':start})
 try:
  for chip in chips:
   p=root/'logs'/f'{name}-chip{chip}.log';p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a');fs.append(f)
   env={**os.environ,'PJRT_DEVICE':'TPU','TPU_VISIBLE_CHIPS':str(chip),'PYTHONUNBUFFERED':'1','STATIC_BFS_CACHE_ROOT':f'/dev/shm/static-bfs-policy-verification/chip{chip}/cache'}
   ps.append(subprocess.Popen([sys.executable,'-m','scripts.static_bfs_graph.policy_verification_compiler_tpu','--run-root',str(root),'--stage',name,'--chip',str(chip)],env=env,stdout=f,stderr=subprocess.STDOUT))
  dump(root/'active_workers.json',{'stage':name,'PIDs':[p.pid for p in ps]})
  while True:
   rc=[p.poll() for p in ps]
   if any(x not in (None,0) for x in rc):raise RuntimeError(f'{name} worker failed: {rc}')
   if all(x==0 for x in rc):break
   time.sleep(5)
  for chip in chips:assert read(root/'workers'/name/f'chip{chip}.done.json')['success']
  dump(root/'stage_ledger'/f'{name}.json',{'complete':True,'wall_seconds':time.time()-start,'chips':chips,'end_unix':time.time()})
 finally:
  for p in ps:
   if p.poll() is None:p.terminate()
  for p in ps:
   try:p.wait(timeout=30)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  for f in fs:f.close()

def seal(root,name,data):
 receipt=root/(name+'.receipt.json');target=root/name
 if target.exists():assert read(target)==data and read(receipt)['sha256']==sha(target);return
 dump(target,data);dump(receipt,{'sha256':sha(target),'timestamp_unix':time.time(),'before_dependent_stage':True})

def graph_analysis(root):
 bank=sorted(read(root/'candidate_bank.json'),key=lambda p:p['id']);s=read(root/'data_split.json');c=read(root/'config.json');seed=c['seeds']['A'][0];js=v2.union(bank)
 raw=[read(unitpath(root,'A',pid,seed,j)) for pid in s['A'] for j in js];assert len(raw)==640
 csvout(root/'graph_edge_samples.csv',[{'prompt':r['prompt'],'seed':r['seed'],'edge_id':v2.jobid(r['job']),'src':r['job']['src'],'dst':r['job']['dst'],'tau':r['job']['tau'],'reference_IR':r['reference_reward'],'intervention_IR':r['reward'],'regret':r['regret']} for r in raw])
 x=np.array([[sum(read(unitpath(root,'A',pid,seed,j))['regret'] for j in v2.edges(p)) for p in bank] for pid in s['A']]);idx=np.random.default_rng(c['stat_seed']).integers(0,20,(20000,20));b=x[idx].mean(axis=1);order=np.argsort(b,axis=1,kind='stable');ranks=np.argsort(order,axis=1)+1;freq=np.bincount(order[:,0],minlength=25)/20000
 mu=x.mean(axis=0);se=x.std(axis=0,ddof=1)/np.sqrt(20)
 rows=[{'id':p['id'],'steps':str(p['resampling_steps']),'tau':str(p['temperature_by_step']),'mean_cost':float(mu[i]),'SE':float(se[i]),'robust_cost':float(mu[i]+se[i]),'winner_frequency':float(freq[i]),'mean_bootstrap_rank':float(ranks[:,i].mean()),'rank_CI_low':float(np.quantile(ranks[:,i],.025)),'rank_CI_high':float(np.quantile(ranks[:,i],.975)),'contains50':50 in p['resampling_steps'],'distinct_times':len(set(p['resampling_steps'])),'distinct_taus':len(set(t for _,t in p['temperature_by_step']))} for i,p in enumerate(bank)]
 csvout(root/'graph_candidate_ranking.csv',sorted(rows,key=lambda r:(r['mean_cost'],r['id'])))
 dump(root/'graph_rank_distributions.json',{p['id']:{str(k):int(sum(ranks[:,i]==k)) for k in range(1,26)} for i,p in enumerate(bank)})
 indexes=list(range(25));criteria=[('GRAPH_MEAN',sorted(indexes,key=lambda i:(mu[i],bank[i]['id']))),('GRAPH_BOOTSTRAP',sorted(indexes,key=lambda i:(-freq[i],mu[i],bank[i]['id']))),('GRAPH_ROBUST',sorted(indexes,key=lambda i:(mu[i]+se[i],bank[i]['id']))),('COVERAGE',[i for i in sorted(indexes,key=lambda i:(mu[i],bank[i]['id'])) if 50 in bank[i]['resampling_steps']])]
 picked=[];used=set();category={}
 for name,seq in criteria:
  i=next(i for i in seq if bank[i]['id'] not in used);picked.append({'policy':bank[i],'reasons':[name]});used.add(bank[i]['id']);category[name]=bank[i]
 inc=records()['TRANSFER4_TO_8'];category['INCUMBENT']=inc
 if inc['id'] in used:next(p for p in picked if p['policy']['id']==inc['id'])['reasons'].append('INCUMBENT')
 else:picked.append({'policy':inc,'reasons':['INCUMBENT']})
 assert len(picked)<=5
 seal(root,'frozen_shortlist.json',{'policies':picked,'category_policies':category,'graph_only':category['GRAPH_MEAN'],'candidate_bank_sha256':sha(root/'candidate_bank.json'),'selection_inputs':'Stage A only, no audit or previous rewards'})


def rewards(root,folder,cohort,pols):
 cfg=read(root/'config.json');split=read(root/'data_split.json');ps={p['id']:p for p in pols};raw=[read(p) for p in (root/folder).glob('*.json')];expected=list(itertools.product(split[cohort],cfg['seeds'][cohort],ps))
 look={(r['prompt'],r['seed'],r['policy']['id']):r for r in raw};assert len(raw)==len(look)==len(expected) and set(look)==set(expected)
 hashes={}
 for r in raw:
  assert r['policy']==ps[r['policy']['id']] and len(r['events'])==3 and np.isfinite(r['reward'])
  assert r['reward']==max(r['scores']) and r['selected_particle_id']==int(np.argmax(r['scores'])) and r['NFE']==800
  hashes.setdefault((r['prompt'],r['seed']),set()).add(r['initial_latents_sha256'])
 assert all(len(v)==1 for v in hashes.values())
 with (root/(folder+'_records.jsonl')).open('w') as f:
  for key in expected:f.write(json.dumps(look[key],sort_keys=True)+'\n')
 return {pid:np.array([np.mean([look[(prompt,seed,pid)]['reward'] for seed in cfg['seeds'][cohort]]) for prompt in split[cohort]]) for pid in ps},raw

def summary(root,folder,R,pols):
 csvout(root/(folder+'_summary.csv'),[{'id':p['id'],'steps':str(p['resampling_steps']),'tau':str(p.get('temperature_by_step',p.get('tempering'))),'mean_IR':float(R[p['id']].mean()),'prompt_SE':float(R[p['id']].std(ddof=1)/np.sqrt(len(R[p['id']])))} for p in pols])

def paired(x,seed):
 return v2.bootstrap(np.asarray(x),seed)

def verification_analysis(root):
 sl=read(root/'frozen_shortlist.json');pols=[v['policy'] for v in sl['policies']];R,raw=rewards(root,'verification','B',pols);summary(root,'verification',R,pols);ids=sorted(R);winner=min(ids,key=lambda pid:(-R[pid].mean(),pid));cfg=read(root/'config.json')
 idx=np.random.default_rng(cfg['stat_seed']+1).integers(0,16,(200000,16));x=np.stack([R[p] for p in ids],axis=1);b=x[idx].mean(axis=1);alpha=.05/(len(ids)*(len(ids)-1));bounds=[]
 for i,a in enumerate(ids):
  for j,q in enumerate(ids):
   if a!=q:bounds.append({'selected_id':a,'competitor_id':q,'mean_regret':float((x[:,j]-x[:,i]).mean()),'upper95_simultaneous':float(np.quantile(b[:,j]-b[:,i],1-alpha))})
 csvout(root/'simultaneous_regret_bounds.csv',bounds);certificate=all(r['upper95_simultaneous']<=.02 for r in bounds if r['selected_id']==winner)
 seal(root,'selection_decision.json',{'tentative_winner':next(p for p in pols if p['id']==winner),'status':'EPSILON_COMPETITIVE_CERTIFIED' if certificate else 'UNDERIDENTIFIED','certified':certificate,'epsilon':.02,'adjustment':'Bonferroni across all ordered pairs; percentile prompt bootstrap, approximate coverage','alpha_per_pair':alpha,'bootstrap_replicates':200000,'no_automatic_fallback':True,'production_unchanged':True})
 csvout(root/'verification_pairwise_comparisons.csv',[{'A':a,'B':q,**paired(R[a]-R[q],cfg['stat_seed']+20+i)} for i,(a,q) in enumerate(itertools.combinations(ids,2))])
 fine=next(p for p in read(root/'candidate_bank.json') if p['resampling_steps']==[50,60,80] and p['temperature_by_step']==[[50,8.],[60,2.],[80,32.]])
 named={'VERIFIED':next(p for p in pols if p['id']==winner),'GRAPH_ONLY':sl['graph_only'],'MANUAL8':records()['MANUAL8'],'FINE':fine,'TRANSFER':records()['TRANSFER4_TO_8']}
 for p in pols:named['SHORTLIST_'+p['id']]=p
 seal(root,'frozen_audit_policies.json',named)

def rank_diag(a,b):
 ids=sorted(a);xa=np.array([a[i] for i in ids]);xb=np.array([b[i] for i in ids]);from scipy.stats import spearmanr
 signs=[float(np.sign(xa[i]-xa[j])==np.sign(xb[i]-xb[j])) for i,j in itertools.combinations(range(len(ids)),2)]
 oa=sorted(ids,key=lambda i:(-a[i],i));ob=sorted(ids,key=lambda i:(-b[i],i))
 return {'pairwise_sign_agreement':float(np.mean(signs)),'Spearman':float(spearmanr(xa,xb).statistic),'top1_overlap':oa[0]==ob[0],'top2_overlap_count':len(set(oa[:2])&set(ob[:2]))}

def aggregate(root):
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 verify(root);sl=read(root/'frozen_shortlist.json');d=read(root/'selection_decision.json');named=read(root/'frozen_audit_policies.json');pols=list({p['id']:p for p in named.values()}.values());R,raw=rewards(root,'audit','C',pols);summary(root,'audit',R,pols);short=[p['policy'] for p in sl['policies']];B,_=rewards(root,'verification','B',short);cfg=read(root/'config.json');vid=named['VERIFIED']['id'];gid=named['GRAPH_ONLY']['id'];pairs=[('VERIFIED','GRAPH_ONLY'),('VERIFIED','MANUAL8'),('VERIFIED','FINE'),('VERIFIED','TRANSFER'),('GRAPH_ONLY','MANUAL8'),('GRAPH_ONLY','FINE')]
 contrasts=[{'A':a,'B':b,'identical_policy':named[a]['id']==named[b]['id'],'identical_rollout_count':96 if named[a]['id']==named[b]['id'] else 0,**paired(R[named[a]['id']]-R[named[b]['id']],cfg['stat_seed']+100+i)} for i,(a,b) in enumerate(pairs)];csvout(root/'paired_comparisons.csv',contrasts)
 ids=[p['id'] for p in short];best=min(ids,key=lambda i:(-R[i].mean(),i));auditorder=sorted(ids,key=lambda i:(-R[i].mean(),i));oracle={'audit_best_in_shortlist':best,'verified':vid,'point_selection_regret':float(R[best].mean()-R[vid].mean()),'verified_in_audit_top2':vid in auditorder[:2],'interpretation':'descriptive hindsight maximization over frozen shortlist; no unbiased oracle CI; no deployment change'};csvout(root/'shortlist_oracle_analysis.csv',[oracle])
 import csv
 with (root/'graph_candidate_ranking.csv').open() as f:grows={r['id']:r for r in csv.DictReader(f)}
 ga={i:-float(grows[i]['mean_cost']) for i in ids};ba={i:float(B[i].mean()) for i in ids};ca={i:float(R[i].mean()) for i in ids};diag={'graph_vs_audit':rank_diag(ga,ca),'verification_vs_audit':rank_diag(ba,ca),'small_sample_policy_count':len(ids),'unbiased_generalization_claim':False};dump(root/'ranking_diagnostics.json',diag)
 curve=[]
 for n in [4,8,12,16]:
  means={i:float(B[i][:n].mean()) for i in ids};w=min(ids,key=lambda i:(-means[i],i));order=sorted(ids,key=lambda i:(-means[i],i))
  for i in ids:curve.append({'prefix_prompts':n,'id':i,'mean_IR':means[i],'rank':order.index(i)+1,'winner':w,'different_from_final':w!=vid})
 csvout(root/'verification_learning_curve.csv',curve)
 ledgers=[read(p) for p in (root/'work_ledgers').glob('*.json')];account={}
 for stage_name in ['gates','graph','verification','audit']:
  rows=[r for r in ledgers if r['stage']==stage_name];stagework={k:sum(r[k] for r in rows) for k in ['actual_denoising_NFE','actual_verifier_candidate_scores','actual_verifier_batch_calls','reference_trajectories','logical_observations_new']};stagework['TPU_chip_hours']=sum(r['wall_seconds_including_load'] for r in rows)/3600;stagework['wall_seconds']=read(root/'stage_ledger'/f'{stage_name}.json')['wall_seconds'];stagework['full_policy_rollouts']=len(list((root/stage_name).glob('*.json'))) if stage_name in ['verification','audit'] else 0;account[stage_name]=stagework
 keys=['actual_denoising_NFE','actual_verifier_candidate_scores','actual_verifier_batch_calls','TPU_chip_hours','wall_seconds']
 account['total_compiler']={k:account['graph'][k]+account['verification'][k] for k in keys};account['total_experiment']={k:sum(account[s][k] for s in ['gates','graph','verification','audit']) for k in keys};account['actual_billing']='UNKNOWN';account['matched_fine_graph_runtime_savings']='NOT ESTABLISHED; historical 2900 observations not cost matched';dump(root/'compute_accounting.json',account)
 fig,ax=plt.subplots(figsize=(8,4))
 for i in ids:
  rankings=[[j for j in sorted(ids,key=lambda j:(-a[j],j))].index(i)+1 for a in [ga,ba,ca]];ax.plot([0,1,2],rankings,'o-',label=i[-6:])
 ax.invert_yaxis();ax.set_xticks([0,1,2],['A: graph','B: verification','C: audit']);ax.set_ylabel('Shortlist rank');ax.legend();fig.tight_layout();fig.savefig(root/'graph_vs_policy_rank.png');plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4));cs=contrasts[:4]
 for i,c in enumerate(cs):ax.plot([c['CI_low'],c['CI_high']],[i,i],'-');ax.plot(c['delta'],i,'o')
 ax.axvline(0,color='gray',ls='--');ax.set_yticks(range(4),[c['A']+' − '+c['B'] for c in cs]);ax.set_xlabel('Paired audit ΔIR, 95% CI');fig.tight_layout();fig.savefig(root/'paired_audit_deltas.png');plt.close(fig)
 fig,ax=plt.subplots();ax.scatter([account['graph']['TPU_chip_hours'],account['total_compiler']['TPU_chip_hours']],[float(R[gid].mean()),float(R[vid].mean())]);
 for label,x,y in [('Graph-only',account['graph']['TPU_chip_hours'],R[gid].mean()),('Graph+verification',account['total_compiler']['TPU_chip_hours'],R[vid].mean())]:ax.annotate(label,(x,y))
 ax.set(xlabel='Measured compiler TPU-chip-hours (audit excluded)',ylabel='Audit ImageReward');fig.tight_layout();fig.savefig(root/'quality_vs_compiler_cost.png');plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(10,4))
 for i in ids:
  rows=[r for r in curve if r['id']==i];axs[0].plot([r['prefix_prompts'] for r in rows],[r['mean_IR'] for r in rows],'o-',label=i[-6:]);axs[1].plot([r['prefix_prompts'] for r in rows],[r['rank'] for r in rows],'o-')
 axs[0].set(xlabel='B prompts, both seeds',ylabel='Mean IR');axs[1].set(xlabel='B prompts, both seeds',ylabel='Rank');axs[1].invert_yaxis();axs[0].legend();fig.tight_layout();fig.savefig(root/'verification_learning_curve.png');plt.close(fig)
 primary=contrasts[0];outcome='A' if vid!=gid and primary['CI_low']>0 else 'B' if vid==gid else 'F'
 # A is performance evidence only; unmatched fine baseline does not establish lightweight savings.
 morning={'Correctness gates':'PASS','graph':{'prompts':20,'seeds':1,'edge_observations':640,'TPU_chip_hours':account['graph']['TPU_chip_hours']},'shortlisted_policies':sl['policies'],'verification':{'rollouts':len(ids)*32,'TPU_chip_hours':account['verification']['TPU_chip_hours'],'tentative_winner':named['VERIFIED'],'epsilon_certificate':d['certified'],'selection_status':d['status']},'frozen_audit':{n:float(R[p['id']].mean()) for n,p in named.items() if not n.startswith('SHORTLIST')},'paired_comparisons':contrasts,'shortlist_oracle':oracle,'total_compiler':account['total_compiler'],'final_outcome':outcome}
 report='# POLICY-LEVEL VERIFICATION COMPILER\n\n'+json.dumps(morning,indent=2)+'\n\n## H1–H4\n\nH1: '+('Zero selection change; verification added cost.' if vid==gid else 'See paired primary CI; no positive claim if it crosses zero.')+'\n\nH2: Small-policy-count ranking diagnostics:\n'+json.dumps(diag,indent=2)+'\n\nH3: '+d['status']+'; all ordered-pair Bonferroni bootstrap bounds are approximate, not finite-sample guarantees. Audit is not used to certify the Stage B choice.\n\nH4: Measured compiler and audit costs are separate. Matched fine-graph wall-clock savings have NOT been established. Actual billing UNKNOWN.\n\nAll prompts historically explored TRAIN; campaign A/B/C disjoint, fresh paired seeds. This is internal diagnosis, not external confirmation. Fine/Manual are audit-only baselines. Initial noise is paired; divergent ancestry and RNG consumption limit causal trajectory claims. Production unchanged. Outcome F covers uncertainty; do not infer equivalence from a nonsignificant contrast. Learning curves are retrospective, not sequential-valid inference.\n'
 (root/'REPORT.md').write_text(report)
 dump(root/'INTEGRITY.json',{'passed':True,'graph_rows':640,'verification_rows':len(ids)*32,'audit_rows':len(pols)*96,'paired_initial_hashes':True,'frozen_shortlist_sha256':sha(root/'frozen_shortlist.json'),'audit_policies_sha256':sha(root/'frozen_audit_policies.json'),'no_production_change':True})
 for name in ['frozen_shortlist.json','selection_decision.json','frozen_audit_policies.json']:assert read(root/(name+'.receipt.json'))['sha256']==sha(root/name)
 verify(root)

def run(root,request,setup_only=False):
 root.mkdir(parents=True,exist_ok=True);lock=(root/'driver.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 freeze(root,request)
 if setup_only:return
 if (root/'status.json').exists() and read(root/'status.json').get('stage')=='DONE':return
 if not (root/'driver.pid.json').exists():dump(root/'driver.pid.json',{'pid':os.getpid(),'started_unix':time.time()})
 try:
  stage(root,'gates');assert read(root/'correctness_gates.json')['passed'];stage(root,'graph');graph_analysis(root)
  stage(root,'verification');verification_analysis(root);stage(root,'audit');aggregate(root)
  dump(root/'status.json',{'stage':'DONE','all_stages_complete':True,'end_unix':time.time()})
 except BaseException as e:dump(root/'status.json',{'stage':'ABORTED','exception':repr(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--request',type=Path,required=True);p.add_argument('--setup-only',action='store_true');a=p.parse_args();run(a.run_root.resolve(),a.request.resolve(),a.setup_only)
