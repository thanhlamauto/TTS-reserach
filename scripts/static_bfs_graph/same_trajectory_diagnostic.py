"""Frozen same-trajectory edge/full-policy diagnostic, existing TPU replay backend."""
import json,hashlib,csv,time,os,sys,subprocess,fcntl,traceback,itertools,shutil
from pathlib import Path
import numpy as np
from .adaptive_refinement import read,dump,sha,csvout,job,jobid,unitpath,intervention
from .graph_builder import path_edges
REPO=Path(__file__).resolve().parents[2]
OLD=Path('/home/thanhlamtba31/static-bfs-dcs/experiments/sequential_graph_refinement_v1')
SEED=2026100901
SOURCE=['same_trajectory_diagnostic.py','adaptive_refinement.py','adaptive_refinement_tpu.py','adapter.py','runner.py','policy.py','graph_builder.py','tpu_reference_compare.py']
def verify(root):
 m=read(root/'manifest.json');assert all(sha(REPO/n)==h for n,h in m['source_sha256'].items())
 assert all(sha(root/n)==h for n,h in m['frozen'].items())
def edges(p):return [job(e.src,e.dst,e.tau_dst) for e in path_edges(tuple(p['resampling_steps']),tuple(t for s,t in p['temperature_by_step']))]
def freeze(root):
 root.mkdir(parents=True,exist_ok=True)
 if (root/'manifest.json').exists():verify(root);return
 from .tpu_reference_compare import checked_inputs
 pool,cfg,base=checked_inputs();old=read(OLD/'frozen_policies.json');pol={n:old[k] for n,k in [('NEW','ADAPTIVE_NEW'),('COARSE','P_COARSE'),('TRANSFER','P_TRANSFER')]}
 ids=read(OLD/'dataset_split.json')['A'][:12];assert len(ids)==12;ref=read(OLD/'reference_schedule.json');assert ref['resampling_steps']==[10,20,30,40,60,80,90]
 js=sorted({jobid(j):j for p in pol.values() for j in edges(p)}.values(),key=jobid);assert len(js)==9
 config={'protocol':'SAME_TRAJECTORY_EDGE_POLICY_DIAGNOSTIC_V1','N':8,'K':3,'steps':100,'eta':1.,'dtype':'BF16','taus':[2,8,32],'seeds':{'A':[SEED],'B':[],'C':[SEED]},'snapshots':list(range(5,100,5)),'reference':ref,'bootstrap_replicates':50000,'stat_seed':2026100910,'practical_residual_margin':.02,'residual_interpretation':'execution-protocol dependent; not pure selection causal effect','paired_definition':'identical initial particles, prompt, seed, common reference reward; not identical full trajectories after intervention','history_exposure':'all prompts historically explored; seed fresh; internal TRAIN diagnostic','selection':'three previously frozen policies; no new ranking/optimization','stop':'all12 units; no reward-dependent early stop'}
 data={'A':ids,'B':[i for i in pool['train_indices'] if i not in ids],'C':ids,'TRAIN_prompts':pool['prompts'],'same_prompt_seed_edges_and_policies':True,'historically_explored':True}
 prereg={'pair_order':[['NEW','COARSE'],['NEW','TRANSFER'],['COARSE','TRANSFER']],'residual':'epsilon=(R_P-R_Q)+sum(w_P)-sum(w_Q)','per_policy_residual':'(IR_reference-IR_policy)-sum(w_edges)','inference':'50000 joint prompt bootstrap; 12prompt onefreshseed; pointwise descriptive95CI; do not claim absence ofmismatch from non-significance','small_residual':'95CI entirely within [-.02,+.02] descriptive precision check','systematic_residual':'95CI excludes0, with magnitude discussed; exploratory multiple comparisons','large_residual':'95CI entirely above+.02 or below-.02','old_vs_fresh':'paired same12 prompts with different seeds; not independent prompt-shift experiment; old selected policy winner curse persists in old data','next_actions':'only recommend; no compiler/fallback/backend changes','smoke':'reference snapshoton/off all8, two independent edge comparisons, initialparticle hash agreement; first measuredunit retained in fullrun','full12_missing_unit_only_resume':True,'extra_smoke_execution_counted_separately':True}
 for name,x in [('config.json',config),('preregistration.json',prereg),('data_split.json',data),('dataset_split.json',data),('reference_schedule.json',ref),('frozen_policies.json',pol),('edge_jobs.json',js)]:dump(root/name,x)
 src={str((REPO/'scripts/static_bfs_graph'/n).relative_to(REPO)):sha(REPO/'scripts/static_bfs_graph'/n) for n in SOURCE}
 for n in ['scripts/diffusion_classical_search_t2i/tpu_runner.py','configs/static_bfs_graph_t2i.json','configs/diffusion_classical_search_t2i_table12.json','modal_n8_gpu/train_pool.json']:src[n]=sha(REPO/n)
 dump(root/'manifest.json',{'source_sha256':src,'frozen':{n:sha(root/n) for n in ['config.json','preregistration.json','dataset_split.json','data_split.json','reference_schedule.json','frozen_policies.json','edge_jobs.json']},'parent_run':str(OLD),'parent_policy_sha256':sha(OLD/'frozen_policies.json'),'parent_gates_sha256':sha(OLD/'gates.json'),'model_revision':base['models']['sd15']['revision']})
 (root/'commands.md').write_text('cd '+str(REPO)+'\n'+sys.executable+' -m scripts.static_bfs_graph.same_trajectory_diagnostic --run-root '+str(root)+'\n')
 source=root/'sources'
 for n in src:
  dst=source/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/n,dst)

def fullpath(root,pid,seed,p):return root/'full_policies'/f'p{pid}-s{seed}-{p["id"]}.json'
def complete(root,pid,seed):return all(unitpath(root,'A',pid,seed,j).exists() for j in read(root/'edge_jobs.json')) and all(fullpath(root,pid,seed,p).exists() for p in read(root/'frozen_policies.json').values())
def unit(eng,pid,seed):
 root=eng.root
 if complete(root,pid,seed):return
 jobs=read(root/'edge_jobs.json');eng.profile('A',pid,seed,jobs);ref=read(root/'references/A'/f'p{pid}-s{seed}.json')
 for p in read(root/'frozen_policies.json').values():
  path=fullpath(root,pid,seed,p)
  if path.exists():continue
  started=time.time();row,_=eng.direct(pid,seed,p);assert row['initial_latents_sha256']==ref['initial_latents_sha256'];assert len(row['events'])==3
  dump(path,{'prompt':pid,'seed':seed,'policy':p,'runtime_seconds':time.time()-started,**row});eng.persist()
def worker(root,chip,smoke=False):
 from . import adaptive_refinement_tpu as backend
 backend.verify=verify;stage='smoke' if smoke else 'measure';eng=backend.Engine(root,chip,stage)
 try:
  ids=read(root/'dataset_split.json')['A']
  if smoke:
   pid=ids[0];before=dict(eng.account);unit(eng,pid,SEED);measurement=dict(eng.account)
   ref=read(root/'references/A'/f'p{pid}-s{SEED}.json');off,_=eng.direct(pid,SEED,eng.ref,False);eng.account['reference_trajectories']+=1
   keys=['scores','RGB_hashes','latent_hash','events','initial_latents_sha256'];assert all(off[k]==ref[k] for k in keys),'snapshot gate mismatch'
   jobs=read(root/'edge_jobs.json')
   selected=[next(j for j in jobs if j['src']==20),next(j for j in jobs if j['dst']==100)]
   for j in selected:
    p=backend.static_record(intervention(root,j['src'],j['dst'],j['tau']));independent,_=eng.direct(pid,SEED,p)
    cached=read(unitpath(root,'A',pid,SEED,j));assert all(independent[k]==cached[k] for k in ['scores','RGB_hashes','latent_hash','events']),'fresh independent edge gate failed'
   extra={k:eng.account[k]-measurement[k] for k in ['actual_denoising_NFE','actual_verifier_candidate_scores','actual_verifier_batch_calls','reference_trajectories']}
   dump(root/'smoke_gates.json',{'passed':True,'reference_snapshot_on_off_all8_exact':True,'two_independent_edges_all8_exact':True,'all3policy_initial_particles_exact':True,'previous8replaygates_passed':read(OLD/'gates.json')['passed'],'measurement_unit_reused':True,'extra_smoke_work':extra,'RNG_guarantee':read(root/'config.json')['paired_definition']})
  else:
   for pid in ids[chip::4]:unit(eng,pid,SEED)
  eng.persist();dump(root/'workers'/stage/f'chip{chip}.done.json',{'success':True,'account':eng.account})
 except BaseException as e:
  eng.persist();dump(root/'failures'/f'{stage}-chip{chip}.json',{'exception':repr(e),'traceback':traceback.format_exc()});raise

def bootstrap(x,cfg,idx=None):
 if idx is None:idx=np.random.default_rng(cfg['stat_seed']).integers(0,len(x),(cfg['bootstrap_replicates'],len(x)))
 m=x[idx].mean(axis=1);lo,hi=np.quantile(m,[.025,.975]);margin=cfg['practical_residual_margin']
 return {'mean':float(x.mean()),'SE':float(x.std(ddof=1)/np.sqrt(len(x))),'CI_low':float(lo),'CI_high':float(hi),'within_margin':bool(lo> -margin and hi<margin),'CI_excludes_zero':bool(lo>0 or hi<0),'CI_beyond_margin':bool(lo>margin or hi< -margin)}
def finalize(root):
 cfg=read(root/'config.json');ids=read(root/'dataset_split.json')['A'];pol=read(root/'frozen_policies.json');jobs=read(root/'edge_jobs.json');assert all(complete(root,p,SEED) for p in ids)
 assert len(list((root/'observations/A').glob('*/*.json')))==108;assert len(list((root/'full_policies').glob('*.json')))==36
 refs={p:read(root/'references/A'/f'p{p}-s{SEED}.json') for p in ids};E={jobid(j):np.array([read(unitpath(root,'A',p,SEED,j))['regret'] for p in ids]) for j in jobs};C={n:sum((E[jobid(j)] for j in edges(p)),np.zeros(12)) for n,p in pol.items()};R={n:np.array([read(fullpath(root,pid,SEED,p))['reward'] for pid in ids]) for n,p in pol.items()}
 refvec=np.array([refs[p]['reward'] for p in ids]);idx=np.random.default_rng(cfg['stat_seed']).integers(0,12,(50000,12));pairrows=[];raw=[];residuals={}
 for k,(a,b) in enumerate(read(root/'preregistration.json')['pair_order']):
  actual=R[a]-R[b];pred=-(C[a]-C[b]);err=actual-pred;residuals[a+'-'+b]=err
  stats={prefix+'_'+name:v for prefix,x in [('actual',actual),('predicted',pred),('residual',err)] for name,v in bootstrap(x,cfg,idx).items()}
  pairrows.append({'A':a,'B':b,**stats,'actual_predicted_sign_agree':bool(np.sign(actual.mean())==np.sign(pred.mean()))})
  for i,pid in enumerate(ids):raw.append({'prompt':pid,'seed':SEED,'A':a,'B':b,'reference_reward':refs[pid]['reward'],'actual_delta':float(actual[i]),'predicted_delta':float(pred[i]),'residual':float(err[i])})
 csvout(root/'paired_residual_summary.csv',pairrows);csvout(root/'paired_residual_raw.csv',raw)
 policyrows=[];perpol=[]
 for n in pol:
  delta=(refvec-R[n])-C[n];perpol.append({'policy':n,'mean_IR':float(R[n].mean()),'mean_predicted_regret':float(C[n].mean()),'mean_actual_regret':float((refvec-R[n]).mean()),**{'residual_'+k:v for k,v in bootstrap(delta,cfg,idx).items()}})
  for i,pid in enumerate(ids):policyrows.append({'prompt':pid,'seed':SEED,'policy':n,'reference_reward':float(refvec[i]),'reward':float(R[n][i]),'edge_sum':float(C[n][i]),'actual_regret':float(refvec[i]-R[n][i]),'actual_regret_minus_edge_sum':float(delta[i])})
 csvout(root/'policy_summary.csv',perpol);csvout(root/'policy_raw.csv',policyrows)
 # Old/fresh comparisons are matched prompts with different seeds; record that limit.
 oldcfg=read(OLD/'config.json');oldseed=oldcfg['seeds']['A'][0];edgechange=[]
 for j in jobs:
  old=np.array([read(unitpath(OLD,'A',pid,oldseed,j))['regret'] for pid in ids]);new=E[jobid(j)];change=new-old
  edgechange.append({**j,'edge_id':jobid(j),'old_seed':oldseed,'fresh_seed':SEED,'old_same12_mean':float(old.mean()),'fresh_same12_mean':float(new.mean()),**{'change_'+k:v for k,v in bootstrap(change,cfg,idx).items()}})
 csvout(root/'old_fresh_edge_comparison.csv',edgechange)
 oldnew=[]
 for a,b in read(root/'preregistration.json')['pair_order']:
  oldcost={n:sum((np.array([read(unitpath(OLD,'A',pid,oldseed,j))['regret'] for pid in ids]) for j in edges(pol[n])),np.zeros(12)) for n in [a,b]}
  oldpred=-(oldcost[a]-oldcost[b]);freshpred=-(C[a]-C[b]);oldnew.append({'A':a,'B':b,'old_same12_predicted_mean':float(oldpred.mean()),'fresh_same12_predicted_mean':float(freshpred.mean()),**{'change_'+k:v for k,v in bootstrap(freshpred-oldpred,cfg,idx).items()}})
 csvout(root/'old_fresh_path_comparison.csv',oldnew)
 ledgers=[read(p) for p in (root/'work_ledgers').glob('*.json')];work={k:sum(d[k] for d in ledgers) for k in ['actual_denoising_NFE','actual_verifier_candidate_scores','actual_verifier_batch_calls','reference_trajectories','logical_observations_new']};work['TPU_chip_hours_including_load']=sum(d['wall_seconds_including_load'] for d in ledgers)/3600;work['extra_smoke']=read(root/'smoke_gates.json')['extra_smoke_work'];work['nominal_requested_NFE']=84768;work['actual_billing']='UNKNOWN';work['wall_seconds']=time.time()-read(root/'driver.pid.json')['started_unix'];dump(root/'compute_accounting.json',work)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(9,4));xs=range(3);means=[r['residual_mean'] for r in pairrows];ax.errorbar(xs,means,yerr=[[r['residual_mean']-r['residual_CI_low'] for r in pairrows],[r['residual_CI_high']-r['residual_mean'] for r in pairrows]],fmt='o');ax.axhspan(-.02,.02,alpha=.15,color='green');ax.axhline(0,c='gray');ax.set_xticks(list(xs),[r['A']+'-'+r['B'] for r in pairrows]);ax.set_ylabel('Same-trajectory actual minus additive predicted delta');fig.tight_layout();fig.savefig(root/'residual_summary.png');plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(12,4))
 for ax,r in zip(axs,pairrows):
  x=-(C[r['A']]-C[r['B']]);y=R[r['A']]-R[r['B']];lo=min(x.min(),y.min());hi=max(x.max(),y.max());ax.scatter(x,y);ax.plot([lo,hi],[lo,hi],'--');ax.set(title=r['A']+'-'+r['B'],xlabel='Predicted delta',ylabel='Actual delta')
 fig.tight_layout();fig.savefig(root/'predicted_actual_same_trajectory.png');plt.close(fig)
 outcome='SYSTEMATIC_EXECUTION_PROTOCOL_RESIDUAL' if any(r['residual_CI_beyond_margin'] for r in pairrows) else 'SMALL_PAIRED_RESIDUAL_WITHIN_MARGIN' if all(r['residual_within_margin'] for r in pairrows) else 'UNDERIDENTIFIED_PRECISION'
 dump(root/'decision.json',{'outcome':outcome,'descriptive_only':True,'same_trajectory_pairs':pairrows,'no_production_change':True,'not_selection_only_causal_claim':True})
 report=['# SAME-TRAJECTORY EDGE VERSUS FULL-POLICY DIAGNOSTIC','Correctness gates: PASS. 12 historically explored TRAIN prompts, fresh seed '+str(SEED)+'. 108 edge observations, 36 full-policy runs and 12 reference trajectories. No new graph fit or policy selection.','Outcome: '+outcome,'Pairwise residual epsilon = (R_P-R_Q) + C_P-C_Q. Shared reference and shared edges cancel. Each pointwise95CI uses50000joint prompt bootstrap resamples of12units; multiple pairs are descriptive, not confirmatory.','## Same-trajectory contrasts\n```json\n'+json.dumps(pairrows,indent=2)+'\n```','## Per-policy residuals\n```json\n'+json.dumps(perpol,indent=2)+'\n```','## Old versus fresh measurements\nSame12prompts, differentseeds. This probes seed sensitivity; it does not identify between-prompt distribution shift. Old selected-policy estimates are subject to selection optimism.\n```json\n'+json.dumps(oldnew,indent=2)+'\n```','## Accounting\n```json\n'+json.dumps(work,indent=2)+'\n```','## Interpretation limits\nResiduals are a direct discrepancy between the current additive predictor and full policies on matching prompt-seed initial particles. Schedule-dependent execution/RNG after divergence remains part of this endpoint protocol, so a nonzero residual is not clean evidence of pure selection-history interaction alone. With12units, failure to excludezero is not evidence of negligible residual. Do not redesign the estimator from a noisy point estimate. No automatic compiler, action-grid or fallback update.']
 (root/'REPORT.md').write_text('\n\n'.join(report));dump(root/'INTEGRITY.json',{'passed':True,'reference_rows':12,'edge_rows':108,'policy_rows':36,'same_trajectory_common_baseline':True,'gates_passed':True,'policy_sha256':sha(root/'frozen_policies.json')})

def stage(root,smoke):
 name='smoke' if smoke else 'measure'
 if (root/'stage_ledger'/f'{name}.json').exists():return
 verify(root);assert shutil.disk_usage(root).free>5*1024**3
 chips=[0] if smoke else list(range(4));procs=[];logs=[];start=time.time();dump(root/'status.json',{'stage':name,'status':'RUNNING','start_unix':start})
 try:
  for chip in chips:
   path=root/'logs'/f'{name}-chip{chip}.log';path.parent.mkdir(parents=True,exist_ok=True);f=path.open('a');logs.append(f)
   cmd=[sys.executable,'-m','scripts.static_bfs_graph.same_trajectory_diagnostic','--run-root',str(root),'--worker','--chip',str(chip)]+(['--smoke'] if smoke else [])
   env={**os.environ,'PJRT_DEVICE':'TPU','TPU_VISIBLE_CHIPS':str(chip),'PYTHONUNBUFFERED':'1','STATIC_BFS_CACHE_ROOT':f'/dev/shm/static-bfs-same-trajectory-v1/chip{chip}/cache'}
   procs.append(subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT))
  dump(root/'active_workers.json',{'stage':name,'PIDs':[p.pid for p in procs]})
  while True:
   statuses=[p.poll() for p in procs]
   if any(s not in (None,0) for s in statuses):raise RuntimeError(f'{name} worker failure: {statuses}')
   if all(s==0 for s in statuses):break
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
def run(root):
 root.mkdir(parents=True,exist_ok=True);lock=(root/'driver.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (root/'status.json').exists() and read(root/'status.json').get('stage')=='DONE':return
 if not (root/'driver.pid.json').exists():dump(root/'driver.pid.json',{'pid':os.getpid(),'started_unix':time.time()})
 try:
  freeze(root);stage(root,True);assert read(root/'smoke_gates.json')['passed'];stage(root,False);finalize(root);verify(root);dump(root/'status.json',{'stage':'DONE','all_stages_complete':True,'end_unix':time.time()})
 except BaseException as e:dump(root/'status.json',{'stage':'ABORTED','exception':repr(e),'traceback':traceback.format_exc(),'end_unix':time.time()});raise
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--worker',action='store_true');p.add_argument('--chip',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.worker:worker(a.run_root.resolve(),a.chip,a.smoke)
 else:run(a.run_root.resolve())
