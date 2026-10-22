"""Offline frozen inference; average4 seeds within prompt before resampling inference."""
from pathlib import Path
import json,csv,itertools,math,time
import numpy as np

PRIMARY=(20,40,60,80,90)
ANCHORS=('NO_SEARCH','DENSE_REFERENCE','DENSE10')
TARGETS=('TRANSFER4_TO_8','MANUAL8','RETUNED8')
METRICS=('ESS','effective_ancestors','max_weight')

def analyze(root):
 plan=json.loads((root/'config.json').read_text());policies=json.loads((root/'policy_configs.json').read_text())
 assert json.loads((root/'smoke_gate.json').read_text())['passed']
 units=[json.loads(p.read_text()) for p in sorted((root/'full').glob('chip*/units/*.json'))]
 assert len(units)==1440
 ids=plan['prompt_ids'];seeds=plan['seeds'];names=list(policies);ts=plan['probe_indices']
 U={(u['prompt_id'],u['seed'],u['policy']):u for u in units};assert len(U)==1440
 probes=[p for u in units for p in u['probes']];assert len(probes)==12960
 P={(p['prompt_id'],p['seed'],p['policy'],p['t']):p for p in probes};assert len(P)==12960
 for pid in ids:
  for seed in seeds:
   assert len({U[pid,seed,m]['initial_latents_sha256'] for m in names})==1
   for m in names:
    u=U[pid,seed,m];assert u['status']=='success' and u['diffusion_NFE']==800 and len(u['probes'])==9
    assert [e['sampling_index'] for e in u['actual_events']]==policies[m]['resampling_steps']
    for p in u['probes']:
     assert p['probe_tau']==8 and len(p['raw_scores'])==8
     w=np.array(p['probe_weights']);assert np.isclose(w.sum(),1)
     v=8*np.array(p['raw_scores']);cw=np.exp(v-v.max());cw/=cw.sum();assert np.allclose(w,cw,rtol=1e-12,atol=1e-12)
     assert np.isclose(sum(p['ancestor_shares']),1)
 def csvout(name,rows):
  with (root/name).open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 (root/'raw_probe_records.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in probes))
 csvout('terminal_rewards.csv',[{k:u[k] for k in ['prompt_id','seed','policy','terminal_ImageReward']} for u in units])
 rng=np.random.default_rng(plan['stat_rng_seed']);indices=rng.integers(0,60,size=(plan['bootstrap_replicates'],60))
 def CI(x):return np.quantile(np.asarray(x)[indices].mean(axis=1),[.025,.975]).tolist()
 def test(x):
  x=np.asarray(x);mean=float(x.mean());flip=[]
  for _ in range(100):
   signs=rng.integers(0,2,size=(1000,60))*2-1;flip.extend((signs*x).mean(axis=1))
  flip=np.asarray(flip);ci=CI(x)
  return {'mean':mean,'SE_across_prompts':float(x.std(ddof=1)/np.sqrt(60)),'CI95':ci,'sign_flip_p_one_sided_negative':float((1+sum(flip<=mean))/(len(flip)+1)),'sign_flip_p_two_sided':float((1+sum(abs(flip)>=abs(mean)))/(len(flip)+1)),'wins_ties_losses_negative_is_win':[int(sum(x<0)),int(sum(x==0)),int(sum(x>0))],'prompt_count':60,'bootstrap_replicates':len(indices),'sign_flips':len(flip),'verdict':'YES' if ci[1]<0 else 'NO' if ci[0]>0 else 'UNDERIDENTIFIED'}
 def distances(a,b,t):
  metrics=[[],[],[],[]]
  for pid in ids:
   vals=[]
   for seed in seeds:
    x=P[pid,seed,a,t];y=P[pid,seed,b,t]
    vals.append([.5*np.sum(np.abs(np.array(x['sorted_probe_weights'])-y['sorted_probe_weights'])),abs(x['ESS']-y['ESS'])/8,abs(x['normalized_weight_entropy']-y['normalized_weight_entropy']),.5*np.sum(np.abs(np.array(x['ancestor_shares'])-y['ancestor_shares']))])
   v=np.mean(vals,axis=0)
   for i,z in enumerate(v):metrics[i].append(z)
  return np.array(metrics)
 D={(a,b,t):distances(a,b,t) for a in ANCHORS for b in TARGETS for t in ts}
 ds=[]
 for target in TARGETS:
  for anchor in ANCHORS:
   value=np.mean([D[anchor,target,t] for t in PRIMARY],axis=0)
   ds.append({'target':target,'anchor':anchor,'D_weight':float(value[0].mean()),'D_ESS':float(value[1].mean()),'D_entropy':float(value[2].mean()),'D_ancestry':float(value[3].mean()),'D_weight_CI_low':CI(value[0])[0],'D_weight_CI_high':CI(value[0])[1]})
 csvout('distance_summary.csv',ds)
 def meanprimary(a,b,k=0):return np.mean([D[a,b,t][k] for t in PRIMARY],axis=0)
 primary=test(meanprimary('DENSE_REFERENCE','TRANSFER4_TO_8')-meanprimary('NO_SEARCH','TRANSFER4_TO_8'))
 primary.update({'mean_DenseRef_to_Transfer':float(meanprimary('DENSE_REFERENCE','TRANSFER4_TO_8').mean()),'mean_NoSearch_to_Transfer':float(meanprimary('NO_SEARCH','TRANSFER4_TO_8').mean()),'Delta_definition':'DenseRef minus NoSearch; average4seeds per prompt then five checkpoints','primary_target':'high-performing frozen TRANSFER4_TO_8; not claimed optimal'})
 dense10=test(meanprimary('DENSE10','TRANSFER4_TO_8')-meanprimary('DENSE_REFERENCE','TRANSFER4_TO_8'))
 secondary={target:test(meanprimary('DENSE_REFERENCE',target)-meanprimary('NO_SEARCH',target)) for target in TARGETS[1:]}
 secondary['Dense10_vs_DenseRef_for_Transfer']=dense10
 (root/'primary_anchor_test.json').write_text(json.dumps(primary,indent=2));(root/'secondary_anchor_tests.json').write_text(json.dumps(secondary,indent=2))
 checkpoints=[]
 for t in ts:
  delta=D['DENSE_REFERENCE','TRANSFER4_TO_8',t][0]-D['NO_SEARCH','TRANSFER4_TO_8',t][0];lo,hi=CI(delta)
  checkpoints.append({'t':t,'nominal_progress':t/100,'D_DenseRef_Transfer':float(D['DENSE_REFERENCE','TRANSFER4_TO_8',t][0].mean()),'D_NoSearch_Transfer':float(D['NO_SEARCH','TRANSFER4_TO_8',t][0].mean()),'D_Dense10_Transfer':float(D['DENSE10','TRANSFER4_TO_8',t][0].mean()),'DenseRef_minus_NoSearch':float(delta.mean()),'CI_low':lo,'CI_high':hi,'interpretation':'descriptive profile; not separate hypothesis tests'})
 csvout('checkpoint_summary.csv',checkpoints)
 profiles=[]
 for m in names:
  for t in ts:
   item={'policy':m,'t':t,'nominal_progress':t/100}
   for key in ['score_mean','score_std','score_range','top1_top2_gap','ESS','max_weight','normalized_weight_entropy','surviving_ancestors','effective_ancestors','max_ancestry_share','pairwise_normalized_L2','pairwise_cosine_distance']:
    x=np.array([np.mean([P[pid,seed,m,t][key] for seed in seeds]) for pid in ids]);lo,hi=CI(x);item[key]=float(x.mean());item[key+'_CI_low']=lo;item[key+'_CI_high']=hi
   profiles.append(item)
 csvout('population_profiles.csv',profiles)
 quality=[]
 for m in names:
  dist=float(np.mean([distances(m,'TRANSFER4_TO_8',t)[0] for t in PRIMARY]));reward=np.mean([np.mean([U[pid,seed,m]['terminal_ImageReward'] for seed in seeds]) for pid in ids])
  quality.append({'policy':m,'mean_terminal_IR':float(reward),'mean_D_weight_to_Transfer':dist})
 from scipy.stats import spearmanr
 transfer=next(x['mean_terminal_IR'] for x in quality if x['policy']=='TRANSFER4_TO_8')
 for x in quality:x['terminal_IR_minus_Transfer']=x['mean_terminal_IR']-transfer
 rho=spearmanr([x['mean_D_weight_to_Transfer'] for x in quality],[x['terminal_IR_minus_Transfer'] for x in quality])
 exploratory={'policy_count':6,'policies':quality,'Spearman_rho':float(rho.statistic),'p_value':float(rho.pvalue),'known_endpoint_RNG_confound':True,'causal_claim':False}
 (root/'exploratory_quality.json').write_text(json.dumps(exploratory,indent=2))
 import matplotlib
 matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(9,5))
 for a in ANCHORS:
  x=np.array([D[a,'TRANSFER4_TO_8',t][0] for t in ts]);mus=x.mean(axis=1);cis=np.array([CI(v) for v in x]);ax.plot(np.array(ts)/100,mus,'o-',label=a);ax.fill_between(np.array(ts)/100,cis[:,0],cis[:,1],alpha=.15)
 ax.set_xlabel('Nominal denoising progress t/100');ax.set_ylabel('Sorted canonical-weight TV distance');ax.set_title('Distance to primary target TRANSFER4_TO_8 (lower is closer)');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(root/'distance_to_transfer.png',dpi=160);plt.close(fig)
 for key,name in [('ESS','ess_profile.png'),('effective_ancestors','ancestry_profile.png'),('max_weight','max_weight_profile.png')]:
  fig,ax=plt.subplots(figsize=(9,5))
  for m in ['NO_SEARCH','DENSE_REFERENCE','TRANSFER4_TO_8','MANUAL8']:
   pp=[p for p in profiles if p['policy']==m];x=[p['nominal_progress'] for p in pp];ax.plot(x,[p[key] for p in pp],'o-',label=m);ax.fill_between(x,[p[key+'_CI_low'] for p in pp],[p[key+'_CI_high'] for p in pp],alpha=.12)
  ax.set_xlabel('Nominal denoising progress');ax.set_ylabel(key);ax.set_title('Pre-resampling population: '+key);ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(root/name,dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,5))
 for q in quality:ax.scatter(q['mean_D_weight_to_Transfer'],q['terminal_IR_minus_Transfer']);ax.annotate(q['policy'],(q['mean_D_weight_to_Transfer'],q['terminal_IR_minus_Transfer']),fontsize=8)
 ax.set_xlabel('Mean D_weight to Transfer');ax.set_ylabel('Terminal IR minus Transfer');ax.set_title('Exploratory association only; known endpoint/RNG confound');fig.tight_layout();fig.savefig(root/'reward_vs_distance.png',dpi=160);plt.close(fig)
 accounting={'logical_full_runs':1440,'full_diffusion_NFE':sum(u['diffusion_NFE'] for u in units),'diagnostic_verifier_evaluations':sum(u['diagnostic_verifier_evaluations'] for u in units),'actual_intermediate_BFS_verifier_evaluations':sum(u['actual_intermediate_verifier_evaluations'] for u in units),'terminal_verifier_evaluations':sum(u['terminal_verifier_evaluations'] for u in units),'probe_rows':len(probes),'approximate_cost':'actual billing unavailable; chip-hours reported without assuming free/paid tariff'}
 launch=json.loads((root/'full_launch.json').read_text());complete=[json.loads(p.read_text()) for p in (root/'full').glob('chip*/completion.json')]
 accounting['worker_TPU_chip_hours']=sum(c['wall_seconds'] for c in complete)/3600
 accounting['wall_seconds']=max(json.loads((p.parent/'started.json').read_text())['unix']+json.loads(p.read_text())['wall_seconds'] for p in (root/'full').glob('chip*/completion.json'))-launch['started_unix']
 accounting['illustrative_historical_rate_USD_per_chip_hour']=4.2
 accounting['illustrative_cost_USD_at_historical_rate']=accounting['worker_TPU_chip_hours']*4.2
 accounting['rate_note']='Sensitivity scenario using prior campaign historical us-east1 rate, not current us-central1/account billing or a refreshed quote'
 smoke_rows=[json.loads(p.read_text()) for base in (root/'smoke',root/'smoke_v3') for p in (base/'raw').rglob('*.json')]
 accounting['smoke_completed_runs_including_repair_attempt']=len(smoke_rows)
 accounting['smoke_diffusion_NFE']=sum(r['diffusion_NFE'] for r in smoke_rows)
 accounting['smoke_diagnostic_verifier_evaluations']=sum(len(json.loads(p.read_text()))*8 for p in (root/'smoke_v3/probe_records').glob('*.json'))
 accounting['full_wall_includes_model_load']=True
 accounting['worker_TPU_chip_hours_excludes_model_load']=True
 accounting['full_4chip_window_hours_including_load']=4*accounting['wall_seconds']/3600
 (root/'compute_accounting.json').write_text(json.dumps(accounting,indent=2));(root/'INTEGRITY.json').write_text(json.dumps({'passed':True,'units':1440,'probe_rows':12960,'initial_particle_pairings':240,'statistical_unit':'60 prompts'},indent=2))
 if primary['verdict']=='YES' and all(secondary[t]['verdict']=='YES' for t in TARGETS[1:]):decision='A'
 elif any(c['CI_high']<0 for c in checkpoints) and any(c['CI_low']>0 for c in checkpoints):decision='B'
 elif dense10['verdict']=='NO':decision='C'
 elif primary['verdict']=='NO':decision='D'
 else:decision='E'
 lines=['# Reference distribution anchor diagnostic','','PRIMARY HYPOTHESIS',f"DenseRef closer to Transfer than NoSearch: {primary['verdict']}",f"Mean distance DenseRef -> Transfer: {primary['mean_DenseRef_to_Transfer']:.6f}",f"Mean distance NoSearch -> Transfer: {primary['mean_NoSearch_to_Transfer']:.6f}",f"Paired difference: {primary['mean']:+.6f}; SE {primary['SE_across_prompts']:.6f}; CI95% {primary['CI95']}",f"One-sided negative sign-flip p: {primary['sign_flip_p_one_sided_negative']:.6g}; two-sided p: {primary['sign_flip_p_two_sided']:.6g}",'','DENSER-REFERENCE QUESTION',f"Dense10 closer than DenseRef: {dense10['verdict']}",'','GENERALITY ACROSS TARGET POLICIES',json.dumps({t:secondary[t] for t in TARGETS[1:]},indent=2),'','MAIN INTERPRETATION',decision,'','Inference uses 60 previously explored TRAIN prompts; four fresh seeds averaged within prompt before bootstrap/sign flips. Checkpoint-wise intervals are descriptive, not separate confirmatory tests. Secondary targets remain separate. No production graph or policy was modified.','', 'Canonical weights use raw IR at tau8, never the policy Max-state or deployment weights. Sorted-weight distances are a verifier-based population proxy, not proof that full latent distributions match. TRANSFER4_TO_8 is a high-performing frozen target, not an optimal-policy claim.','', '## Distance summary','',json.dumps(ds,indent=2),'','## Exploratory quality association','',json.dumps(exploratory,indent=2),'','## Compute accounting','',json.dumps(accounting,indent=2)]
 (root/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'primary':primary,'dense10':dense10,'interpretation':decision,'accounting':accounting},indent=2))

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();analyze(a.run_root.resolve())
