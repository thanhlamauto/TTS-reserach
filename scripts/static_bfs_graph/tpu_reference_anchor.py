"""TRAIN-only reference-population anchor diagnostic; abort on probe side effects."""
from pathlib import Path
from hashlib import sha256
import json,os,random,subprocess,time
import numpy as np
from .tpu_reference_compare import checked_inputs,REPO,read,digest,write_once,records as audited_records
from .policy import StaticBFSPolicy
from .runner import Runtime,static_record
from .adapter import StaticGraphBFSAdapter
from .tpu_early20_train60 import CapturePipeline
from scripts.diffusion_classical_search_t2i import tpu_runner as baseline

SEEDS=(2026100701,2026100702,2026100703,2026100704)
PROBES=(10,20,30,40,50,60,70,80,90)
PROTOCOL='REFERENCE_DISTRIBUTION_ANCHOR_V1'

def freeze(root):
 pool,cfg,old=checked_inputs();audited=audited_records()
 dense=StaticBFSPolicy(tuple(cfg['pilot']['candidate_steps']),tuple((t,cfg['pilot']['reference_tau']) for t in cfg['pilot']['candidate_steps']),8)
 def rec(p):return static_record(p)
 policies={'NO_SEARCH':rec(StaticBFSPolicy((),(),8)),'DENSE_REFERENCE':rec(dense),'DENSE10':rec(StaticBFSPolicy(PROBES,tuple((t,8.) for t in PROBES),8)),**{k:audited[k] for k in ['MANUAL8','TRANSFER4_TO_8','RETUNED8']}}
 assert policies['DENSE_REFERENCE']==audited['DENSE_REFERENCE']
 assert policies['MANUAL8']['gamma']==.024 and policies['MANUAL8']['resampling_steps']==[20,40,80]
 assert policies['TRANSFER4_TO_8']['temperature_by_step']==[[40,8.],[60,8.],[90,32.]]
 assert policies['RETUNED8']['temperature_by_step']==[[40,2.],[60,8.],[90,32.]]
 plan={'protocol':PROTOCOL,'prompt_ids':pool['train_indices'],'seeds':list(SEEDS),'N':8,'SD':'SD1.5','dtype':'BF16','steps':100,'eta':1.,'scoring':'Max','resampling':'SSP','verifier':'ImageReward-v1.0','probe_tau':8.,'probe_indices':list(PROBES),'primary_indices':[20,40,60,80,90],'probe_timing':'after DDIM update, before any BFS resample; t is existing zero-based sampling_idx; completed update=t+1','plot_progress':'nominal t/100 as protocol requests','canonical_weights':'softmax(8 * raw verifier scores), no deployment Max buffer','bootstrap_replicates':50000,'sign_flip_replicates':100000,'statistical_unit':'prompt, average4seeds then checkpoints','stat_rng_seed':20261007,'logical_full_runs':1440,'diffusion_NFE_full':1152000,'diagnostic_verifier_evaluations_full':103680,'actual_intermediate_BFS_verifier_evaluations_full':48000,'terminal_verifier_evaluations_full':11520,'primary_target':'TRANSFER4_TO_8','validation_test_external_used':False,'production_policy_modified':False,'probe_strategy':'clone tensors at checkpoints; diagnostic scoring deferred until endpoint generation completes, RNG restored; gate still requires exact endpoint/log equivalence','seeds_fixed_before_scoring':True,'freshness_scan':'no selected seed found in existing local artifacts/configs; remote metadata scanned by launch preflight'}
 write_once(root/'config.json',plan);write_once(root/'policy_configs.json',policies)
 files=['scripts/static_bfs_graph/tpu_reference_anchor.py','scripts/static_bfs_graph/tpu_reference_compare.py','scripts/static_bfs_graph/runner.py','scripts/static_bfs_graph/adapter.py','scripts/diffusion_classical_search_t2i/tpu_runner.py','configs/static_bfs_graph_t2i.json','configs/diffusion_classical_search_t2i_table12.json']
 write_once(root/'manifest_implementation_v3.json',{'source_sha256':{p:digest(REPO/p) for p in files},'config_sha256':digest(root/'config.json'),'policies_sha256':digest(root/'policy_configs.json'),'repo_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'model_revision':old['models']['sd15']['revision']})
 return pool,cfg,old,policies,dense

class ProbeAdapter(StaticGraphBFSAdapter):
 def __init__(self,**kwargs):
  super().__init__(**kwargs);self.snapshots=[];self.ancestry=np.arange(self.num_particles)
 def resample(self,*,sampling_idx,latents,x0_preds):
  if sampling_idx in PROBES:
   self.snapshots.append((sampling_idx,latents.detach().clone(),x0_preds.detach().clone(),self.ancestry.copy()))
  result=super().resample(sampling_idx=sampling_idx,latents=latents,x0_preds=x0_preds)
  if sampling_idx in self.resampling_steps:self.ancestry=self.ancestry[np.asarray(self.events[-1]['parent_indices'])]
  return result

def tensor_hash(t):
 # Float conversion is exact for BF16 values; include shape and dtype.
 a=t.detach().float().cpu().numpy()
 return sha256(str((tuple(t.shape),str(t.dtype))).encode()+a.tobytes()).hexdigest()

def probe_stats(adapter,prompt,seed,method):
 import torch
 import torch_xla.core.xla_model as xm
 state=(random.getstate(),np.random.get_state(),torch.get_rng_state(),xm.get_rng_state(adapter.device))
 rows=[]
 try:
  for t,latents,x0,anc in adapter.snapshots:
   images=adapter.latent_to_decode_fn(x0)
   v=np.asarray(adapter.reward_fn(images).detach().float().cpu(),dtype=np.float64)
   assert len(v)==8 and np.isfinite(v).all()
   z=8*v;w=np.exp(z-z.max());w/=w.sum();a=np.bincount(anc,minlength=8)/8
   hw=-sum(x*np.log(x) for x in w if x>0);ha=-sum(x*np.log(x) for x in a if x>0)
   x=latents.detach().float().cpu().numpy().reshape(8,-1);norm=np.linalg.norm(x,axis=1);ds=[];cs=[]
   for i in range(8):
    for j in range(i):
     ds.append(np.linalg.norm(x[i]-x[j])/(.5*(norm[i]+norm[j])+1e-12));cs.append(1-np.dot(x[i],x[j])/(norm[i]*norm[j]+1e-12))
   order=np.sort(v)[::-1]
   rows.append({'prompt_id':prompt,'seed':seed,'policy':method,'t':t,'completed_update':t+1,'phase':'pre_resample','raw_scores':v.tolist(),'score_mean':float(v.mean()),'score_std':float(v.std()),'score_range':float(np.ptp(v)),'top1_top2_gap':float(order[0]-order[1]),'probe_tau':8.,'probe_weights':w.tolist(),'sorted_probe_weights':np.sort(w)[::-1].tolist(),'ESS':float(1/np.sum(w*w)),'max_weight':float(w.max()),'weight_entropy':float(hw),'normalized_weight_entropy':float(hw/np.log(8)),'ancestor_shares':a.tolist(),'surviving_ancestors':int(np.count_nonzero(a)),'ancestry_entropy':float(ha),'effective_ancestors':float(np.exp(ha)),'normalized_effective_ancestry':float(np.exp(ha)/8),'max_ancestry_share':float(a.max()),'pairwise_normalized_L2':float(np.mean(ds)),'pairwise_cosine_distance':float(np.mean(cs))})
 finally:
  random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2]);xm.set_rng_state(state[3],adapter.device)
 return rows

class InitialCapture(CapturePipeline):
 def __init__(self,pipe):
  super().__init__(pipe);self.initial=None;original=pipe.prepare_latents
  def capture(*args,**kwargs):
   result=original(*args,**kwargs);self.initial=result.detach().clone();return result
  pipe.prepare_latents=capture

def run_one(runtime,policy,reference,prompt,method,on):
 import fkd_pipeline_sd as module
 module.FKD=ProbeAdapter if on else StaticGraphBFSAdapter
 stage='probe_on' if on else 'probe_off'
 row=runtime.run(stage,policy,prompt,reference)
 hashes=[sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest() for im in runtime.pipe.images]
 initial=tensor_hash(runtime.pipe.initial)
 adapter=baseline._LAST_ADAPTER
 write_once(runtime.run_root/'endpoint_checks'/f'{method}-{stage}.json',{'all_particle_RGB_hashes':hashes,'initial_latents_sha256':initial,'row':row})
 return row,hashes,initial,adapter

def smoke(root,dcs,chip):
 pool,cfg,old,policies,ref=freeze(root)
 cfg={**cfg,'particles':8,'protocol_version':PROTOCOL}
 runtime=Runtime(root/'smoke_v3',dcs,SEEDS[0],chip,cfg,old,[{'prompt':pool['prompts'].get(str(i))} for i in range(100)])
 runtime.pipe=InitialCapture(runtime.pipe)
 prompt=pool['train_indices'][0];started=time.time();tests=[];initials=[]
 for method,policy in policies.items():
  off,h0,i0,_=run_one(runtime,policy,ref,prompt,method,False)
  on,h1,i1,adapter=run_one(runtime,policy,ref,prompt,method,True)
  same={'method':method,'terminal_all8_RGB_equal':h0==h1,'terminal_all8_scores_equal':off['final_particle_scores']==on['final_particle_scores'],'event_logs_equal':off['resampling_events']==on['resampling_events'],'initial_latents_equal':i0==i1,'off_selected_IR':off['selected_final_score'],'on_selected_IR':on['selected_final_score']}
  tests.append(same);initials.extend([i0,i1])
  if not all(same[k] for k in ['terminal_all8_RGB_equal','terminal_all8_scores_equal','event_logs_equal','initial_latents_equal']):
   failure={'passed':False,'failed_gate':2,'abort':True,'tests':tests,'wall_seconds':time.time()-started,'full_launch_allowed':False,'diagnostic_scoring_performed_before_failure':False}
   write_once(root/'smoke_gate.json',failure)
   (root/'REPORT.md').write_text('# Reference distribution anchor: ABORTED\n\nGate2 failed: probe-on and probe-off must have identical endpoints and event logs.\n\n'+json.dumps(failure,indent=2)+'\n\nNo full campaign or anchor hypothesis inference was launched. This does not test or reject the scientific anchor hypothesis. Only the correctness gate failed. Production graph/deployment/backend were unchanged.\n')
   print(json.dumps(failure,indent=2),flush=True);return
  assert [s[0] for s in adapter.snapshots]==list(PROBES)
  assert [e['sampling_index'] for e in on['resampling_events']]==policy['resampling_steps']
  import torch
  with torch.no_grad():probes=probe_stats(adapter,prompt,SEEDS[0],method)
  assert len(probes)==9 and all(p['probe_tau']==8. for p in probes)
  write_once(root/'smoke_v3'/'probe_records'/f'{method}.json',probes)
 assert len(set(initials))==1
 write_once(root/'smoke_gate.json',{'passed':True,'tests':tests,'gates_1_to_7_passed':True,'identical_initial_population_all_policies':True,'probe_scoring_deferred':True,'wall_seconds':time.time()-started,'full_launch_allowed':True})
 print('SMOKE GATES PASS',flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','smoke']);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--dcs-root',type=Path,required=True);p.add_argument('--chip',type=int,default=0)
 a=p.parse_args();(freeze(a.run_root) if a.stage=='freeze' else smoke(a.run_root,a.dcs_root,a.chip))
