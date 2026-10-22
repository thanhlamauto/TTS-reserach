"""Full workers, gated strictly by frozen smoke equivalence tests."""
import argparse,json,os,time
from pathlib import Path
from . import tpu_reference_anchor as A
from .tpu_reference_compare import checked_inputs,REPO,read,digest,write_once
from .runner import Runtime

def worker(root,dcs,chip):
 import torch
 gate=read(root/'smoke_gate.json');assert gate['passed'] and gate['full_launch_allowed']
 pool,cfg,old=checked_inputs();plan=read(root/'config.json');policies=read(root/'policy_configs.json')
 manifest=read(root/'manifest_implementation_v3.json')
 assert digest(root/'config.json')==manifest['config_sha256'] and digest(root/'policy_configs.json')==manifest['policies_sha256']
 assert all(digest(REPO/p)==h for p,h in manifest['source_sha256'].items())
 assert plan['seeds']==list(A.SEEDS) and plan['prompt_ids']==pool['train_indices']
 write_once(root/f'full_worker_source_chip{chip}.json',{'source_sha256':digest(Path(__file__)),'smoke_gate_sha256':digest(root/'smoke_gate.json')})
 shard=root/'full'/f'chip{chip}';shard.mkdir(parents=True,exist_ok=True)
 cfg={**cfg,'particles':8,'protocol_version':A.PROTOCOL}
 runtime=Runtime(shard,dcs,A.SEEDS[0],chip,cfg,old,[{'prompt':pool['prompts'].get(str(i))} for i in range(100)])
 runtime.pipe=A.InitialCapture(runtime.pipe)
 ref=A.StaticBFSPolicy(tuple(cfg['pilot']['candidate_steps']),tuple((t,cfg['pilot']['reference_tau']) for t in cfg['pilot']['candidate_steps']),8)
 started=time.time();write_once(shard/'started.json',{'unix':started})
 for prompt in pool['train_indices'][chip::4]:
  for seed in A.SEEDS:
   runtime.seed=seed;initial=None
   for method,policy in policies.items():
    key=f'p{prompt:03d}-s{seed}-{method}';dest=shard/'units'/f'{key}.json'
    if dest.exists():
     prior=read(dest);assert prior['status']=='success';current=prior['initial_latents_sha256']
    else:
     try:
      row,hashes,current,adapter=A.run_one(runtime,policy,ref,prompt,key,True)
      before=list(adapter.events)
      with torch.no_grad():probes=A.probe_stats(adapter,prompt,seed,method)
      adapter.snapshots.clear()
      assert adapter.events==before and [p['t'] for p in probes]==list(A.PROBES)
      assert [e['sampling_index'] for e in row['resampling_events']]==policy['resampling_steps']
      assert all(p['probe_tau']==8. for p in probes) and row['diffusion_NFE']==800
      unit={'prompt_id':prompt,'seed':seed,'policy':method,'initial_latents_sha256':current,'terminal_all8_RGB_hashes':hashes,'terminal_scores':row['final_particle_scores'],'terminal_ImageReward':row['selected_final_score'],'actual_events':row['resampling_events'],'actual_intermediate_verifier_evaluations':8*len(policy['resampling_steps']),'terminal_verifier_evaluations':8,'diagnostic_verifier_evaluations':72,'diffusion_NFE':800,'probes':probes,'status':'success'}
      write_once(dest,unit)
      print(json.dumps({'completed_unit':key,'IR':unit['terminal_ImageReward'],'seconds_since_start':time.time()-started}),flush=True)
     except BaseException as e:
      write_once(shard/'worker_failure.json',{'unit':key,'error':repr(e),'partial_work_preserved':True});raise
    if initial is None:initial=current
    assert current==initial,'initial population pairing failed'
 write_once(shard/'completion.json',{'rows':360,'wall_seconds':time.time()-started,'chip':chip})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--dcs-root',type=Path,required=True);p.add_argument('--chip',type=int,required=True);a=p.parse_args();worker(a.run_root,a.dcs_root,a.chip)
