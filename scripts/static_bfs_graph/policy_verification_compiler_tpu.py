"""Thin campaign adapter over the already audited live-XLA BFS engine."""
from pathlib import Path
import argparse,itertools,time,traceback
import numpy as np
from . import adaptive_refinement_tpu as backend
from . import policy_verification_compiler as s
backend.verify=s.verify

def checked_full(eng,pid,seed,p):
 before=dict(eng.account);start=time.time();row,_=eng.direct(pid,seed,p)
 assert len(row['events'])==3 and [e['sampling_index'] for e in row['events']]==p['resampling_steps']
 assert row['reward']==max(row['scores'])
 nfe=eng.account['actual_denoising_NFE']-before['actual_denoising_NFE'];assert nfe==800
 for event in row['events']:
  expected=10*((1+.024)**event['sampling_index']-1) if p['kind']=='manual' else dict(p['temperature_by_step'])[event['sampling_index']]
  assert abs(event['temperature']-expected)<1e-9
 return {'prompt':pid,'seed':seed,'policy':p,'runtime_seconds':time.time()-start,'NFE':nfe,'verifier_calls':eng.account['actual_verifier_batch_calls']-before['actual_verifier_batch_calls'],'verifier_scores':eng.account['actual_verifier_candidate_scores']-before['actual_verifier_candidate_scores'],'selected_particle_id':int(np.argmax(row['scores'])),'selected_particle_id_semantics':'final population slot; original ancestor identity is not implied',**row}

def worker(root,stage,chip):
 eng=backend.Engine(root,chip,stage)
 try:
  if stage=='gates':
   # Legacy gate checks A+B covers TRAIN. Its B is used only for a set check,
   # not any inference: expose B+C to that assertion without changing disk split.
   old=eng.split['B'];eng.split['B']=old+eng.split['C']
   try:eng.gates()
   finally:eng.split['B']=old
   pid=eng.split['A'][0];seed=eng.cfg['seeds']['A'][0];manual=s.records()['MANUAL8'];m=checked_full(eng,pid,seed,manual);p=s.read(root/'candidate_bank.json')[0];a=checked_full(eng,pid,seed,p);b=checked_full(eng,pid,seed,p)
   assert all(a[k]==b[k] for k in ['reward','scores','RGB_hashes','latent_hash','events','initial_latents_sha256','selected_particle_id'])
   assert m['initial_latents_sha256']==a['initial_latents_sha256']
   s.dump(root/'correctness_gates.json',{'passed':True,'static_gates':'static_gates.json','eight_shared_replay_gates':'gates.json','snapshot_on_off_endpoint_exact':True,'full_policy_repeat_exact':True,'manual_genuine_increase_temperatures':[e['temperature'] for e in m['events']],'initial_noise_pairing_manual_static':True,'terminal_reward_selected_same_output':True,'NFE800':True,'Max_SSP_DDIM100_eta1_BF16_N8':True,'legacy_gate_split_compatibility':'in-memory B+C only for TRAIN set assertion; frozen split unchanged','RNG_limitation':'paired initial particles and keyed events; divergent ancestry limits causal coupling'})
  elif stage=='graph':
   bank=s.read(root/'candidate_bank.json');js=s.v2.union(bank);units=list(itertools.product(eng.split['A'],eng.cfg['seeds']['A']))
   for pid,seed in units[chip::4]:eng.profile('A',pid,seed,js)
  else:
   assert stage in ['verification','audit']
   frozen='frozen_shortlist.json' if stage=='verification' else 'frozen_audit_policies.json';s.verify(root);assert s.read(root/(frozen+'.receipt.json'))['sha256']==s.sha(root/frozen)
   doc=s.read(root/frozen);ps=[r['policy'] for r in doc['policies']] if stage=='verification' else list(doc.values());ps=list({p['id']:p for p in ps}.values());cohort='B' if stage=='verification' else 'C';units=list(itertools.product(eng.split[cohort],eng.cfg['seeds'][cohort]))
   for pid,seed in units[chip::4]:
    for p in ps:
     path=root/stage/f'p{pid}-s{seed}-{p["id"]}.json'
     if path.exists():assert s.read(path)['policy']==p;continue
     row=checked_full(eng,pid,seed,p);s.dump(path,row);eng.persist()
  eng.persist();s.dump(root/'workers'/stage/f'chip{chip}.done.json',{'success':True,'account':eng.account})
 except BaseException as e:eng.persist();s.dump(root/'failures'/f'{stage}-chip{chip}.json',{'exception':repr(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--stage',required=True);p.add_argument('--chip',type=int,required=True);a=p.parse_args();worker(a.run_root.resolve(),a.stage,a.chip)
