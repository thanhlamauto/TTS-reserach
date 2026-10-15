"""CPU-only tests against the installed adapter. No model, TPU or CUDA execution."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
sys.path[:0]=['/home/thanhlamtba31/TTS-reserach','/home/thanhlamtba31/static-bfs-dcs/official/text_to_image/fkd_diffusers']
from scripts.static_bfs_graph.adapter import StaticGraphBFSAdapter
from scripts.static_bfs_graph.policy import StaticBFSPolicy
checks=[]
def check(name,condition):
 assert condition,name
 checks.append({'name':name,'passed':True})
steps=[20,40,80];beta=[10*((1+.024)**i-1) for i in steps]
raw=[torch.tensor(v,dtype=torch.float32) for v in [[-1,.1,.3,-.5,.2,.05,-2,.4],[.2,.4,-.2,.05,.1,.3,0,.2],[.8,.1,.4,.2,-.1,.7,.05,.3]]]
def run(mode,temperatures):
 values=iter(raw)
 a=StaticGraphBFSAdapter(selection_mode=mode,temperature_by_step=temperatures,graph_trial_seed=42,graph_prompt_index=0,potential_type='max',lmbda=10,num_particles=8,time_steps=100,reward_fn=lambda x:next(values),latent_to_decode_fn=lambda x:x,device=torch.device('cpu'),resampling='ssp',tempering_schedule='increase' if mode=='manual' else 'constant',gamma=.024,resampling_steps=steps)
 lat=torch.arange(16,dtype=torch.float32).reshape(8,2)
 previous=np.zeros(8,dtype=np.float32)
 for i,r in zip(steps,raw):
  expected=np.maximum(r.numpy()*np.float32(10*((1+.024)**i-1)),previous)
  out,_=a.resample(sampling_idx=i,latents=lat,x0_preds=lat)
  e=a.events[-1];scores=np.asarray(e['particle_scores'],dtype=np.float32)
  check(mode+f'_max_history_{i}',np.allclose(scores,expected,rtol=1e-6,atol=1e-6))
  expected_weights=torch.softmax(torch.from_numpy(expected),0).numpy()
  check(mode+f'_score_multiplier_softmax_{i}',np.allclose(e['normalized_weights'],expected_weights,rtol=1e-6,atol=1e-7))
  check(mode+f'_fixed_shape_{i}',tuple(out.shape)==(8,2) and len(e['parent_indices'])==8)
  previous=expected[np.asarray(e['parent_indices'])];lat=out
 return a.events
m=run('manual',[]);s=run('raw_tau',list(zip(steps,beta)))
check('manual_exact_event_temperatures',[e['temperature'] for e in m]==beta)
check('manual_and_explicit_equivalence',m==s)
# Counterexample: denominator convention gives different weights.
r=torch.tensor([.1,.3,.7],dtype=torch.float32);check('multiplier_not_denominator',not torch.allclose(torch.softmax(8*r,0),torch.softmax(r/8,0)))
check('zero_based_index',StaticBFSPolicy((19,),((19,8.),),8).resampling_steps==(19,))
print(json.dumps({'passed':True,'checks':checks,'manual_beta':beta,'device':'cpu','neural_model_forwards':0,'TPU_initializations':0,'source_sha256':hashlib.sha256(Path('/home/thanhlamtba31/TTS-reserach/scripts/static_bfs_graph/adapter.py').read_bytes()).hexdigest()},indent=2))
