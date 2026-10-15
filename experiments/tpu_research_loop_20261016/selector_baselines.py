"""Finite-bank compiler baseline rules. No runtime/controller or provider calls."""
import hashlib,json
import numpy as np

def cache_key(*,model_revision,scheduler_sha256,precision,code_sha256,policy_prefix,input_sha256,initial_noise_sha256,rng_state_sha256,kind,branch_suffix=None):
 required=[model_revision,scheduler_sha256,precision,code_sha256,input_sha256,initial_noise_sha256,rng_state_sha256]
 if any(not x for x in required):raise ValueError('cache identity incomplete')
 if kind not in ['prefix','suffix']:raise ValueError('cache kind')
 if kind=='suffix' and branch_suffix is None:raise ValueError('suffix must include full branch policy and RNG continuation identity')
 payload=dict(model_revision=model_revision,scheduler_sha256=scheduler_sha256,precision=precision,code_sha256=code_sha256,policy_prefix=policy_prefix,input_sha256=input_sha256,initial_noise_sha256=initial_noise_sha256,rng_state_sha256=rng_state_sha256,kind=kind,branch_suffix=branch_suffix)
 return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def direct_full_policy_select(rewards,semantic_hashes):
 """Uniform fixed-bank direct search, one fixed policy for all future instances.
 rewards: independent task x paired seed x candidate. No graph/shortlist restriction.
 Cost collection must finish a whole paired block before updating the estimate.
 """
 x=np.asarray(rewards,dtype=float)
 if x.ndim!=3 or x.shape[2]!=len(semantic_hashes) or not np.isfinite(x).all():raise ValueError('complete finite task/seed/candidate rectangle required')
 if len(set(semantic_hashes))!=len(semantic_hashes):raise ValueError('duplicate policy hashes')
 means=x.mean(axis=1).mean(axis=0);leader=min(range(len(semantic_hashes)),key=lambda i:(-means[i],semantic_hashes[i]))
 return {'policy_hash':semantic_hashes[leader],'mean_IR':float(means[leader]),'status':'EMPIRICAL_CHOICE_NO_CERTIFICATE','task_count':x.shape[0],'seeds_per_task':x.shape[1]}

def graph_only_select(edge_blocks,path_coefficients,semantic_hashes):
 """Mean additive cost on a covered finite bank; covariance kept as whole blocks."""
 d=np.asarray(edge_blocks,dtype=float);v=np.asarray(path_coefficients,dtype=float)
 if d.ndim!=2 or v.shape!=(len(semantic_hashes),d.shape[1]) or not np.isfinite(d).all():raise ValueError('edge block and candidate coverage mismatch')
 costs=d@v.T;leader=min(range(len(semantic_hashes)),key=lambda i:(costs[:,i].mean(),semantic_hashes[i]))
 return {'policy_hash':semantic_hashes[leader],'paired_candidate_cost_blocks':costs,'status':'SURROGATE_CHOICE_NO_CERTIFICATE'}

def full_policy_verify_select(rewards,semantic_hashes,frozen_shortlist):
 ix=[semantic_hashes.index(h) for h in frozen_shortlist]
 return direct_full_policy_select(np.asarray(rewards)[:,:,ix],frozen_shortlist)
