import math
import time
import torch
from sd15_sinkhorn_tpu.sd15_sampler import ddim_update


def policy(name,steps):
    if name=='manual':
        original=[20,40,80]
        tau=[10*((1+.024)**i-1) for i in original]
    elif name=='graph':
        original=[40,60,90]; tau=[8.,8.,32.]
    else: raise ValueError(name)
    # Transfer *completed-update* progress and preserve original temperature.
    mapped=[math.ceil((i+1)*steps/100)-1 for i in original]
    if len(set(mapped))!=3: raise ValueError('mapped BFS events collide')
    return dict(name=name,original_indices=original,resampling_steps=mapped,
                temperature_by_step=list(zip(mapped,tau)),particles=None,
                gamma_original=.024 if name=='manual' else None)


@torch.no_grad()
def advance(sampler,prompt,initial,start,stop,noise_seed):
    """Same arithmetic/RNG as Sampler.run, stopped at a global population event."""
    if not 0<=start<stop<=len(sampler.timesteps): raise ValueError('bad segment')
    initial=initial.to(device='cpu',dtype=torch.bfloat16)
    embed=sampler.embedding(prompt); batch=sampler.batch
    final=[]; predictions=[]
    begin=time.perf_counter()
    for offset in range(0,len(initial),batch):
        chunk=initial[offset:offset+batch]; count=len(chunk)
        if count<batch: chunk=torch.cat([chunk,chunk[-1:].repeat(batch-count,1,1,1)])
        x=chunk.to(sampler.rt.device)
        for k in range(start,stop):
            timestep=torch.tensor(sampler.timesteps[k],dtype=torch.int64,device=sampler.rt.device)
            pred=sampler.pipe.unet(torch.cat([x,x]),timestep,encoder_hidden_states=embed,return_dict=False)[0]
            uncond,cond=pred.chunk(2)
            eps=uncond.float()+sampler.cfg['cfg_scale']*(cond.float()-uncond.float())
            coeff=sampler.coefficients[k].to(sampler.rt.device)
            noise=None
            if sampler.cfg['ddim_eta']>0:
                generator=torch.Generator().manual_seed(noise_seed+k*100003+offset)
                noise=torch.randn(chunk.shape,generator=generator).to(sampler.rt.device)
            # x0 is the scheduler prediction BEFORE this update, as in FKD.
            x0=((x.float()-coeff[1]*eps)/coeff[0]).to(torch.bfloat16)
            x=ddim_update(x,eps,coeff,noise).to(torch.bfloat16)
            sampler.rt.xm.mark_step()
        final.append(x[:count].cpu()); predictions.append(x0[:count].cpu())
    sampler.rt.sync()
    final=torch.cat(final); predictions=torch.cat(predictions)
    if not torch.isfinite(final).all() or not torch.isfinite(predictions).all():
        raise RuntimeError('nonfinite BFS state/x0')
    print(f'BFS segment {start}:{stop}, N={len(initial)}',flush=True)
    return final, predictions, time.perf_counter()-begin


def make_adapter(n,record,seed,index,reward_fn):
    from scripts.static_bfs_graph.adapter import StaticGraphBFSAdapter
    return StaticGraphBFSAdapter(
        potential_type='max',lmbda=10.,num_particles=n,time_steps=100,
        reward_fn=reward_fn,latent_to_decode_fn=lambda x:x,device=torch.device('cpu'),
        resampling='ssp',tempering_schedule='constant',
        resampling_steps=record['resampling_steps'],selection_mode='raw_tau',
        temperature_by_step=record['temperature_by_step'],
        graph_trial_seed=seed,graph_prompt_index=index)


def run_bfs(sampler,reward,prompt,initial,noise_seed,record,seed,index):
    adapter=make_adapter(len(initial),record,seed,index,
        lambda x:torch.tensor(reward.score(prompt,sampler.decode(x)),dtype=torch.float32))
    x=initial; start=0; denoise_seconds=0.
    original_ancestors=torch.arange(len(initial))
    for event in record['resampling_steps']:
        x,x0,seconds=advance(sampler,prompt,x,start,event+1,noise_seed)
        denoise_seconds+=seconds
        x,_=adapter.resample(sampling_idx=event,latents=x,x0_preds=x0)
        parents=adapter.events[-1]['parent_indices']
        original_ancestors=original_ancestors[parents]
        adapter.events[-1]['unique_initial_ancestors']=len(set(original_ancestors.tolist()))
        start=event+1
    x,_,seconds=advance(sampler,prompt,x,start,len(sampler.timesteps),noise_seed)
    denoise_seconds+=seconds
    if x.shape!=initial.shape: raise RuntimeError('BFS particle count changed')
    return x,adapter.events,denoise_seconds
