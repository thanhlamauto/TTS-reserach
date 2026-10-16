"""One event, 24 online particles, exact copied state AND prediction."""
import json
from pathlib import Path
import time
import numpy as np
import torch

from otlab.export import historical_tape
from otlab.io import atomic_json, tensor_sha
from .allocation import allocate
from .e0 import contact_sheet, mean_diversity


@torch.no_grad()
def prediction(adapter, state, index):
    pieces, clean = [], []
    s=adapter.sampler
    embed=s.embedding(adapter.prompt)
    for offset in range(0,len(state),s.batch):
        chunk=state[offset:offset+s.batch]
        count=len(chunk)
        if count<s.batch:chunk=torch.cat([chunk,chunk[-1:].repeat(s.batch-count,1,1,1)])
        x=chunk.to(device=s.rt.device,dtype=torch.bfloat16)
        timestep=torch.tensor(s.timesteps[index],dtype=torch.int64,device=s.rt.device)
        pred=s.pipe.unet(torch.cat([x,x]),timestep,encoder_hidden_states=embed,return_dict=False)[0]
        uncond,cond=pred.chunk(2)
        eps=uncond.float()+s.cfg['cfg_scale']*(cond.float()-uncond.float())
        coeff=s.coefficients[index].to(s.rt.device)
        x0 = (x.float() - coeff[1]*eps.float()) / coeff[0]
        adapter.sampler.rt.xm.mark_step()
        # Preserve the native BF16 UNet outputs, NOT a separately rounded FP32
        # CFG expression. CFG arithmetic remains fused with DDIM at transition.
        pieces.append(torch.stack([uncond[:count],cond[:count]],dim=1).cpu())
        clean.append(x0[:count].cpu())
        adapter.counters['event_prediction_sample_evaluations'] += count
        adapter.counters['event_prediction_padding_sample_evaluations'] += adapter.sampler.batch-count
        adapter.counters['unet_cfg_branch_sample_evaluations'] += 2*adapter.sampler.batch
    adapter.sampler.rt.sync()
    return torch.cat(pieces), torch.cat(clean)


@torch.no_grad()
def transition(adapter, state, raw_prediction, index, noise):
    result = []
    for offset in range(0, len(state), adapter.sampler.batch):
        x, raw, n = [t[offset:offset+adapter.sampler.batch] for t in (state,raw_prediction,noise)]
        count = len(x)
        if count < adapter.sampler.batch:
            x=torch.cat([x,x[-1:].repeat(adapter.sampler.batch-count,1,1,1)])
            n=torch.cat([n,n[-1:].repeat(adapter.sampler.batch-count,1,1,1)])
            raw=torch.cat([raw,raw[-1:].repeat(adapter.sampler.batch-count,1,1,1,1)])
        raw=raw.to(adapter.sampler.rt.device)
        e=raw[:,0].float()+adapter.sampler.cfg['cfg_scale']*(raw[:,1].float()-raw[:,0].float())
        y = adapter.update(x.to(adapter.sampler.rt.device), e,
                           adapter.sampler.coefficients[index].to(adapter.sampler.rt.device),
                           n.to(adapter.sampler.rt.device)).to(torch.bfloat16)
        adapter.sampler.rt.xm.mark_step()
        result.append(y[:count].cpu())
        # No UNet call here: event prediction was already counted above.
        adapter.counters['cached_prediction_transitions'] += count
    adapter.sampler.rt.sync()
    return torch.cat(result)


def execute(adapter, features, config, output, merge_distance, status):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    before = dict(adapter.counters)
    from sd15_sinkhorn_tpu.common import initial_noise
    seed, prompt_id = config['r1_smoke_seed'], 0
    initial, noise_seed = initial_noise(seed,prompt_id,0,24)
    tape = historical_tape(adapter.sampler,noise_seed,24,start=0)
    ids = tuple(range(24))
    status('R1_prefix', prompt_id=0, seed=seed, event_next_index=40)
    if config.get('reuse_r1_prefix'):
        from otlab.io import load_tensor
        saved=load_tensor(config['reuse_r1_prefix'])
        if (saved['next_index']!=40 or saved['noise_seed']!=noise_seed
                or saved['tape_sha']!=tensor_sha(tape.variance_noise) or not torch.equal(saved['initial'],initial)):
            raise RuntimeError('saved R1 prefix input identity mismatch')
        checkpoint=saved['checkpoint']
        atomic_json(output/'PREFIX_REUSE.json',dict(path=config['reuse_r1_prefix'],exact_inputs_verified=True,
                                                  reused_sample_nfe=960))
    else:
        checkpoint = adapter.advance(initial,0,40,ids,tape)
    torch.save(dict(initial=initial,checkpoint=checkpoint,noise_seed=noise_seed,
                    tape_sha=tensor_sha(tape.variance_noise),next_index=40),output/'prefix.pt')
    if config.get('transition_mode')!='original_fused_recompute':
        raise RuntimeError('cached-prediction optimization did not pass exactness; use frozen original transition')
    status('R1_original_full_replay_gate')
    trusted,checkpoints,_=adapter.sampler.run(adapter.prompt,initial[:4],start=0,checkpoints=(40,),noise_seed=noise_seed)
    adapter.counters['original_full_replay_sample_steps']+=400
    adapter.counters['unet_cfg_branch_sample_evaluations']+=800
    if not torch.equal(checkpoint[:4],checkpoints[40]):
        raise RuntimeError('new keyed prefix differs from original sampler.run prefix')
    status('R1_event_prediction')
    # Safe compatibility path: use the existing preview evaluator and ORIGINAL
    # fused transition after donor copy. Charge24 auxiliary UNet evaluations.
    clean=adapter.predict_clean(checkpoint,40)
    proxy, rgb = adapter.score(clean,'proxy')
    dino = features.encode(rgb,'R1_event40')
    rng = np.random.default_rng(np.random.SeedSequence([seed,prompt_id,40,7341]))
    uniforms = rng.random(6)
    common, methods = allocate(proxy.numpy(),dino.numpy(),merge_distance,1.,uniforms)
    common.update(event_next_index=40,state_after_updates=40,innovation_first_applied_index=40,
                  prefix_seed=seed,source_slots=list(ids),noise_key='historical per-step + child-slot microbatch',
                  donor_state_copy=True,donor_prediction_copy=False,correct_donor_prediction_recomputed=True,
                  cached_prediction_optimization_passed=False,transition_mode=config['transition_mode'],
                  latent_interpolation=False,
                  encoder_gradient=False,external_bank=False,merge_distance=merge_distance)
    atomic_json(output/'allocation.json',dict(common=common,methods=methods))
    torch.save(dict(clean=clean,rgb=rgb,reward=proxy,features=dino,
                    prediction_recomputed_after_donor_copy=True),output/'event.pt')
    rows=[]
    for name,allocation in methods.items():
        start=time.perf_counter()
        status('R1_terminal',method=name)
        parent = allocation['parent_ids']
        method_before=dict(adapter.counters)
        terminal=adapter.advance(checkpoint[parent],40,100,ids,tape)
        if name=='none':
            if not torch.equal(terminal[:4],trusted):
                raise RuntimeError('none checkpoint resume differs from original full100 parent')
            atomic_json(output/'replay_gate.json',dict(passed=True,original_full100_vs_resume_bitwise=True,
                original_prefix40_bitwise=True,gamma_zero_recovers_reward=True,
                original_gate_sample_nfe=400,cached_prediction_optimization_passed=False,
                transition_mode=config['transition_mode'],aux_event_nfe=24))
        reward,terminal_rgb=adapter.score(terminal,'terminal')
        terminal_dino=features.encode(terminal_rgb,name+'_terminal')
        top=torch.argsort(reward,descending=True,stable=True)[:8]
        from PIL import Image
        from sd15_sinkhorn_tpu.reward import CLIPCPU
        if not hasattr(adapter,'ot_next_clip'):
            adapter.ot_next_clip=CLIPCPU(adapter.sampler.cfg,config['dcs_root'])
        clip=adapter.ot_next_clip.score(adapter.prompt,[Image.fromarray(x.permute(1,2,0).numpy()) for x in terminal_rgb])
        clip_f=torch.tensor(clip['clip_embeddings'])
        sibling=[]
        for donor in sorted(set(parent)):
            children=[i for i,p in enumerate(parent) if p==donor]
            if len(children)>1:
                for pos,a in enumerate(children):
                    for b in children[pos+1:]:
                        sibling.append(dict(parent=donor,slots=[a,b],pair_mean_ir=float(reward[[a,b]].mean()),
                                            pair_best_ir=float(reward[[a,b]].max()),
                                            dino_distance=float(1-terminal_dino[a].double()@terminal_dino[b].double()),
                                            clip_distance=float(1-clip_f[a].double()@clip_f[b].double())))
        row=dict(method=name,prompt_id=0,seed=seed,mean_ir=float(reward.mean()),best_ir=float(reward.max()),
                 top8_ir=float(reward[top].mean()),top8_indices=top.tolist(),
                 top8_dino_diversity=mean_diversity(terminal_dino[top]),
                 top8_clip_diversity=mean_diversity(clip_f[top]),
                 all24_dino_diversity=mean_diversity(terminal_dino),
                 unique_ancestors=len(set(parent)),sibling_pairs=sibling,
                 nominal_cold_sampling_nfe=2400,aux_unet_nfe=24,total_cold_nfe=2424,
                 event_prediction_included_in_denoising_nfe=False,
                 transition_mode=config['transition_mode'],cached_prediction_optimization_passed=False,
                 current_job_counters={k:v-method_before.get(k,0) for k,v in adapter.counters.items()},
                 shared_prefix_cold_wall_measured=False,elapsed_seconds=time.perf_counter()-start,
                 primary_good_distinct='NOT_EVALUATED_THRESHOLDS_UNSET',
                 semantic_success='PENDING_BLINDED_VISUAL_REVIEW',
                 endpoint_sha=tensor_sha(terminal))
        torch.save(dict(latent=terminal,rgb=terminal_rgb,rewards=reward,dino=terminal_dino,
                        clip=clip_f,allocation=allocation),output/(name+'.pt'))
        atomic_json(output/(name+'.json'),row)
        contact_sheet(terminal_rgb[top],[f'rank{j+1} slot{int(i)} IR {float(reward[i]):.2f}' for j,i in enumerate(top)],
                      output/(name+'_top8.png'))
        with (output/'rows.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
        rows.append(row)
    if len(rows)!=4 or len({r['method'] for r in rows})!=4:
        raise RuntimeError('R1 smoke incomplete')
    # Verify survivors remain exact across all four methods, not just ancestry logs.
    from otlab.io import load_tensor
    none=load_tensor(output/'none.pt')
    survivors=common['survivors']
    for name in methods:
        current=load_tensor(output/(name+'.pt'))
        if not torch.equal(none['latent'][survivors],current['latent'][survivors]):
            raise RuntimeError('unchanged survivor trajectory drifted')
    result=dict(passed=True,rows=rows,prompts=1,prefix_seeds=1,
                engineering_only=True,scientific_winner=None,unchanged18survivors_bitwise=True,
                transition_mode=config['transition_mode'],cached_prediction_optimization_passed=False,
                all_counters={k:v-before.get(k,0) for k,v in adapter.counters.items()},
                future_antithetic_or_sweep='NOT_RUN')
    atomic_json(output/'COMPLETE.json',result)
    return result
