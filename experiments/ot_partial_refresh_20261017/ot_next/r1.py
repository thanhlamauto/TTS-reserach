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
    for _, count, x, eps, coeff in adapter._chunks(state, index):
        x0 = (x.float() - coeff[1]*eps.float()) / coeff[0]
        adapter.sampler.rt.xm.mark_step()
        pieces.append(eps[:count].cpu())
        clean.append(x0[:count].cpu())
        adapter.counters['event_prediction_sample_evaluations'] += count
        adapter.counters['event_prediction_padding_sample_evaluations'] += adapter.sampler.batch-count
        adapter.counters['unet_cfg_branch_sample_evaluations'] += 2*adapter.sampler.batch
    adapter.sampler.rt.sync()
    return torch.cat(pieces), torch.cat(clean)


@torch.no_grad()
def transition(adapter, state, eps, index, noise):
    result = []
    for offset in range(0, len(state), adapter.sampler.batch):
        x, e, n = [t[offset:offset+adapter.sampler.batch] for t in (state,eps,noise)]
        count = len(x)
        if count < adapter.sampler.batch:
            x, e, n = [torch.cat([t,t[-1:].repeat(adapter.sampler.batch-count,1,1,1)]) for t in (x,e,n)]
        y = adapter.update(x.to(adapter.sampler.rt.device), e.to(adapter.sampler.rt.device),
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
    checkpoint = adapter.advance(initial,0,40,ids,tape)
    torch.save(dict(initial=initial,checkpoint=checkpoint,noise_seed=noise_seed,
                    tape_sha=tensor_sha(tape.variance_noise),next_index=40),output/'prefix.pt')
    status('R1_event_prediction')
    eps, clean = prediction(adapter,checkpoint,40)
    proxy, rgb = adapter.score(clean,'proxy')
    dino = features.encode(rgb,'R1_event40')
    rng = np.random.default_rng(np.random.SeedSequence([seed,prompt_id,40,7341]))
    uniforms = rng.random(6)
    common, methods = allocate(proxy.numpy(),dino.numpy(),merge_distance,1.,uniforms)
    common.update(event_next_index=40,state_after_updates=40,innovation_first_applied_index=40,
                  prefix_seed=seed,source_slots=list(ids),noise_key='historical per-step + child-slot microbatch',
                  donor_state_copy=True,donor_prediction_copy=True,latent_interpolation=False,
                  encoder_gradient=False,external_bank=False,merge_distance=merge_distance)
    atomic_json(output/'allocation.json',dict(common=common,methods=methods))
    torch.save(dict(eps=eps,clean=clean,rgb=rgb,reward=proxy,features=dino),output/'event.pt')
    # Single unconditional no-op state/prediction reuse gate on the exact tape.
    replay_before = dict(adapter.counters)
    direct41 = adapter.advance(checkpoint[:4],40,41,ids[:4],tape)
    reused41 = transition(adapter,checkpoint[:4],eps[:4],40,tape.at(40,ids[:4]))
    if not torch.equal(direct41,reused41):
        raise RuntimeError('cached epsilon transition is not exact')
    # Copy/reorder test includes a donor different from every source being replaced.
    parents = methods['partial_ot']['parent_ids']
    replaced = common['replaced']
    check_parents = [parents[i] for i in replaced[:4]]
    check_noise = tape.at(40,replaced[:4])
    donor_direct = adapter.advance(checkpoint[check_parents],40,41,replaced[:4],tape)
    donor_reused = transition(adapter,checkpoint[check_parents],eps[check_parents],40,check_noise)
    if not torch.equal(donor_direct,donor_reused):
        raise RuntimeError('copied donor prediction/state mismatch')
    atomic_json(output/'replay_gate.json',dict(passed=True,no_op_bitwise=True,donor_reorder_bitwise=True,
                gamma_zero_recovers_reward=common['gamma_zero_recovers_reward'],
                gate_counters={k:v-replay_before.get(k,0) for k,v in adapter.counters.items()}))
    rows=[]
    for name,allocation in methods.items():
        start=time.perf_counter()
        status('R1_terminal',method=name)
        parent = allocation['parent_ids']
        method_before=dict(adapter.counters)
        state41=transition(adapter,checkpoint[parent],eps[parent],40,tape.at(40,ids))
        terminal=adapter.advance(state41,41,100,ids,tape)
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
                 nominal_cold_sampling_nfe=2400,aux_unet_nfe=0,
                 event_prediction_included_in_denoising_nfe=True,
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
                all_counters={k:v-before.get(k,0) for k,v in adapter.counters.items()},
                future_antithetic_or_sweep='NOT_RUN')
    atomic_json(output/'COMPLETE.json',result)
    return result
