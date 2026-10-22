import argparse
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from .common import atomic_json,digest,event_seed,initial_noise,load_config,read_prompts,source_hashes
from .collect import collect_bank,load_tensor,save_images,validate_replay
from .evaluate import summary_metrics
from .reward import ImageRewardCPU,CLIPCPU,evaluate_images
from .sd15_sampler import Sampler,checkpoint_step
from .steering import Config,intervene
from .tpu_runtime import Runtime


def configurations(stage,root):
    if stage=='smoke':
        return ([Config(method=m,mode=mode,eta=.1,gamma_neg=.2 if mode=='contrast' else 0)
                 for m in 'ABC' for mode in ('soft','top','contrast')]
                +[Config(beta=0),Config(method='nearest')])
    if stage=='compare':
        return ([Config(method=m,mode=mode,eta=eta)
                 for m in 'ABC' for mode in ('soft','top') for eta in (.1,.3)]
                +[Config(method='nearest',eta=eta) for eta in (.1,.3)])
    path=Path(root)/'selection.json'
    if not path.exists(): raise RuntimeError('freeze TRAIN-only selection using analyze --select first')
    selected=json.loads(path.read_text())
    if selected['selection_split']!='train': raise RuntimeError('selection must use TRAIN only')
    base=[Config(**c) for c in selected['configs']]
    if stage=='heldout': return base
    if stage=='focused':
        best=base[0]
        # One-factor sweeps, no Cartesian product; duplicate baseline removed below.
        configs=[best]
        fields=dict(checkpoint=[.35,.5,.65],epsilon=[.03,.1,.3],beta=[.5,1,2],
                    top_fraction=[.25,.5],gamma_neg=[0,.2],cost=['raw','pooled'],trust_region=[.1,.2])
        for key,values in fields.items():
            for v in values:
                if key=='gamma_neg': configs.append(replace(best,mode='contrast',gamma_neg=v))
                elif key=='beta': configs.append(replace(best,mode='soft',beta=v))
                elif key=='top_fraction': configs.append(replace(best,mode='top',top_fraction=v))
                else: configs.append(replace(best,**{key:v}))
        return list({digest(c.record()):c for c in configs}.values())
    raise ValueError(stage)


def trial(sampler,reward,clip,bank_dir,bank,cfg,stage_dir,seed,index):
    config=cfg.record(); cid=digest(config)[:12]
    out=Path(stage_dir)/'trials'/f'p{index:03d}'/cid
    row_path=out/'result.json'
    if row_path.exists():
        row=json.loads(row_path.read_text())
        if row['bank_identity']!=bank['identity']: raise RuntimeError('stale trial')
        return row
    completed=checkpoint_step(cfg.checkpoint,sampler.cfg['steps'])
    validate_replay(sampler,bank_dir,bank,completed)
    source=load_tensor(Path(bank_dir)/'source'/'trajectories.pt')
    reference=load_tensor(Path(bank_dir)/'reference'/'trajectories.pt')
    x,z=source['checkpoints'][completed],reference['checkpoints'][completed]
    start=time.perf_counter(); y,diagnostics=intervene(x,z,bank['sets']['reference']['rewards'],cfg,event_seed(seed,index,completed))
    ot_s=time.perf_counter()-start
    # Quantization happens exactly once when re-entering the BF16 denoising loop.
    diagnostics['bf16_update_rms']=float((y.to(torch.bfloat16).float()-x.float()).square().mean().sqrt())
    out.mkdir(parents=True,exist_ok=True)
    torch.save(dict(steered=y,diagnostics=diagnostics),out/'intervention.pt')
    final_path=out/'final.pt'
    if final_path.exists():
        cached=load_tensor(final_path); final=cached['latents']; denoise_s=cached['seconds']
    else:
        final,_,denoise_s=sampler.run(bank['prompt'],y,start=completed,noise_seed=bank['sets']['source']['seed'])
        torch.save(dict(latents=final,seconds=denoise_s),final_path)
    start=time.perf_counter(); images=sampler.decode(final); decode_s=time.perf_counter()-start
    save_images(images,out/'images')
    scored=evaluate_images(bank['prompt'],images,reward,clip)
    n,m=bank['n_source'],bank['n_reference']; steps=sampler.cfg['steps']
    logical=(m+n)*steps+n*(steps-completed)
    row=dict(config_id=cid,config=config,prompt=bank['prompt'],prompt_index=index,seed=seed,
             bank_identity=bank['identity'],**diagnostics,**summary_metrics(scored,bank['sets']['source']['rewards']),
             scores=scored,ot_seconds=ot_s,denoise_seconds=denoise_s,decode_seconds=decode_s,
             wall_seconds=ot_s+denoise_s+decode_s+scored['reward_seconds']+scored['clip_seconds'],
             shared_bank_seconds=sum(s['denoise_seconds']+s['decode_seconds']+s['reward_seconds']+s['clip_seconds'] for s in bank['sets'].values()),
             incremental_sample_steps=n*(steps-completed),logical_total_sample_steps=logical,
             logical_unet_cfg_branch_evaluations=2*logical,reward_evaluations=m+2*n,
             incremental_reward_evaluations=n,reference_sample_steps=m*steps,
             bf16_finite=bool(torch.isfinite(final).all()),actual_padded_batch_size=sampler.batch)
    if cfg.method=='A' and cfg.mode=='soft' and cfg.beta==0:
        delta=final.float()-source['final'].float()
        row['zero_beta_final_max_abs']=float(delta.abs().max())
        if diagnostics['update_rms']>1e-7 or float(delta.abs().max())>sampler.cfg['replay_latent_atol']:
            raise RuntimeError('A beta=0 invariance failed')
    atomic_json(row_path,row)
    print('TRIAL',index,cfg,row['mean_reward'],row['reward_gain'],flush=True)
    return row


def best_of_n(sampler,reward,clip,bank_dir,bank,completed,seed,index):
    steps=sampler.cfg['steps']; n=bank['n_source']; m=bank['n_reference']
    budget=(m+n)*steps+n*(steps-completed)
    count=budget//steps
    out=Path(bank_dir)/f'fresh_bon_{count}'; marker=out/'result.json'
    if marker.exists(): return json.loads(marker.read_text())
    out.mkdir(parents=True,exist_ok=True)
    path=out/'final.pt'
    initial,pseed=initial_noise(seed,index,2,count)
    if pseed in [s['seed'] for s in bank['sets'].values()]: raise RuntimeError('fresh control seed collision')
    if path.exists():
        data=load_tensor(path); final,denoise_s=data['latents'],data['seconds']
    else:
        final,_,denoise_s=sampler.run(bank['prompt'],initial,noise_seed=pseed)
        torch.save(dict(latents=final,seconds=denoise_s),path)
    start=time.perf_counter(); images=sampler.decode(final); decode_s=time.perf_counter()-start
    save_images(images,out/'images'); scored=evaluate_images(bank['prompt'],images,reward,clip)
    order=np.argsort(scored['rewards'],kind='stable')[-n:]
    # Full batch and selected top-N are both reported; best is selected over all fresh draws.
    result=dict(config_id=f'fresh_bon_{count}_top{n}',config={'method':'fresh_bon'},prompt_index=index,
                seed=seed,prompt=bank['prompt'],**summary_metrics(scored,indices=order),scores=scored,
                selected_indices=order.tolist(),fresh_count=count,seed_fresh=pseed,
                logical_total_sample_steps=count*steps,budget_unused_sample_steps=budget-count*steps,
                physical_padded_sample_steps=math.ceil(count/sampler.batch)*sampler.batch*steps,
                reward_evaluations=count,incremental_reward_evaluations=count,
                incremental_sample_steps=count*steps,denoise_seconds=denoise_s,decode_seconds=decode_s,
                wall_seconds=denoise_s+decode_s+scored['reward_seconds']+scored['clip_seconds'],
                unselected=summary_metrics(scored),paired_improved_fraction=None,reward_gain=None)
    atomic_json(marker,result); print('FRESH_BON',index,result['mean_reward'],count,flush=True)
    return result


def run_worker(rank,args):
    root=Path(args.output); cfg=load_config(args.config)
    split='test' if args.stage=='heldout' else 'train'
    prompts=read_prompts(split)
    if args.stage=='smoke':
        cfg={**cfg,'n_reference':8,'n_source':4,'checkpoints':[.5]}; prompts=prompts[:1]
    seed=cfg['seed']+(100000 if split=='test' else 0)
    workers=args.workers; worker_rank=rank if args.multi else args.rank
    configs=configurations(args.stage,root)
    stage_dir=root/args.stage
    if args.stage!='smoke':
        gate=root/'smoke'/'gate.json'
        if not gate.exists() or not json.loads(gate.read_text())['passed']:
            raise RuntimeError('passing smoke gate required')
    plan=dict(stage=args.stage,split=split,prompts=prompts,seed=seed,sampler=cfg,
              configs=[c.record() for c in configs],code_sha256=source_hashes())
    stage_dir.mkdir(parents=True,exist_ok=True)
    # A worker writes only its own plan/status; immutable shared design must agree.
    planpath=stage_dir/f'plan_worker{worker_rank}.json'
    if planpath.exists() and json.loads(planpath.read_text())!=plan:
        raise RuntimeError('plan changed; choose a new output directory')
    atomic_json(planpath,plan)
    assigned=[i for i in range(len(prompts)) if i%workers==worker_rank]
    if args.max_prompts: assigned=assigned[:args.max_prompts]
    if not assigned: return
    rt=Runtime(root/args.stage/f'worker{worker_rank}',worker_rank)
    sampler=Sampler(rt,cfg,args.dcs_root)
    sampler.warmup(prompts[assigned[0]])
    reward=ImageRewardCPU(args.dcs_root); clip=CLIPCPU(cfg,args.dcs_root)
    try:
        for index in assigned:
            bank_dir=root/'banks'/split/f'seed{seed}'/f'm{cfg["n_reference"]}_n{cfg["n_source"]}'/f'p{index:03d}'
            bank=collect_bank(sampler,reward,clip,bank_dir,prompts[index],index,seed,cfg['n_reference'],cfg['n_source'])
            validate_replay(sampler,bank_dir,bank,checkpoint_step(.5,cfg['steps']))
            if args.action!='collect':
                for c in configs: trial(sampler,reward,clip,bank_dir,bank,c,stage_dir,seed,index)
                for step in sorted({checkpoint_step(c.checkpoint,cfg['steps']) for c in configs}):
                    best_of_n(sampler,reward,clip,bank_dir,bank,step,seed,index)
            atomic_json(stage_dir/f'status_worker{worker_rank}.json',dict(state='running',completed_prompt=index,rank=worker_rank))
        if args.stage=='smoke' and args.action!='collect':
            atomic_json(stage_dir/'gate.json',dict(passed=True,prompt=prompts[0],seed=seed,configs=len(configs),
                                                  replay=True,beta_zero=True,all_modes=True))
        atomic_json(stage_dir/f'status_worker{worker_rank}.json',dict(state='complete',prompt_indices=assigned,rank=worker_rank,
                                                                  verifier_evaluations_this_process=reward.evaluations))
    finally:
        (rt.output/'xla_metrics.txt').write_text(rt.metrics())


def main(default_action='run'):
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['smoke','compare','focused','heldout'],default='smoke')
    p.add_argument('--action',choices=['run','collect'],default=default_action)
    p.add_argument('--output',required=True); p.add_argument('--dcs-root',default='/home/thanhlamtba31/static-bfs-dcs')
    p.add_argument('--config'); p.add_argument('--workers',type=int,default=1)
    p.add_argument('--rank',type=int,default=0); p.add_argument('--multi',action='store_true')
    p.add_argument('--max-prompts',type=int)
    args=p.parse_args()
    if args.multi:
        from .tpu_runtime import launch
        launch(run_worker,(args,))
    else: run_worker(args.rank,args)


if __name__=='__main__': main()
