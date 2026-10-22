import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from sd15_sinkhorn_tpu.common import atomic_json,load_config,initial_noise,read_prompts,source_hashes
from sd15_sinkhorn_tpu.collect import collect_bank,validate_replay,load_tensor,save_images
from sd15_sinkhorn_tpu.evaluate import summary_metrics
from sd15_sinkhorn_tpu.reward import ImageRewardCPU,CLIPCPU,evaluate_images
from sd15_sinkhorn_tpu.sd15_sampler import Sampler,ddim_coefficients
from sd15_sinkhorn_tpu.steering import Config
from sd15_sinkhorn_tpu.sweep import trial,best_of_n
from sd15_sinkhorn_tpu.tpu_runtime import Runtime
from .bfs import advance,run_bfs,policy

OLD=Path('/home/thanhlamtba31/static-bfs-dcs/experiments/sd15-sinkhorn-v2')
DCS=Path('/home/thanhlamtba31/static-bfs-dcs')


def reconfigure(sampler,cfg,output):
    sampler.cfg=cfg
    sampler.scheduler.set_timesteps(cfg['steps'],device='cpu')
    sampler.timesteps=[int(t) for t in sampler.scheduler.timesteps]
    sampler.coefficients=[ddim_coefficients(sampler.scheduler,t,cfg['ddim_eta']) for t in sampler.timesteps]
    sampler.rt.output=Path(output); sampler.rt.output.mkdir(parents=True,exist_ok=True)
    atomic_json(sampler.rt.output/'schedule.json',dict(cfg=cfg,timesteps=sampler.timesteps,
                                                    scheduler=dict(sampler.scheduler.config)))


def segmented_gate(sampler,bank_dir,bank,out):
    out=Path(out)
    if out.exists():
        data=json.loads(out.read_text()); assert data['passed']; return data
    d=load_tensor(Path(bank_dir)/'source'/'trajectories.pt')
    x=d['initial']; start=0
    stops=[i+1 for i in policy('manual',sampler.cfg['steps'])['resampling_steps']]+[sampler.cfg['steps']]
    for stop in stops:
        x,_,_=advance(sampler,bank['prompt'],x,start,stop,bank['sets']['source']['seed']); start=stop
    delta=x.float()-d['final'].float()
    a,b=sampler.decode(x),sampler.decode(d['final'])
    mae=float(np.mean([np.abs(np.asarray(i).astype('float32')-np.asarray(j).astype('float32')).mean() for i,j in zip(a,b)]))
    passed=bool(torch.allclose(x.float(),d['final'].float(),atol=sampler.cfg['replay_latent_atol'],rtol=sampler.cfg['replay_latent_rtol']) and mae<=sampler.cfg['replay_image_mae_max'])
    result=dict(passed=passed,latent_max_abs=float(delta.abs().max()),image_mae=mae,
                n=len(x),sample_steps=len(x)*sampler.cfg['steps'],stops=stops)
    atomic_json(out,result); print('SEGMENTED_GATE',result,flush=True)
    if not passed: raise RuntimeError('BFS segment math/RNG differs from shared sampler')
    return result


def bfs_trial(sampler,reward,clip,prompt,index,seed,n,name,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    result_path=out/'result.json'
    if result_path.exists(): return json.loads(result_path.read_text())
    initial,pseed=initial_noise(seed,index,0,n)
    rec=policy(name,sampler.cfg['steps']); rec['particles']=n
    start=time.perf_counter(); final_path=out/'final.pt'
    if final_path.exists():
        data=load_tensor(final_path); final=data['latents']; events=data['events']; denoise=data['denoise_seconds']
    else:
        final,events,denoise=run_bfs(sampler,reward,prompt,initial,pseed,rec,seed,index)
        torch.save(dict(latents=final,events=events,denoise_seconds=denoise),final_path)
    images=sampler.decode(final); save_images(images,out/'images')
    scored=evaluate_images(prompt,images,reward,clip)
    order=np.argsort(scored['rewards'],kind='stable')[-min(8,n):]
    rounded=final.flatten(1)
    unique_latents=len(torch.unique(rounded,dim=0))
    record=dict(method=f'bfs_{name}_n{n}',policy=rec,prompt=prompt,prompt_index=index,seed=seed,
                initial_seed=pseed,initial_noise_sha256=hashlib.sha256(initial.view(torch.uint8).numpy().tobytes()).hexdigest(),
                steps=sampler.cfg['steps'],ddim_eta=sampler.cfg['ddim_eta'],n=n,scores=scored,
                **summary_metrics(scored,indices=order),all_population=summary_metrics(scored),
                selected_indices=order.tolist(),events=events,unique_final_latents=unique_latents,
                denoise_seconds=denoise,wall_seconds=time.perf_counter()-start,
                diffusion_sample_steps=n*sampler.cfg['steps'],reward_evaluations=n*(len(events)+1),
                unet_cfg_branch_evaluations=2*n*sampler.cfg['steps'],finite=bool(torch.isfinite(final).all()))
    atomic_json(result_path,record)
    print('BFS_RESULT',record['method'],index,record['mean_reward'],record['best_reward'],unique_latents,flush=True)
    return record


def provenance(root):
    root=Path(root); selection=json.loads((OLD/'selection_frozen_before_test.json').read_text())
    plan=dict(selection=selection,selection_rule='reuse previously frozen Sinkhorn configs; BFS policies from prior independent experiments, no new tuning',
              phases={'30eta0':dict(steps=30,eta=0,splits=['train','test'],bfs=[['manual',8],['manual',28],['graph',28]],budget=840),
                      '100eta1':dict(steps=100,eta=1,splits=['test'],bfs=[['manual',8],['manual',28],['graph',28]],budget=2800)},
              train_prompts=read_prompts('train'),test_prompts=read_prompts('test'),
              base_seed=load_config()['seed'],test_seed=load_config()['seed']+100000,
              source_sha256={**{'sd15_sinkhorn_tpu/'+k:v for k,v in source_hashes().items()},
                **{str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}},
              old_run=str(OLD),model=load_config()['model'],model_revision=load_config()['revision'],
              interpretation='Follow-up on previously evaluated prompts. Same sampler/RNG/precision within each phase; no cross-scheduler causal claim.',
              bfs_implementation='Existing StaticGraphBFSAdapter/PaperBFSAdapter and upstream SSP, shared BF16 UNet + FP32 DDIM arithmetic. Not bitwise historical pipeline reproduction.',
              schedule_transfer='ceil((original_zero_based_index+1)*steps/100)-1; preserve original tau; manual gamma=.024',
              output_set='top8 for N28 BFS and fresh BoN; all8 for Sinkhorn/N8 BFS. Also retain full population mean and best.')
    dest=root/'frozen_plan.json'
    if dest.exists() and json.loads(dest.read_text())!=plan: raise RuntimeError('frozen comparison plan changed')
    atomic_json(dest,plan)
    return plan


def run(args):
    root=Path(args.output); plan=json.loads((root/'frozen_plan.json').read_text())
    if not args.smoke:
        if not (root/'smoke_gate.json').exists() or not json.loads((root/'smoke_gate.json').read_text())['passed']:
            raise RuntimeError('single-chip paired smoke required')
    rt=Runtime(root/'runtime'/f'worker{args.rank}',args.rank)
    cfg=load_config(); sampler=Sampler(rt,cfg,DCS)
    reward=ImageRewardCPU(DCS); clip=CLIPCPU(cfg,DCS)
    selection=[Config(**c) for c in plan['selection']['configs']]
    try:
        for phase,steps,eta in [('30eta0',30,0.),('100eta1',100,1.)]:
            phase_out=root/('smoke' if args.smoke else 'full')/phase
            scfg={**cfg,'steps':steps,'ddim_eta':eta}
            if args.smoke: scfg={**scfg,'n_reference':8,'n_source':4,'checkpoints':[.5]}
            reconfigure(sampler,scfg,phase_out/f'worker{args.rank}')
            sampler.warmup(read_prompts('train' if args.smoke else 'test')[0])
            splits=['train'] if args.smoke else (['train','test'] if steps==30 else ['test'])
            for split in splits:
                prompts=read_prompts(split); seed=cfg['seed']+(100000 if split=='test' else 0)
                indices=[0] if args.smoke else [i for i in range(len(prompts)) if i%args.workers==args.rank]
                for ii,index in enumerate(indices):
                    prompt=prompts[index]
                    if steps==30:
                        suffix='m8_n4' if args.smoke else 'm16_n8'
                        bank_dir=OLD/'banks'/split/f'seed{seed}'/suffix/f'p{index:03d}'
                        bank=json.loads((bank_dir/'bank.json').read_text())
                    else:
                        bank_dir=phase_out/'banks'/split/f'p{index:03d}'
                        bank=collect_bank(sampler,reward,clip,bank_dir,prompt,index,seed,scfg['n_reference'],scfg['n_source'])
                        validate_replay(sampler,bank_dir,bank,steps//2)
                    # This gate checks stochastic eta1 noise after segmentation as well.
                    if ii==0: segmented_gate(sampler,bank_dir,bank,phase_out/f'gate_{split}_worker{args.rank}.json')
                    variants=[('manual',4),('graph',4)] if args.smoke else [('manual',8),('manual',28),('graph',28)]
                    for name,n in variants:
                        bfs_trial(sampler,reward,clip,prompt,index,seed,n,name,
                                  phase_out/split/'bfs'/f'p{index:03d}'/f'{name}_n{n}')
                    if steps==100:
                        for c in selection: trial(sampler,reward,clip,bank_dir,bank,c,phase_out/split,seed,index)
                        best_of_n(sampler,reward,clip,bank_dir,bank,steps//2,seed,index)
                    atomic_json(phase_out/f'status_worker{args.rank}.json',dict(state='running',split=split,last_prompt=index))
            atomic_json(phase_out/f'status_worker{args.rank}.json',dict(state='complete'))
            (sampler.rt.output/'xla_metrics.txt').write_text(rt.metrics())
        if args.smoke: atomic_json(root/'smoke_gate.json',dict(passed=True,steps=[30,100],etas=[0,1],segmentation=True,particle_count=True))
    finally:
        (root/f'xla_metrics_worker{args.rank}.txt').write_text(rt.metrics())


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True)
    p.add_argument('--prepare',action='store_true'); p.add_argument('--smoke',action='store_true')
    p.add_argument('--rank',type=int,default=0); p.add_argument('--workers',type=int,default=1)
    args=p.parse_args()
    if args.prepare: provenance(args.output)
    else: run(args)


if __name__=='__main__': main()
