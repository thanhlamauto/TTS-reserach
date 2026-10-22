import argparse
import time
from pathlib import Path
import numpy as np
import torch

from sd15_sinkhorn_tpu.common import atomic_json,load_config
from sd15_sinkhorn_tpu.collect import load_tensor,save_images
from sd15_sinkhorn_tpu.sd15_sampler import Sampler,ddim_update
from sd15_sinkhorn_tpu.reward import ImageRewardCPU,CLIPCPU,evaluate_images
from sd15_sinkhorn_tpu.evaluate import summary_metrics
from sd15_sinkhorn_tpu.tpu_runtime import Runtime
from sd15_sinkhorn_bfs_compare.analyze import read
from .stage0 import DCS,PRIOR,sha
from .core import METHODS,noise_tape,tensor_sha


@torch.no_grad()
def suffix(sampler,prompt,initial,tape):
    assert sampler.cfg['steps']==100 and sampler.cfg['ddim_eta']==1
    assert initial.dtype==torch.bfloat16 and tape.dtype==torch.float32
    assert tape.shape==(50,len(initial),4,64,64)
    embed=sampler.embedding(prompt); final=[]; begin=time.perf_counter()
    for offset in range(0,len(initial),sampler.batch):
        x=initial[offset:offset+sampler.batch].to(sampler.rt.device)
        assert len(x)==sampler.batch # fixed N8 / microbatch4; no padding
        for k in range(50,100):
            timestep=torch.tensor(sampler.timesteps[k],dtype=torch.int64,device=sampler.rt.device)
            pred=sampler.pipe.unet(torch.cat([x,x]),timestep,encoder_hidden_states=embed,return_dict=False)[0]
            uncond,cond=pred.chunk(2)
            eps=uncond.float()+sampler.cfg['cfg_scale']*(cond.float()-uncond.float())
            coeff=sampler.coefficients[k].to(sampler.rt.device)
            variance=tape[k-50,offset:offset+sampler.batch].to(sampler.rt.device)
            x=ddim_update(x,eps,coeff,variance).to(torch.bfloat16)
            sampler.rt.xm.mark_step()
        final.append(x.cpu())
    sampler.rt.sync(); result=torch.cat(final)
    assert torch.isfinite(result).all()
    return result,time.perf_counter()-begin


def branch(sampler,reward,clip,root,prompt,index,rep,name,initial,tape):
    out=Path(root)/'full'/f'rep{rep}'/f'p{index:03d}'/name
    out.mkdir(parents=True,exist_ok=True); result_path=out/'result.json'
    tape_sha=tensor_sha(tape); initial_sha=tensor_sha(initial)
    if result_path.exists():
        r=read(result_path)
        assert r['noise_tape_sha256']==tape_sha and r['initial_sha256']==initial_sha
        return r
    final_path=out/'final.pt'
    if final_path.exists():
        data=load_tensor(final_path); final,denoise=data['latents'],data['denoise_seconds']
        assert data['noise_tape_sha256']==tape_sha and data['initial_sha256']==initial_sha
    else:
        final,denoise=suffix(sampler,prompt,initial,tape)
        torch.save(dict(latents=final,denoise_seconds=denoise,noise_tape_sha256=tape_sha,
                        initial_sha256=initial_sha),final_path)
    start=time.perf_counter(); images=sampler.decode(final); decode=time.perf_counter()-start
    save_images(images,out/'images'); scored=evaluate_images(prompt,images,reward,clip)
    r=dict(method=name,prompt_index=index,prompt=prompt,replicate=rep,scores=scored,
           **summary_metrics(scored),noise_tape_sha256=tape_sha,initial_sha256=initial_sha,
           diffusion_sample_steps=400,denoise_seconds=denoise,decode_seconds=decode,
           wall_seconds=denoise+decode+scored['reward_seconds']+scored['clip_seconds'],
           finite=True,n=8)
    atomic_json(result_path,r); print('BRANCH_COMPLETE',rep,index,name,r['mean_reward'],flush=True)
    return r


def replay_gate(sampler,reward,root,prompt,initial,tape):
    root=Path(root); path=root/'replay_gate.json'
    if path.exists(): assert read(path)['passed']; return
    base=root/'full/rep0/p000/none'
    first=load_tensor(base/'final.pt')['latents']; scored=read(base/'result.json')['scores']['rewards']
    second,seconds=suffix(sampler,prompt,initial,tape)
    a,b=sampler.decode(first),sampler.decode(second)
    mae=float(np.mean([np.abs(np.asarray(x).astype('float32')-np.asarray(y).astype('float32')).mean() for x,y in zip(a,b)]))
    rewards=reward.score(prompt,b)
    delta=float((first.float()-second.float()).abs().max())
    reward_delta=float(np.max(np.abs(np.array(rewards)-np.array(scored))))
    passed=delta==0 and mae==0 and reward_delta==0
    atomic_json(path,dict(passed=passed,latent_max_abs=delta,image_mae=mae,
        reward_max_abs=reward_delta,new_sample_steps=400,new_reward_evaluations=8,
        noise_tape_sha256=tensor_sha(tape),seconds=seconds))
    if not passed: raise RuntimeError('paired suffix zero-update replay failed')
    print('REPLAY_GATE_EXACT',flush=True)


def run_prompt(sampler,reward,clip,root,plan,index,rep,smoke=False):
    control_path=Path(root)/'controls'/f'p{index:03d}'/f'rep{rep}.pt'
    control=load_tensor(control_path)
    assert control['max_rms_relative_error']<=plan['max_control_rms_relative_error']
    path=Path(root)/'noise'/f'p{index:03d}'/f'rep{rep}.pt'; path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): tape=load_tensor(path)['variance_noise']
    else:
        tape=noise_tape(plan['noise_root_seed'],index,rep)
        torch.save(dict(variance_noise=tape,sha256=tensor_sha(tape),protocol=plan['variance_noise_protocol']),path)
    for name in METHODS:
        branch(sampler,reward,clip,root,plan['prompts'][index],index,rep,name,control['latents'][name],tape)
        if smoke and name=='none':
            replay_gate(sampler,reward,root,plan['prompts'][index],control['latents']['none'],tape)


def run(args):
    root=Path(args.output); plan=read(root/'frozen_plan.json')
    assert read(root/'stage0_gate.json')['passed']
    for p,h in plan['source_sha256'].items(): assert sha(p)==h,p
    if not args.smoke: assert read(root/'smoke_gate.json')['passed']
    mode='smoke' if args.smoke else 'full'
    rt=Runtime(root/'runtime'/mode/f'worker{args.rank}',args.rank)
    cfg={**load_config(),'steps':100,'ddim_eta':1.}
    sampler=Sampler(rt,cfg,DCS); sampler.warmup(plan['prompts'][0])
    reward=ImageRewardCPU(DCS); clip=CLIPCPU(cfg,DCS)
    bank=read(PRIOR/'banks/test/p000/bank.json')
    assert sampler.timesteps==bank['timesteps']
    assigned=[0] if args.smoke else [i for i in plan['stage1_indices'] if i%args.workers==args.rank]
    try:
        for rep in ([0] if args.smoke else plan['replicates']):
            if rep==1:
                print('WAIT_REPLICATE0_GATE',flush=True)
                while not (root/'replicate0_gate.json').exists(): time.sleep(10)
                assert read(root/'replicate0_gate.json')['passed']
            for i in assigned:
                run_prompt(sampler,reward,clip,root,plan,i,rep,args.smoke)
                if not args.smoke:
                    atomic_json(root/'status'/f'rep{rep}_worker{args.rank}.json',
                        dict(state='running',last_prompt=i,assigned=assigned))
            if not args.smoke:
                atomic_json(root/'status'/f'rep{rep}_worker{args.rank}.json',dict(state='complete',assigned=assigned))
        if args.smoke:
            assert read(root/'replay_gate.json')['passed']
            atomic_json(root/'smoke_gate.json',dict(passed=True,prompt_index=0,replicate=0,methods=METHODS,
                         suffix_steps=50,stage1_outputs_reusable=True))
        atomic_json(root/'status'/f'{mode}_worker{args.rank}.json',dict(state='complete',
                    reward_evaluations_this_process=reward.evaluations))
    finally:
        (rt.output/'xla_metrics.txt').write_text(rt.metrics())


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True)
    p.add_argument('--smoke',action='store_true'); p.add_argument('--rank',type=int,default=0)
    p.add_argument('--workers',type=int,default=1); run(p.parse_args())


if __name__=='__main__': main()
