"""Original SD1.5 DDIM base continuations; no preview, edits or cached eps."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from PIL import Image,ImageDraw
from otlab.adapter import frozen_xla_factory,schedule_identity
from otlab.export import historical_tape
from otlab.noise import new_tape
from otlab.io import atomic_json,load_tensor,file_sha,tensor_sha,digest
from ot_next.identity import compatible_scheduler_metadata
from ot_next.features import RGBClient
from ot_next.output_selection import quality_diversity_indices,cosine_distance
from ot_next.e0 import mean_diversity
from .branching import allocation,anchors


class CompactRGB(RGBClient):
    def encode(self,rgb,label):
        result=super().encode(rgb,label)
        # Requests are transient IPC, not a second durable copy of images.
        # Input pixels and output features are persisted in root/method bundles.
        paths=[self.directory/f'{self.calls:03d}_{label}.{kind}.pt' for kind in ['input','output']]
        with (self.directory/'IPC_HASHES.jsonl').open('a') as f:
            f.write(json.dumps(dict(label=label,files={p.name:file_sha(p) for p in paths}))+'\n')
        for path in paths:path.unlink()
        return result


def unit(adapter,encoder,cfg,u,case,status):
    from sd15_sinkhorn_tpu.common import initial_noise
    from sd15_sinkhorn_tpu.reward import CLIPCPU
    case.mkdir()
    seed,pid=u['seed'],u['prompt_id']
    adapter.set_prompt(u['prompt'])
    initial,noise_seed=initial_noise(seed,pid,0,24)
    tape=historical_tape(adapter.sampler,noise_seed,24,start=0)
    ids=tuple(range(8))
    status('roots',prompt_id=pid)
    states20=adapter.advance(initial[:8],0,20,ids,tape)
    states40=adapter.advance(states20,20,40,ids,tape)
    root_end=adapter.advance(states40,40,100,ids,tape)
    trusted,cp,_=adapter.sampler.run(adapter.prompt,initial[:4],start=0,checkpoints=(20,40),noise_seed=noise_seed)
    adapter.counters['original_full_replay_sample_steps']+=400
    adapter.counters['unet_cfg_branch_sample_evaluations']+=800
    if not (torch.equal(root_end[:4],trusted) and torch.equal(states20[:4],cp[20]) and torch.equal(states40[:4],cp[40])):
        raise RuntimeError('full100 versus segmented20/40 root exact replay failed')
    atomic_json(case/'replay_gate.json',dict(passed=True,prefix20_bitwise=True,prefix40_bitwise=True,endpoint_bitwise=True,
                                             gate_nfe=400,prediction_cache_used=False))
    rr,root_rgb=adapter.score(root_end,'terminal')
    rf=encoder.encode(root_rgb,f'p{pid}_roots')
    torch.save(dict(initial=initial,noise_seed=noise_seed,states20=states20,states40=states40,
                    latent=root_end,rgb=root_rgb,rewards=rr,dino=rf,tape_sha=tensor_sha(tape.variance_noise)),case/'roots.pt')
    if not hasattr(adapter,'r2clip'):adapter.r2clip=CLIPCPU(adapter.sampler.cfg,cfg['dcs_root'])
    def finish(name,parts,logs,root_ids,levels):
        data={k:torch.cat([v[k] for v in parts]) for k in ['latent','rgb','rewards','dino']}
        clip=adapter.r2clip.score(adapter.prompt,[Image.fromarray(x.permute(1,2,0).numpy()) for x in data['rgb']])
        data['clip']=torch.tensor(clip['clip_embeddings'])
        data.update(root_ids=root_ids,levels=levels)
        torch.save(data,case/(name+'.pt'))
        atomic_json(case/(name+'.json'),dict(method=name,prompt_id=pid,seed=seed,pool_size=len(data['latent']),
            pool_mean_ir=float(data['rewards'].mean()),best_ir=float(data['rewards'].max()),
            terminal_sha=tensor_sha(data['latent']),root_ids=root_ids,levels=levels,allocation_logs=logs,
            cold_sampling_nfe=2400,cold_aux_unet_nfe=0,prediction_cache_used=False))
    root_data=dict(latent=root_end,rgb=root_rgb,rewards=rr,dino=rf)
    for name in cfg['branch_methods']:
        status('branch_method',prompt_id=pid,method=name)
        archive=[dict(root=i,direct_levels=[20,40],reward=float(rr[i]),feature=rf[i].numpy()) for i in range(8)]
        centers=anchors(rf.numpy(),rr.numpy(),cfg['calibration']['quality_threshold'],cfg['merge_distance'])
        parts=[root_data];logs=[];root_ids=list(range(8));levels=[0]*8
        for b,(level,count) in enumerate(cfg['batches']):
            uniforms=np.random.default_rng(np.random.SeedSequence([seed,pid,b,1729])).random(count)
            parents,log=allocation(name,level,count,archive,centers,cfg['calibration']['quality_threshold'],uniforms)
            # The identical child-slot tape is used for every method at this batch.
            child_tape=new_tape(seed,pid,100+b,tuple(range(count)),(4,64,64),adapter.schedule_sha,start=level)
            state=(states20 if level==20 else states40)[parents]
            end=adapter.advance(state,level,100,tuple(range(count)),child_tape)
            reward,rgb=adapter.score(end,'terminal')
            feature=encoder.encode(rgb,f'p{pid}_{name}_b{b}')
            log.update(batch=b,parents=parents,tape_sha=tensor_sha(child_tape.variance_noise),
                       frozen_anchors_sha=None if centers is None else digest(centers.tolist()),
                       batch_reward=reward.tolist())
            logs.append(log)
            # No feedback from a pending action can enter its own decision.
            archive.extend(dict(root=p,direct_levels=[level],reward=float(reward[j]),feature=feature[j].numpy())
                           for j,p in enumerate(parents))
            if centers is None:
                centers=anchors(np.stack([a['feature'] for a in archive]),np.array([a['reward'] for a in archive]),
                                cfg['calibration']['quality_threshold'],cfg['merge_distance'])
            parts.append(dict(latent=end,rgb=rgb,rewards=reward,dino=feature))
            root_ids.extend(parents);levels.extend([level]*count)
        finish(name,parts,logs,root_ids,levels)
    status('paired_BoN24',prompt_id=pid)
    extra=adapter.advance(initial[8:],0,100,tuple(range(8,24)),tape)
    extra_r,extra_rgb=adapter.score(extra,'terminal')
    extra_f=encoder.encode(extra_rgb,f'p{pid}_bon')
    finish('bon24',[root_data,dict(latent=extra,rgb=extra_rgb,rewards=extra_r,dino=extra_f)],[],list(range(24)),[0]*24)
    atomic_json(case/'COMPLETE.json',dict(passed=True,methods=cfg['methods'],root_replay=True,
                new_nfe=9200,cold_five_methods_nfe=12000,shared_root_nfe_saved=3200,gate_nfe=400,
                prediction_cache_used=False,semantic_review='PENDING'))


def worker(cfg,rank,source):
    output=Path(cfg['root'])/f'chip{rank}';output.mkdir()
    start=time.time();adapter=encoder=None
    def status(stage,**more):
        value=dict(stage=stage,rank=rank,pid=os.getpid(),elapsed_seconds=time.time()-start,**more)
        atomic_json(output/'status.json',value);print(json.dumps(value),flush=True)
    try:
        status('load_frozen_backend')
        old=Path(cfg['old_root'])/'attempts/8074f94c847d'
        ac=json.loads((old/'adapter_config.json').read_text())
        runtime=output/'runtime';runtime.mkdir()
        cache=Path('/dev/shm')/('ot_R2_'+cfg['protocol_sha'][:12])/f'chip{rank}';cache.mkdir(parents=True)
        (runtime/'xla_cache').symlink_to(cache,target_is_directory=True)
        ac['runtime_output']=str(runtime);adapter=frozen_xla_factory(ac)
        prior=json.loads((old/'neural_runtime/environment/sampler0.json').read_text())['scheduler_config']
        if not compatible_scheduler_metadata(prior,dict(adapter.sampler.scheduler.config)):raise RuntimeError('numeric scheduler changed')
        adapter.sampler.scheduler.register_to_config(_use_default_values=prior['_use_default_values'])
        expected=json.loads((old/'replay.json').read_text())['records'][0]['adapter_identity']['schedule_sha']
        if schedule_identity(adapter.sampler)!=expected:raise RuntimeError('full schedule authentication failed')
        adapter.schedule_sha=expected;adapter.base_identity['schedule_sha']=expected;adapter.set_prompt(adapter.prompt)
        encoder=CompactRGB(str(Path(cfg['old_root'])/'worker_venv/bin/python'),
                           source.parents[1]/'ot_tts_lab_v2_reimpl/configs/worker_dinov2.yaml',output/'rgb_worker')
        atomic_json(output/'identity.json',dict(adapter=adapter.identity,encoder=encoder.identity,visible_chip=os.environ['TPU_VISIBLE_CHIPS']))
        units=[u for i,u in enumerate(cfg['units']) if i%4==rank]
        for pos,u in enumerate(units):
            unit(adapter,encoder,cfg,u,output/f'p{u["prompt_id"]:03d}',status)
            if rank==0 and pos==0:
                atomic_json(Path(cfg['root'])/'SMOKE_COMPLETE.json',dict(passed=True,prompt_id=u['prompt_id'],
                    five_methods=True,base_replay=True,allocated_chip=0))
                while not (Path(cfg['root'])/'RELEASE_FULL.json').exists():time.sleep(1)
        receipt=dict(passed=True,rank=rank,units=[u['prompt_id'] for u in units],counters=dict(adapter.counters),
                     elapsed_seconds=time.time()-start,dino_images=encoder.count,dino_rpc_seconds=encoder.seconds)
        atomic_json(output/'COMPLETE.json',receipt);status('complete')
    except BaseException as exc:
        atomic_json(output/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),
                     counters={} if adapter is None else dict(adapter.counters)))
        raise
    finally:
        if encoder:encoder.close()
        if adapter:(output/'xla_metrics.txt').write_text(adapter.sampler.rt.metrics())


def export(cfg,root):
    review={};maps=[];inputs=[]
    for u in cfg['units']:
        cases=list(root.glob(f'chip*/p{u["prompt_id"]:03d}'))
        if len(cases)!=1:raise RuntimeError('case rectangle incomplete')
        case=cases[0]
        for name in cfg['methods']:
            data=load_tensor(case/(name+'.pt'))
            if len(data['latent'])!=(24 if name=='bon24' else 32):raise RuntimeError('wrong endpoint count')
            if not torch.isfinite(data['latent']).all():raise RuntimeError('nonfinite endpoint')
            selections=dict(top8_ir=torch.argsort(data['rewards'],descending=True,stable=True)[:8].tolist(),
                quality_diversity=quality_diversity_indices(data['rewards'].numpy(),data['dino'].numpy(),cfg['calibration']['quality_threshold']))
            for view,indices in selections.items():
                hashes=[]
                for i in indices:
                    sha=tensor_sha(data['rgb'][i]);hashes.append(sha)
                    review.setdefault((u['prompt_id'],sha),data['rgb'][i])
                inputs.append(dict(prompt_id=u['prompt_id'],seed=u['seed'],method=name,selector=view,rewards=data['rewards'][indices].tolist(),
                    clip_cosine_distance=cosine_distance(data['clip'][indices].numpy()).tolist(),
                    clip_diversity=mean_diversity(data['clip'][indices]),dino_diversity=mean_diversity(data['dino'][indices]),hashes=hashes))
            maps.append(dict(prompt_id=u['prompt_id'],method=name,case=str(case.relative_to(root)),
                             terminal_file_sha256=file_sha(case/(name+'.pt'))))
    ordered=sorted(review,key=lambda x:digest(dict(key=2026101705,prompt=x[0],pixels=x[1])))
    ids={key:f'J{i:03d}' for i,key in enumerate(ordered)}
    out=root/'blinded_review';out.mkdir()
    for u in cfg['units']:
        keys=[k for k in ordered if k[0]==u['prompt_id']]
        sheet=Image.new('RGB',(1152,214*((len(keys)+5)//6)),'white');draw=ImageDraw.Draw(sheet)
        for j,key in enumerate(keys):
            im=Image.fromarray(review[key].permute(1,2,0).numpy());im.save(out/(ids[key]+'.png'))
            x,y=j%6*192,j//6*214;sheet.paste(im.resize((192,192)),(x,y));draw.text((x+5,y+195),ids[key],fill='black')
        sheet.save(out/f'p{u["prompt_id"]:03d}.png')
    for item in inputs:
        item['image_ids']=[ids[(item['prompt_id'],h)] for h in item.pop('hashes')]
    atomic_json(root/'OUTPUT_EVAL_INPUTS.json',inputs)
    atomic_json(root/'BLINDED_IMAGE_INDEX.json',[dict(id=ids[k],prompt_id=k[0],sha=k[1]) for k in ordered])
    atomic_json(root/'OUTPUT_MAPS.json',maps)
    return dict(prompts=6,methods=5,selector_rows=len(inputs),unique_selected_images=len(ordered),
                terminal_candidates=912,semantic_complete=False,scientific_winner=None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--worker',type=int);args=p.parse_args()
    source=Path(args.protocol).resolve().parent;cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_sha']=file_sha(args.protocol)
    if args.worker is not None:return worker(cfg,args.worker,source)
    root=Path(cfg['root']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'NEURAL_COMPLETE.json').exists():return
    if list(root.glob('chip*')):raise RuntimeError('prior partial attempt exists; inspect and resume explicitly, never duplicate')
    owners=subprocess.run(['fuser','/dev/vfio/0','/dev/vfio/1','/dev/vfio/2','/dev/vfio/3'],capture_output=True,text=True)
    if owners.stdout.strip():raise RuntimeError('TPU devices occupied')
    if shutil.disk_usage(root).free<1500*1024**2:raise RuntimeError('less than1.5GiB free before compact run')
    for path,sha in json.loads((source/'old_input_manifest.json').read_text()).items():
        if file_sha(path)!=sha:raise RuntimeError('frozen input changed:'+path)
    atomic_json(root/'preregistered.json',cfg)
    begin=time.time();processes=[]
    try:
        for rank in range(4):
            if rank==1:
                while not (root/'SMOKE_COMPLETE.json').exists():
                    if processes[0][1].poll() is not None:raise RuntimeError('single-chip R2 smoke did not pass')
                    time.sleep(1)
                atomic_json(root/'RELEASE_FULL.json',dict(smoke_passed=True,expanded_unix=time.time()))
            log=(root/f'chip{rank}.log').open('w')
            child=subprocess.Popen([sys.executable,'-B','-u','-m','ot_remainder.r2','--protocol',args.protocol,'--worker',str(rank)],
                env=dict(os.environ,TPU_VISIBLE_CHIPS=str(rank)),stdout=log,stderr=subprocess.STDOUT)
            processes.append((rank,child,log))
        atomic_json(root/'worker_pids.json',{str(r):p.pid for r,p,l in processes})
        exits=[dict(rank=r,exit_code=p.wait()) for r,p,l in processes]
        atomic_json(root/'worker_exit_codes.json',exits)
        if any(e['exit_code'] for e in exits):raise RuntimeError('worker failed; preserve independent units')
        summary=export(cfg,root)
        summary.update(elapsed_seconds=time.time()-begin,worker_receipts=[json.loads((root/f'chip{r}/COMPLETE.json').read_text()) for r in range(4)],
                       passed=True,new_sampling_nfe=52800,gate_nfe=2400,total_new_nfe=55200,aux_unet_nfe=0,
                       cold_five_methods_nfe=72000,owner_attested_no_charge=True,actual_billing_usd=None)
        atomic_json(root/'NEURAL_COMPLETE.json',summary);print('NEURAL_COMPLETE',json.dumps(summary),flush=True)
    except BaseException as exc:
        atomic_json(root/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()));raise
    finally:
        for r,p,l in processes:
            if p.poll() is None:p.terminate()
            l.close()


if __name__=='__main__':main()
