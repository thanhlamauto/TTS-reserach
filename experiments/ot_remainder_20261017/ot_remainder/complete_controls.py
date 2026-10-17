"""Bounded paired TRAIN-only BFS frontier and single-event image Poisson pilot."""
import argparse,fcntl,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from otlab.adapter import frozen_xla_factory,schedule_identity
from otlab.export import historical_tape
from otlab.io import atomic_json,load_tensor,file_sha,tensor_sha,digest
from ot_next.identity import compatible_scheduler_metadata
from ot_next.features import RGBClient
from ot_next.output_selection import quality_diversity_indices,cosine_distance
from .poisson_image import edit,gates

class CompactRGB(RGBClient):
    def encode(self,rgb,label):
        result=super().encode(rgb,label)
        for kind in ['input','output']:(self.directory/f'{self.calls:03d}_{label}.{kind}.pt').unlink()
        return result

def case_for(cfg,pid):
    cases=list(Path(cfg['e1_root']).glob(f'chip*/p{pid:03d}'))
    assert len(cases)==1
    return cases[0]

def bfs_run(adapter,cfg,u,prefix,initial,noise_seed,method,status):
    from sd15_sinkhorn_bfs_compare.bfs import advance,make_adapter,policy
    record=policy('graph',100)
    record['temperature_by_step']=[(k,t*method['strength']) for k,t in record['temperature_by_step']]
    n=12 if method.get('hybrid') else 24
    def score(clean):
        images=adapter.sampler.decode(clean)
        values=torch.tensor(adapter.reward.score(adapter.prompt,images),dtype=torch.float32)
        adapter.counters['bfs_verifier_scores']+=len(images);adapter.counters['bfs_verifier_calls']+=1
        return values
    search=make_adapter(n,record,u['seed'],u['prompt_id'],score);search.resampling=method['resampling']
    x=prefix.clone();start=40;anc=torch.arange(24)
    for event in record['resampling_steps']:
        x,x0,_=advance(adapter.sampler,adapter.prompt,x,start,event+1,noise_seed)
        adapter.counters['bfs_denoise_sample_steps']+=24*(event+1-start)
        if method.get('hybrid'):
            child,_=search.resample(sampling_idx=event,latents=x[12:],x0_preds=x0[12:]);x=torch.cat([x[:12],child])
            ids=list(range(12))+[12+i for i in search.events[-1]['parent_indices']]
        else:
            x,_=search.resample(sampling_idx=event,latents=x,x0_preds=x0);ids=search.events[-1]['parent_indices']
        anc=anc[ids]
        log=search.events[-1];w=np.array(log['normalized_weights'])
        log.update(ess=float(1/(w@w)),offspring_counts=np.bincount(log['parent_indices'],minlength=n).tolist(),
                   unique_initial_ancestors=len(set(anc.tolist())),completed_updates=event+1,
                   strength=method['strength'],resampling=method['resampling'])
        start=event+1
    x,_,_=advance(adapter.sampler,adapter.prompt,x,start,100,noise_seed)
    adapter.counters['bfs_denoise_sample_steps']+=24*(100-start)
    return x,dict(events=search.events,root_ids=anc.tolist(),cold_sampling_nfe=2400,cold_aux_unet_nfe=0,
                  max_history_zero_initialization='original frozen adapter, unchanged')

def unit(adapter,encoder,cfg,u,case,status):
    from sd15_sinkhorn_bfs_compare.bfs import advance
    from sd15_sinkhorn_tpu.reward import CLIPCPU
    old=case_for(cfg,u['prompt_id']);saved=load_tensor(old/'prefix.pt');event=load_tensor(old/'event.pt')
    none=load_tensor(old/'none.pt');initial=saved['initial'];cp=saved['checkpoint'];ns=saved['noise_seed']
    assert saved['next_index']==40 and initial.dtype==torch.bfloat16
    adapter.set_prompt(u['prompt']);tape=historical_tape(adapter.sampler,ns,24,start=0)
    assert tensor_sha(tape.variance_noise)==saved['tape_sha']
    case.mkdir();status('paired_replay_gate',prompt_id=u['prompt_id'])
    trusted=adapter.advance(cp[:4],40,100,tuple(range(4)),tape)
    direct,_,_=advance(adapter.sampler,adapter.prompt,cp[:4],40,100,ns)
    adapter.counters['bfs_original_replay_sample_steps']+=240
    assert torch.equal(trusted,none['latent'][:4]),'original fused continuation replay'
    assert torch.equal(direct,trusted),'BFS original transition changed endpoint'
    atomic_json(case/'replay_gate.json',dict(passed=True,adapter_resume=True,bfs_resume=True,new_gate_nfe=480))
    if not hasattr(adapter,'clip'):adapter.clip=CLIPCPU(adapter.sampler.cfg,cfg['dcs_root'])
    for method in cfg['methods']:
        name=method['name'];status('terminal_method',prompt_id=u['prompt_id'],method=name)
        before=dict(adapter.counters);begin=time.time()
        if method['kind']=='bfs':
            end,log=bfs_run(adapter,cfg,u,cp,initial,ns,method,status)
            if method.get('hybrid'):assert torch.equal(end[:12],none['latent'][:12]),'hybrid independent half drifted'
        else:
            changed,log=edit(cp,event['reward'].numpy(),method['field'],method['bound'],
                             int(np.random.SeedSequence([u['seed'],u['prompt_id'],991]).generate_state(1)[0]))
            # Original base transition recalculates UNet on the edited state.
            end=adapter.advance(changed,40,100,tuple(range(24)),tape)
            log.update(cold_sampling_nfe=2400,cold_aux_unet_nfe=24,prediction_recomputed=True)
            torch.save(dict(state=changed,original_sha=tensor_sha(cp)),case/(name+'_edited_state.pt'))
        rewards,rgb=adapter.score(end,'terminal');feature=encoder.encode(rgb,f'p{u["prompt_id"]}_{name}')
        clip=adapter.clip.score(adapter.prompt,[Image.fromarray(im.permute(1,2,0).numpy()) for im in rgb])
        clip=torch.tensor(clip['clip_embeddings'])
        assert torch.isfinite(end).all() and torch.isfinite(rewards).all()
        torch.save(dict(latent=end,rgb=rgb,rewards=rewards,dino=feature,clip=clip),case/(name+'.pt'))
        log.update(method=name,prompt_id=u['prompt_id'],seed=u['seed'],terminal_sha=tensor_sha(end),
                   elapsed_seconds=time.time()-begin,counters={k:v-before.get(k,0) for k,v in adapter.counters.items()})
        atomic_json(case/(name+'.json'),log)
    atomic_json(case/'COMPLETE.json',dict(passed=True,methods=[m['name'] for m in cfg['methods']]))

def worker(cfg,rank,source):
    out=Path(cfg['root'])/f'chip{rank}';out.mkdir();begin=time.time();adapter=encoder=None
    def status(stage,**kw):
        r=dict(stage=stage,pid=os.getpid(),rank=rank,elapsed_seconds=time.time()-begin,**kw)
        atomic_json(out/'status.json',r);print(json.dumps(r),flush=True)
    def interrupted(signum,frame):raise TimeoutError('bounded worker interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        status('load_existing_backend');old=Path(cfg['old_root'])/'attempts/8074f94c847d'
        ac=json.loads((old/'adapter_config.json').read_text());runtime=out/'runtime';runtime.mkdir()
        cache=Path('/dev/shm')/('ot_controls_'+cfg['protocol_sha'][:12])/f'chip{rank}';cache.mkdir(parents=True)
        (runtime/'xla_cache').symlink_to(cache,target_is_directory=True);ac['runtime_output']=str(runtime)
        adapter=frozen_xla_factory(ac)
        prior=json.loads((old/'neural_runtime/environment/sampler0.json').read_text())['scheduler_config']
        assert compatible_scheduler_metadata(prior,dict(adapter.sampler.scheduler.config))
        adapter.sampler.scheduler.register_to_config(_use_default_values=prior['_use_default_values'])
        expected=json.loads((old/'replay.json').read_text())['records'][0]['adapter_identity']['schedule_sha']
        assert schedule_identity(adapter.sampler)==expected
        adapter.schedule_sha=expected;adapter.base_identity['schedule_sha']=expected;adapter.set_prompt(adapter.prompt)
        encoder=CompactRGB(str(Path(cfg['old_root'])/'worker_venv/bin/python'),
                           source.parents[1]/'ot_tts_lab_v2_reimpl/configs/worker_dinov2.yaml',out/'rgb_worker')
        atomic_json(out/'identity.json',dict(adapter=adapter.identity,encoder=encoder.identity,visible_chip=os.environ['TPU_VISIBLE_CHIPS']))
        for pos,u in enumerate(cfg['units'][rank::4]):
            unit(adapter,encoder,cfg,u,out/f'p{u["prompt_id"]:03d}',status)
            if rank==0 and pos==0:
                atomic_json(Path(cfg['root'])/'SMOKE_COMPLETE.json',dict(passed=True,all_methods=True))
                while not (Path(cfg['root'])/'RELEASE_FULL.json').exists():time.sleep(1)
        atomic_json(out/'COMPLETE.json',dict(passed=True,rank=rank,counters=dict(adapter.counters),
                   dino_images=encoder.count,elapsed_seconds=time.time()-begin));status('complete')
    except BaseException as exc:
        atomic_json(out/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()));raise
    finally:
        if encoder:encoder.close()
        if adapter:(out/'xla_metrics.txt').write_text(adapter.sampler.rt.metrics())

def export(cfg,root):
    review={};inputs=[];maps=[]
    for u in cfg['units']:
        case=next(root.glob(f'chip*/p{u["prompt_id"]:03d}'))
        methods=[(m['name'],case/(m['name']+'.pt')) for m in cfg['methods']]
        old=case_for(cfg,u['prompt_id'])
        methods += [(name,old/(name+'.pt')) for name in ['none','partial_reward','partial_group','partial_ot']]
        for name,path in methods:
            data=load_tensor(path);assert len(data['latent'])==24
            maps.append(dict(prompt_id=u['prompt_id'],method=name,path=str(path),sha256=file_sha(path)))
            views=dict(top8_ir=torch.argsort(data['rewards'],descending=True,stable=True)[:8].tolist(),
                       quality_diversity=quality_diversity_indices(data['rewards'].numpy(),data['dino'].numpy(),cfg['calibration']['quality_threshold']))
            for view,ids in views.items():
                hashes=[]
                for i in ids:
                    h=tensor_sha(data['rgb'][i]);hashes.append(h);review.setdefault((u['prompt_id'],h),data['rgb'][i])
                inputs.append(dict(prompt_id=u['prompt_id'],seed=u['seed'],method=name,selector=view,rewards=data['rewards'][ids].tolist(),
                    clip_cosine_distance=cosine_distance(data['clip'][ids].numpy()).tolist(),hashes=hashes,
                    cold_sampling_nfe=2400,cold_aux_unet_nfe=24 if name.startswith('r3_') or name.startswith('partial_') else 0))
    ordered=sorted(review,key=lambda k:digest(dict(key=2026101710,prompt=k[0],pixels=k[1])))
    ids={k:f'K{i:04d}' for i,k in enumerate(ordered)};out=root/'blinded_review';out.mkdir()
    for u in cfg['units']:
        keys=[k for k in ordered if k[0]==u['prompt_id']]
        for page in range((len(keys)+35)//36):
            subset=keys[page*36:(page+1)*36];sheet=Image.new('RGB',(1152,214*((len(subset)+5)//6)),'white');draw=ImageDraw.Draw(sheet)
            for j,k in enumerate(subset):
                x,y=j%6*192,j//6*214;sheet.paste(Image.fromarray(review[k].permute(1,2,0).numpy()).resize((192,192)),(x,y));draw.text((x+4,y+195),ids[k],fill='black')
            sheet.save(out/f'p{u["prompt_id"]:03d}_{page:02d}.jpg',quality=90)
    for row in inputs:row['image_ids']=[ids[(row['prompt_id'],h)] for h in row.pop('hashes')]
    atomic_json(root/'BLINDED_IMAGE_INDEX.json',[dict(id=ids[k],prompt_id=k[0],sha=k[1]) for k in ordered])
    atomic_json(root/'OUTPUT_EVAL_INPUTS.json',inputs);atomic_json(root/'OUTPUT_MAPS.json',maps)
    return dict(output_rows=len(inputs),unique_review_images=len(ordered),old_methods_reused=4,new_methods=len(cfg['methods']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--worker',type=int);a=p.parse_args()
    source=Path(a.protocol).resolve().parent;cfg=json.loads(Path(a.protocol).read_text());cfg['protocol_sha']=file_sha(a.protocol)
    if a.worker is not None:return worker(cfg,a.worker,source)
    root=Path(cfg['root']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'NEURAL_COMPLETE.json').exists():return
    assert not list(root.glob('chip*')),'existing partial attempt, inspect before resume'
    assert shutil.disk_usage(root).free>4*1024**3,'need4GiB free for all terminal tensors'
    for p,h in cfg['frozen_input_hashes'].items():assert file_sha(p)==h,p
    assert not subprocess.run(['fuser','/dev/vfio/0','/dev/vfio/1','/dev/vfio/2','/dev/vfio/3'],capture_output=True,text=True).stdout.strip(),'TPU owned'
    assert json.loads((Path(cfg['toy_gate'])/'COMPLETE.json').read_text())['nonlinear_gate']
    atomic_json(root/'NUMERICAL_GATES.json',gates());atomic_json(root/'preregistered.json',cfg)
    processes=[];begin=time.time()
    try:
        for rank in range(4):
            if rank==1:
                while not (root/'SMOKE_COMPLETE.json').exists():
                    assert processes[0][1].poll() is None,'single-chip smoke failed'
                    time.sleep(1)
                atomic_json(root/'RELEASE_FULL.json',dict(passed=True,time=time.time()))
            f=(root/f'chip{rank}.log').open('w');child=subprocess.Popen([sys.executable,'-B','-u','-m','ot_remainder.complete_controls','--protocol',a.protocol,'--worker',str(rank)],
                env=dict(os.environ,TPU_VISIBLE_CHIPS=str(rank)),stdout=f,stderr=subprocess.STDOUT);processes.append((rank,child,f))
        atomic_json(root/'worker_pids.json',{str(r):p.pid for r,p,f in processes})
        exits=[dict(rank=r,exit_code=p.wait()) for r,p,f in processes];atomic_json(root/'worker_exit_codes.json',exits)
        assert all(e['exit_code']==0 for e in exits),'worker failed, preserve successful units'
        summary=export(cfg,root)
        for p,h in cfg['frozen_input_hashes'].items():assert file_sha(p)==h,p
        summary.update(passed=True,elapsed_seconds=time.time()-begin,reserved_final_opened=False,scientific_winner=None,
            receipts=[json.loads((root/f'chip{r}/COMPLETE.json').read_text()) for r in range(4)],frozen_inputs_verified=len(cfg['frozen_input_hashes']))
        atomic_json(root/'NEURAL_COMPLETE.json',summary);print(json.dumps(summary),flush=True)
    except BaseException as exc:
        atomic_json(root/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()));raise
    finally:
        for r,p,f in processes:
            if p.poll() is None:p.terminate()
            f.close()

if __name__=='__main__':main()
