"""Frozen mechanism rectangle, shared actual work with explicit cold recipe costs."""
import argparse,fcntl,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
import torch
import numpy as np
from PIL import Image,ImageDraw
from otlab.adapter import frozen_xla_factory,schedule_identity
from otlab.export import historical_tape
from otlab.noise import NoiseTape,new_tape
from otlab.io import atomic_json,file_sha,tensor_sha
from ot_next.identity import compatible_scheduler_metadata
from noise_avoid.campaign import CompactRGB
from .core import rms,direction,capped_state,matched_random,tests

def diff(a,b):return {k:v-b.get(k,0) for k,v in a.items()}

def sheet(rgb,rewards,path):
    canvas=Image.new('RGB',(6*160,4*190),'white');draw=ImageDraw.Draw(canvas)
    for parent in range(4):
        scout=4+parent*8+int(rewards[4+parent*8:12+parent*8].argmax())
        for col,(index,label) in enumerate([(parent,'base'),(scout,'best scout')]+[(36+k*4+parent,n) for k,n in enumerate(['cov025','rand025','cov050','rand050'])]):
            im=Image.fromarray(rgb[index].permute(1,2,0).numpy()).resize((160,160));x,y=col*160,parent*190
            canvas.paste(im,(x,y));draw.text((x+2,y+161),f'{label} IR{float(rewards[index]):.3f}',fill='black')
    canvas.save(path,quality=90)

def unit(adapter,encoder,cfg,u,case,status,smoke):
    from sd15_sinkhorn_tpu.common import initial_noise
    from sd15_sinkhorn_tpu.reward import CLIPCPU
    case.mkdir(exist_ok=False);start=time.time();before=dict(adapter.counters)
    seed,pid=u['seed'],u['prompt_id'];adapter.set_prompt(u['prompt'])
    initial,noise_seed=initial_noise(seed,pid,0,4)
    history=historical_tape(adapter.sampler,noise_seed,4,start=0)
    status('prefix50',unit=u['unit_id']);parent=adapter.advance(initial,0,50,tuple(range(4)),history)
    status('diffusion_prior_roundtrip',unit=u['unit_id'])
    a50=float(adapter.sampler.coefficients[50][0])**2;a45=float(adapter.sampler.coefficients[45][0])**2;q=a45/a50
    assert 0<q<1
    g=torch.Generator().manual_seed(int(np.random.SeedSequence([seed,pid,210133]).generate_state(1)[0]))
    proposal_noise=torch.randn((32,4,64,64),generator=g)
    forward=(q**.5*parent.float().repeat_interleave(8,0)+(1-q)**.5*proposal_noise).to(torch.bfloat16)
    proposal_tape=new_tape(seed,pid,210145,tuple(range(32)),(4,64,64),adapter.schedule_sha,start=45)
    z=adapter.advance(forward,45,50,tuple(range(32)),proposal_tape)
    shared_state=torch.cat([parent,z]);owner=[0,1,2,3]+[p for p in range(4) for k in range(8)]
    shared_noise=history.variance_noise[50:,owner].clone()
    shared_tape=NoiseTape(shared_noise,50,tuple(range(36)),adapter.schedule_sha)
    status('shared_baseline_and_scout_rollouts36',unit=u['unit_id'])
    endpoint=adapter.advance(shared_state,50,100,tuple(range(36)),shared_tape)
    rewards,rgb=adapter.score(endpoint,'terminal')
    if smoke:
        direct,checkpoints,_=adapter.sampler.run(adapter.prompt,initial,start=0,checkpoints=(50,),noise_seed=noise_seed)
        adapter.counters['original_full_replay_sample_steps']+=400
        adapter.counters['unet_cfg_branch_sample_evaluations']+=800
        assert torch.equal(parent,checkpoints[50]) and torch.equal(endpoint[:4],direct),'Original full100 vs prefix/resume mismatch'
        atomic_json(case/'REPLAY.json',dict(passed=True,cases=4,prefix50_bitwise=True,endpoint_bitwise=True,extra_nfe=400))
    proposed={n:[] for n in cfg['methods'][1:]};logs=[]
    for p in range(4):
        local=z[p*8:(p+1)*8];v,info=direction(parent[p],local,rewards[4+p*8:12+p*8],float(rewards[p]),cfg)
        rng_seed=int(np.random.SeedSequence([seed,pid,p,210156]).generate_state(1)[0])
        info.update(parent=p,proposals={})
        for ratio,cn,rn in [(.25,'cov025','random025'),(.5,'cov050','random050')]:
            corrected,log=capped_state(parent[p],v,ratio*info['r_Q'],cfg['eta'])
            random,rlog=matched_random(parent[p],local,log['actual_rms'],torch.Generator().manual_seed(rng_seed))
            proposed[cn].append(corrected);proposed[rn].append(random)
            info['proposals'][cn]=log;info['proposals'][rn]=rlog
        logs.append(info)
    states=torch.stack([s for n in cfg['methods'][1:] for s in proposed[n]])
    correction_tape=NoiseTape(history.variance_noise[50:,[0,1,2,3]*4].clone(),50,tuple(range(16)),adapter.schedule_sha)
    status('covariance_and_matched_random16',unit=u['unit_id'])
    corrected_final=adapter.advance(states,50,100,tuple(range(16)),correction_tape)
    cr,cim=adapter.score(corrected_final,'terminal')
    all_rewards=torch.cat([rewards,cr]);all_rgb=torch.cat([rgb,cim]);all_final=torch.cat([endpoint,corrected_final])
    assert len(all_rewards)==52 and torch.isfinite(all_rewards).all() and torch.isfinite(all_final).all()
    for k,name in enumerate(cfg['methods'][1:]):
        for p in range(4):
            if not logs[p]['active'] or logs[p]['proposals'][name]['actual_rms']==0:
                assert torch.equal(corrected_final[k*4+p],endpoint[p]),'Zero correction changed paired endpoint'
    status('independent_terminal_features52',unit=u['unit_id'])
    features=encoder.encode(all_rgb,u['unit_id'])
    if not hasattr(adapter,'clip'):adapter.clip=CLIPCPU(adapter.sampler.cfg,cfg['dcs_root'])
    clip=adapter.clip.score(adapter.prompt,[Image.fromarray(im.permute(1,2,0).numpy()) for im in all_rgb])
    clip=torch.tensor(clip['clip_embeddings'])
    from .analyze import measure
    rows=measure(cfg,u,all_rewards,features,clip,logs)
    torch.save(dict(parent=parent,scout_states=z,correction_states=states,final=all_final,rgb=all_rgb,
        rewards=all_rewards,dino=features,clip=clip,initial_noise=initial,noise_seed=noise_seed,
        schedule_sha=adapter.schedule_sha),case/'raw.pt')
    atomic_json(case/'DIRECTION.json',logs)
    atomic_json(case/'PAIRING.json',dict(parent_future_tape_sha=[tensor_sha(history.variance_noise[50:,p:p+1]) for p in range(4)],
        scout_and_correction_owners=owner+[0,1,2,3]*4,proposal_noise_sha=tensor_sha(proposal_noise),
        proposal_tape_sha=tensor_sha(proposal_tape.variance_noise),q=q,alpha_index45=a45,alpha_index50=a50,
        all_roots_independent=True,all_comparisons_within_parent_paired=True))
    sheet(all_rgb,all_rewards,case/'comparison.jpg')
    atomic_json(case/'metrics.json',dict(unit=u,rows=rows))
    counters=diff(adapter.counters,before);nfe=counters['denoise_sample_steps']+counters.get('original_full_replay_sample_steps',0)
    assert nfe==2960+(400 if smoke else 0),counters
    assert counters.get('denoise_padding_sample_steps',0)==0 and counters.get('aux_clean_sample_evaluations',0)==0
    assert counters['unet_cfg_branch_sample_evaluations']==2*nfe
    assert counters['terminal_verifier_scores']==52
    atomic_json(case/'COMPLETE.json',dict(passed=True,unit=u,actual_nfe=nfe,counters=counters,
        raw_sha256=file_sha(case/'raw.pt'),elapsed_seconds=time.time()-start,terminal_scores=52,
        no_correction_cold_nfe=2160,each_corrected_cold_nfe=2360))
    status('unit_complete',unit=u['unit_id'],actual_nfe=nfe)

def worker(cfg,rank,source):
    out=Path(cfg['root'])/f'chip{rank}';out.mkdir(exist_ok=True);begin=time.time();adapter=encoder=None
    def status(stage,**kw):
        r=dict(stage=stage,rank=rank,pid=os.getpid(),elapsed_seconds=time.time()-begin,**kw)
        atomic_json(out/'status.json',r);print(json.dumps(r),flush=True)
    def interrupted(sig,frame):raise TimeoutError('Persistent neural deadline reached')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        status('load_original_backend')
        old=Path(cfg['old_root'])/'attempts/8074f94c847d';ac=json.loads((old/'adapter_config.json').read_text())
        ac['runtime_output']=str(out/'runtime');adapter=frozen_xla_factory(ac)
        prior=json.loads((old/'neural_runtime/environment/sampler0.json').read_text())['scheduler_config']
        assert compatible_scheduler_metadata(prior,dict(adapter.sampler.scheduler.config))
        adapter.sampler.scheduler.register_to_config(_use_default_values=prior['_use_default_values'])
        expected=json.loads((old/'replay.json').read_text())['records'][0]['adapter_identity']['schedule_sha']
        assert schedule_identity(adapter.sampler)==expected
        adapter.schedule_sha=expected;adapter.base_identity['schedule_sha']=expected
        encoder=CompactRGB(str(Path(cfg['old_root'])/'worker_venv/bin/python'),source.parents[1]/'ot_tts_lab_v2_reimpl/configs/worker_dinov2.yaml',out/'rgb_worker')
        atomic_json(out/'identity.json',dict(adapter=adapter.identity,encoder=encoder.identity,visible_chip=os.environ['TPU_VISIBLE_CHIPS']))
        for pos,u in enumerate(cfg['units'][rank::4]):
            unit(adapter,encoder,cfg,u,out/u['unit_id'],status,rank==0 and pos==0)
            if rank==0 and pos==0:
                atomic_json(Path(cfg['root'])/'SMOKE_COMPLETE.json',dict(passed=True,replay=4,nfe=3360))
                while not (Path(cfg['root'])/'RELEASE_FULL.json').exists():time.sleep(1)
        atomic_json(out/'COMPLETE.json',dict(passed=True,elapsed_seconds=time.time()-begin,counters=dict(adapter.counters)))
    except BaseException as exc:
        atomic_json(out/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),counters={} if adapter is None else dict(adapter.counters)));raise
    finally:
        if encoder:encoder.close()
        if adapter:(out/'xla_metrics.txt').write_text(adapter.sampler.rt.metrics())

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--worker',type=int);a=p.parse_args()
    source=Path(a.protocol).resolve().parent;cfg=json.loads(Path(a.protocol).read_text());root=Path(cfg['root']);root.mkdir(parents=True,exist_ok=True)
    if a.worker is not None:return worker(cfg,a.worker,source)
    lock=(root/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'COMPLETE.json').exists():print('ALREADY_COMPLETE');return
    assert not list(root.glob('chip*')),'Existing attempt: inspect; no duplicate launch'
    assert shutil.disk_usage(root).free>3*1024**3
    owners=subprocess.run(['fuser']+[f'/dev/vfio/{r}' for r in range(4)],capture_output=True,text=True)
    assert not owners.stdout.strip(),'TPU already owned'
    atomic_json(root/'STATIC_GATES.json',tests());atomic_json(root/'preregistered.json',cfg)
    atomic_json(root/'SOURCE_HASHES.json',{str(p.relative_to(source)):file_sha(p) for p in source.rglob('*') if p.is_file()})
    processes=[];begin=time.time()
    def interrupted(sig,frame):raise TimeoutError('Campaign deadline')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for rank in range(4):
            if rank==1:
                while not (root/'SMOKE_COMPLETE.json').exists():
                    assert processes[0][0].poll() is None,'Single-chip smoke failed'
                    time.sleep(1)
                atomic_json(root/'RELEASE_FULL.json',dict(released_unix=time.time()));print('SMOKE_PASS_RELEASE4',flush=True)
            log=(root/f'chip{rank}.log').open('a');child=subprocess.Popen([sys.executable,'-B','-u','-m','cov_repair.campaign','--protocol',a.protocol,'--worker',str(rank)],env=dict(os.environ,TPU_VISIBLE_CHIPS=str(rank)),stdout=log,stderr=subprocess.STDOUT)
            processes.append((child,log));atomic_json(root/'worker_pids.json',[c.pid for c,l in processes])
        exits=[c.wait() for c,l in processes];assert not any(exits),f'Worker failures{exits}'
        receipts=[json.loads(p.read_text()) for p in root.glob('chip*/p*/COMPLETE.json')]
        assert len(receipts)==12 and sum(r['actual_nfe'] for r in receipts)==cfg['neural_nfe_cap']
        from .analyze import aggregate
        result=aggregate(root,cfg)
        atomic_json(root/'COMPLETE.json',dict(passed=True,units=12,parents=48,prompts=6,seeds_per_prompt=2,
            actual_sampling_nfe=cfg['neural_nfe_cap'],cold_all_recipes_nfe=139200,padding=0,aux_unet=0,
            replay_nfe=400,terminal_ir_scores=624,clip_dino_images=624,
            cfg_branch_evaluations=2*cfg['neural_nfe_cap'],elapsed_seconds=time.time()-begin,
            occupied_worker_chip_hours=sum(json.loads((root/f'chip{r}/COMPLETE.json').read_text())['elapsed_seconds'] for r in range(4))/3600,
            visible_chips=[0,1,2,3],actual_billing_usd=None,reserved_final_used=False,decision=result['decision']))
        print('COMPLETE',json.dumps(result['contrasts']),flush=True)
    except BaseException as exc:
        atomic_json(root/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),elapsed_seconds=time.time()-begin));raise
    finally:
        for child,log in processes:
            if child.poll() is None:child.terminate()
            log.close()
if __name__=='__main__':main()
