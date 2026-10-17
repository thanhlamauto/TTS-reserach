"""Stage1 paired local-noise screening on the unchanged SD1.5 XLA sampler."""
import argparse,fcntl,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from otlab.adapter import frozen_xla_factory,schedule_identity
from otlab.export import historical_tape
from otlab.noise import NoiseTape,new_tape
from otlab.io import atomic_json,file_sha,tensor_sha,load_tensor,digest
from ot_next.identity import compatible_scheduler_metadata
from ot_next.features import RGBClient
from .model import probe_actions,fit,calibrate,evaluate

class CompactRGB(RGBClient):
    def encode(self,rgb,label):
        value=super().encode(rgb,label)
        for kind in ['input','output']:(self.directory/f'{self.calls:03d}_{label}.{kind}.pt').unlink()
        return value


def make_basis(sigmas,seed,pid):
    rng=np.random.default_rng(np.random.SeedSequence([seed,pid,6361]))
    raw=rng.normal(size=(5,4*64*64,4))
    s=np.asarray(sigmas,dtype=np.float64)
    gram=np.einsum('j,jna,jnb->ab',s*s,raw,raw)/(4*64*64)
    eigen,v=np.linalg.eigh(gram)
    if eigen.min()<=1e-8*eigen.max():raise RuntimeError('degenerate injected-energy basis; do not amplify')
    basis=raw@((v/np.sqrt(eigen))@v.T)
    basis=basis.astype(np.float32)
    actual=np.einsum('j,jna,jnb->ab',s*s,basis.astype(float),basis.astype(float))/(4*64*64)
    if np.max(np.abs(actual-np.eye(4)))>1e-6:raise RuntimeError('basis whitening failed')
    return torch.from_numpy(basis.reshape(5,4,64,64,4)),dict(weighted_gram=actual.tolist(),raw_eigenvalues=eigen.tolist(),condition=float(eigen.max()/eigen.min()))


def unit(adapter,encoder,cfg,u,case,status):
    from sd15_sinkhorn_tpu.common import initial_noise
    from sd15_sinkhorn_tpu.reward import CLIPCPU
    case.mkdir(parents=True,exist_ok=False);before=dict(adapter.counters)
    seed,pid=u['prefix_seed'],u['prompt_id'];adapter.set_prompt(u['prompt'])
    initial,noise_seed=initial_noise(seed,pid,0,4)
    history=historical_tape(adapter.sampler,noise_seed,4,start=0)
    status('prefix_and_original_replay',prompt_id=pid)
    checkpoint=adapter.advance(initial,0,40,tuple(range(4)),history)
    original,cp,_=adapter.sampler.run(adapter.prompt,initial,start=0,checkpoints=(40,),noise_seed=noise_seed)
    adapter.counters['original_full_replay_sample_steps']+=400
    adapter.counters['unet_cfg_branch_sample_evaluations']+=800
    resumed=adapter.advance(checkpoint,40,100,tuple(range(4)),history)
    if not torch.equal(cp[40],checkpoint) or not torch.equal(resumed,original):raise RuntimeError('original full100 checkpoint replay failed')
    atomic_json(case/'REPLAY.json',dict(passed=True,cases=4,prefix_bitwise=True,endpoint_bitwise=True,extra_sampling_nfe=760))
    z,labels=probe_actions(cfg['radius'],pid,seed)
    sigmas=[float(adapter.sampler.coefficients[i][4]) for i in cfg['noise_indices']]
    basis,basis_info=make_basis(sigmas,seed,pid)
    # The five action-block noises are identical across A/B/C. Only45..99 vary.
    tapes=[new_tape(seed,pid,100+k,(0,),(4,64,64),adapter.schedule_sha,start=40) for k in range(3)]
    noise=torch.cat([t.variance_noise.repeat(1,9,1,1,1) for t in tapes],dim=1)
    change=torch.einsum('jchwk,nk->jnchw',basis,torch.tensor(z,dtype=torch.float32))
    common=history.variance_noise[40:45,:1]
    noise[:5]=common+change
    measured=noise[:5]-common
    energy=((measured.double()*torch.tensor(sigmas,dtype=torch.float64)[:,None,None,None,None])**2).mean((2,3,4)).sum(0).sqrt()
    norm=torch.tensor(np.linalg.norm(z,axis=1))
    if not torch.allclose(energy,norm,atol=2e-7,rtol=1e-5):raise RuntimeError('actual injected control does not match action norm')
    tape=NoiseTape(noise,40,tuple(range(27)),adapter.schedule_sha)
    tape_hash=tensor_sha(noise);status('paired27_suffixes',prompt_id=pid)
    state45=adapter.advance(checkpoint[:1].repeat(27,1,1,1),40,45,tuple(range(27)),tape)
    if not torch.equal(state45[:9],state45[9:18]) or not torch.equal(state45[0],state45[18]):raise RuntimeError('fixed action block diverged across tapes')
    final=adapter.advance(state45,45,100,tuple(range(27)),tape)
    if not torch.isfinite(final).all():raise RuntimeError('nonfinite endpoints')
    rewards,rgb=adapter.score(final,'terminal')
    dino=encoder.encode(rgb,f'p{pid}_terminal27')
    if not hasattr(adapter,'noise_clip'):adapter.noise_clip=CLIPCPU(adapter.sampler.cfg,cfg['dcs_root'])
    clip=adapter.noise_clip.score(adapter.prompt,[Image.fromarray(x.permute(1,2,0).numpy()) for x in rgb])
    clip_features=torch.tensor(clip['clip_embeddings'])
    state_rms=[];terminal_rms=[];image_mae=[]
    for i in range(27):
        base=(i//9)*9
        state_rms.append(float((state45[i].float()-state45[base].float()).square().mean().sqrt()))
        terminal_rms.append(float((final[i].float()-final[base].float()).square().mean().sqrt()))
        image_mae.append(float((rgb[i].float()-rgb[base].float()).abs().mean()))
    if any(state_rms[i]==0 for i in range(27) if i%9):raise RuntimeError('intervention vanished in BF16; no radius retuning on outcomes')
    bundle=dict(initial=initial,checkpoint40=checkpoint,states45=state45,final=final,rgb=rgb,rewards=rewards,dino=dino,clip=clip_features,
                actions=torch.tensor(z),basis=basis,sigmas=sigmas,noise_seed=noise_seed,noise_tape_sha=tape_hash,tail_replicates=[100,101,102])
    torch.save(bundle,case/'endpoints.pt')
    row=dict(prompt_id=pid,prompt=u['prompt'],split=u['split'],prefix_seed=seed,actions=z.tolist(),tapes=labels,rewards=rewards.tolist(),
             model=fit(rewards.numpy(),cfg['radius']),injected_energy=energy.tolist(),basis=basis_info,
             state45_rms_from_tape_noop=state_rms,terminal_rms_from_tape_noop=terminal_rms,image_mae_from_tape_noop=image_mae,
             exact_shared_action_block=True,endpoint_file_sha256=file_sha(case/'endpoints.pt'),replay_cases=4,
             counters={k:v-before.get(k,0) for k,v in adapter.counters.items()},diagnostic_nfe=1660,extra_replay_nfe=760,
             actual_sampling_nfe=2420,padding_sample_steps=60,aux_unet_nfe=0,vae_nominal_samples=27,vae_padded_samples=28)
    atomic_json(case/'COMPLETE.json',row);status('anchor_complete',prompt_id=pid)
    return row


def worker(cfg,rank,phase,source):
    root=Path(cfg['root']);output=root/phase/f'chip{rank}';output.mkdir(parents=True,exist_ok=True)
    u=[x for x in cfg['units'] if x['split']==phase][rank]
    if (output/'case/COMPLETE.json').exists():return
    begin=time.time();adapter=encoder=None
    def status(stage,**more):
        r=dict(stage=stage,phase=phase,rank=rank,pid=os.getpid(),elapsed_seconds=time.time()-begin,**more)
        atomic_json(output/'status.json',r);print(json.dumps(r),flush=True)
    try:
        status('load_frozen_backend');old=Path(cfg['old_root'])/'attempts/8074f94c847d'
        ac=json.loads((old/'adapter_config.json').read_text());runtime=output/'runtime';runtime.mkdir(exist_ok=True)
        cache=Path('/dev/shm')/('noise_avoid_'+file_sha(source/'protocol.json')[:12])/f'chip{rank}';cache.mkdir(parents=True,exist_ok=True)
        if not (runtime/'xla_cache').exists():(runtime/'xla_cache').symlink_to(cache,target_is_directory=True)
        ac['runtime_output']=str(runtime);adapter=frozen_xla_factory(ac)
        previous=json.loads((old/'neural_runtime/environment/sampler0.json').read_text())['scheduler_config']
        if not compatible_scheduler_metadata(previous,dict(adapter.sampler.scheduler.config)):raise RuntimeError('scheduler changed')
        adapter.sampler.scheduler.register_to_config(_use_default_values=previous['_use_default_values'])
        expected=json.loads((old/'replay.json').read_text())['records'][0]['adapter_identity']['schedule_sha']
        if schedule_identity(adapter.sampler)!=expected:raise RuntimeError('original full schedule hash mismatch')
        adapter.schedule_sha=expected;adapter.base_identity['schedule_sha']=expected;adapter.set_prompt(u['prompt'])
        repo=source.parents[1]
        encoder=CompactRGB(str(Path(cfg['old_root'])/'worker_venv/bin/python'),repo/'ot_tts_lab_v2_reimpl/configs/worker_dinov2.yaml',output/'rgb_worker')
        atomic_json(output/'identity.json',dict(adapter=adapter.identity,encoder=encoder.identity,visible_chip=os.environ['TPU_VISIBLE_CHIPS']))
        row=unit(adapter,encoder,cfg,u,output/'case',status)
        atomic_json(output/'COMPLETE.json',dict(passed=True,prompt_id=u['prompt_id'],elapsed_seconds=time.time()-begin,counters=dict(adapter.counters),dino_images=encoder.count))
    except BaseException as exc:
        atomic_json(output/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),counters={} if adapter is None else dict(adapter.counters)));raise
    finally:
        if encoder:encoder.close()
        if adapter:(output/'xla_metrics.txt').write_text(adapter.sampler.rt.metrics())


def read_rows(root,phase):
    paths=sorted((root/phase).glob('chip*/case/COMPLETE.json'))
    if len(paths)!=3:raise RuntimeError('phase rectangle incomplete')
    return [json.loads(p.read_text()) for p in paths]


def export_blind(root,cfg):
    # Screen only. Hide action, tape, reward, model and scouting status.
    out=root/'blinded_review';out.mkdir(exist_ok=True);mapping=[]
    for p in sorted((root/'screen').glob('chip*/case/COMPLETE.json')):
        row=json.loads(p.read_text());data=load_tensor(p.with_name('endpoints.pt'));pid=row['prompt_id']
        order=np.random.default_rng(np.random.SeedSequence([2026101712,pid])).permutation(27)
        sheet=Image.new('RGB',(5*192,6*214),'white');draw=ImageDraw.Draw(sheet)
        for j,i in enumerate(order):
            key=f'N{pid}{j:02d}';im=Image.fromarray(data['rgb'][i].permute(1,2,0).numpy());im.save(out/(key+'.png'))
            x,y=(j%5)*192,(j//5)*214;sheet.paste(im.resize((192,192)),(x,y));draw.text((x+4,y+195),key,fill='black')
            mapping.append(dict(image_id=key,prompt_id=pid,slot=int(i),rgb_sha=tensor_sha(data['rgb'][i]),case=str(p.parent.relative_to(root))))
        sheet.save(out/f'p{pid:03d}.png')
    atomic_json(root/'BLINDED_MAP.json',mapping)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--worker',type=int);parser.add_argument('--phase');args=parser.parse_args()
    source=Path(args.protocol).resolve().parent;cfg=json.loads(Path(args.protocol).read_text())
    if args.worker is not None:return worker(cfg,args.worker,args.phase,source)
    root=Path(cfg['root']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'COMPLETE.json').exists():print('ALREADY_COMPLETE',root);return
    if (root/'failure.json').exists():raise RuntimeError('prior failure must be classified and repaired, not silently duplicated')
    if shutil.disk_usage(root).free<1024**3:raise RuntimeError('less than1GiB free')
    owners=subprocess.run(['fuser','/dev/vfio/0','/dev/vfio/1','/dev/vfio/2','/dev/vfio/3'],capture_output=True,text=True)
    if owners.stdout.strip():raise RuntimeError('TPU devices already occupied')
    atomic_json(root/'preregistered.json',cfg)
    atomic_json(root/'SOURCE_HASHES.json',{str(p.relative_to(source)):file_sha(p) for p in source.rglob('*') if p.is_file()})
    processes=[];begin=time.time()
    def interrupted(sig,frame):raise TimeoutError('persistent wall deadline or cancellation')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for phase in ['calibration','screen']:
            atomic_json(root/'status.json',dict(stage=phase,elapsed_seconds=time.time()-begin))
            processes=[]
            for rank in range(3):
                # Expand only after the first complete calibration anchor passes replay and energy gates.
                if phase=='calibration' and rank==1:
                    while not (root/'calibration/chip0/COMPLETE.json').exists():
                        if processes[0][0].poll() is not None:raise RuntimeError('single-chip smoke failed')
                        time.sleep(1)
                log=(root/f'{phase}_chip{rank}.log').open('a')
                p=subprocess.Popen([sys.executable,'-B','-u','-m','noise_avoid.campaign','--protocol',str(source/'protocol.json'),'--worker',str(rank),'--phase',phase],
                    env=dict(os.environ,TPU_VISIBLE_CHIPS=str(rank)),stdout=log,stderr=subprocess.STDOUT)
                processes.append((p,log))
                atomic_json(root/'worker_pids.json',[p.pid for p,l in processes])
            exits=[p.wait() for p,l in processes]
            for p,l in processes:l.close()
            if any(exits):raise RuntimeError('worker failure; retain all independently completed anchors')
            if phase=='calibration':
                frozen=calibrate(read_rows(root,phase),cfg);atomic_json(root/'FROZEN_ERROR_ALLOWANCE.json',frozen)
                print('CALIBRATION_FROZEN',json.dumps(frozen),flush=True)
        calibration=json.loads((root/'FROZEN_ERROR_ALLOWANCE.json').read_text())
        result=evaluate(read_rows(root,'screen'),calibration,cfg)
        atomic_json(root/'RESULTS.json',result);export_blind(root,cfg)
        receipts=[json.loads(p.read_text()) for p in sorted(root.glob('*/chip*/COMPLETE.json'))]
        totals={key:sum(r['counters'].get(key,0) for r in receipts) for key in set().union(*(r['counters'] for r in receipts))}
        if totals.get('denoise_sample_steps')!=12120 or totals.get('original_full_replay_sample_steps')!=2400 or totals.get('denoise_padding_sample_steps')!=360:
            raise RuntimeError('NFE accounting differs from frozen allocation')
        complete=dict(passed=True,anchors=6,calibration_anchors=3,screen_anchors=3,terminal_rows=162,replay_cases=24,replay_bitwise=True,
            sampling_nfe=14520,diagnostic_nfe=9960,replay_extra_nfe=4560,padding_sample_steps=360,physical_sample_steps=14880,aux_unet_nfe=0,
            cfg_branch_sample_evaluations=totals['unet_cfg_branch_sample_evaluations'],counters=totals,worker_receipts=receipts,
            elapsed_seconds=time.time()-begin,occupied_worker_chip_hours=sum(r['elapsed_seconds'] for r in receipts)/3600,
            physical_visible_chips=[0,1,2],chip3_unused=True,owner_attested_no_charge=True,actual_billing_usd=None,
            stage1_gate_passed=result['stage1_gate_passed'],decision=result['decision'],final_pool_used=False)
        atomic_json(root/'COMPLETE.json',complete);atomic_json(root/'status.json',dict(stage='COMPLETE',decision=result['decision']))
        print('COMPLETE',json.dumps(complete),flush=True)
    except BaseException as exc:
        atomic_json(root/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),elapsed_seconds=time.time()-begin));raise
    finally:
        for p,l in processes:
            if p.poll() is None:p.terminate()
            if not l.closed:l.close()

if __name__=='__main__':main()
