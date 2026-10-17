"""Finish five fixed old variants, CPU edit first, then original TPU suffix."""
import argparse,contextlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch,yaml
from otlab.io import atomic_json,load_tensor,file_sha,tensor_sha,validate_checkpoint


def cpu(cfg,source,root):
    from otlab.features import TensorFeatures
    from otlab.proximal import proximal_edit
    from .p2 import repaired_solve
    import otlab.ot
    gate=json.loads(Path(cfg['repair_gate']).read_text())
    if not gate['passed'] or not gate['old_failure_reproduced']:raise RuntimeError('P2 repair gate missing')
    cp=validate_checkpoint(load_tensor(cfg['checkpoint']))
    with contextlib.redirect_stdout(sys.stderr):
        encoder=TensorFeatures(yaml.safe_load((source.parents[1]/'ot_tts_lab_v2_reimpl/configs/worker_dinov2.yaml').read_text()))
    exact=load_tensor(cfg['exact_instance'])
    if encoder.identity!=exact['encoder_identity']:raise RuntimeError('repaired encoder identity changed')
    cache={};hits=0;solves=[]
    for key,features in [('source_clean',exact['source_features']),('bank_clean',exact['target_features'])]:
        for i in range(len(cp[key])):cache[tensor_sha(cp[key][i:i+1])]=features[i:i+1].clone()
    def encode(x):
        nonlocal hits
        key=tensor_sha(x)
        if not x.requires_grad and key in cache:hits+=len(x);return cache[key].clone()
        f=encoder.encode_clean(x)
        if not x.requires_grad:cache[key]=f.detach().clone()
        return f
    def solve(*args,**kwargs):
        plan,val,info=repaired_solve(*args,**kwargs);solves.append(info);return plan,val,info
    otlab.ot.solve=solve
    rows=[]
    for item in cfg['variants']:
        name=item['name'];start=time.time();before=dict(encoder.counters);previous=hits;solve_start=len(solves)
        atomic_json(root/'cpu_status.json',dict(stage='proximal_edit',method=name,started_unix=start))
        try:
            kwargs={k:v for k,v in item.items() if k!='name'}
            out=proximal_edit(cp,encode,**kwargs,bound=.025,factor=4,steps=2)
            if out['entry_states'].dtype!=torch.bfloat16:raise RuntimeError('entry must preserve BF16')
            if max(out['trust']['actual_rms'])>.025 or not any(out['trust']['active']):raise RuntimeError('inactive or invalid edit')
            out['solver_audit']=solves[solve_start:]
            out['encoder_counters']={k:v-before.get(k,0) for k,v in encoder.counters.items()}
            out['cached_feature_hits']=hits-previous
            out['elapsed_seconds']=time.time()-start
            torch.save(out,root/(name+'_edit.pt'))
            row=dict(method=name,passed=True,elapsed_seconds=out['elapsed_seconds'],losses=out['losses'],
                trust=out['trust'],cached_feature_hits=out['cached_feature_hits'],encoder_counters=out['encoder_counters'],
                edit_sha=file_sha(root/(name+'_edit.pt')))
        except Exception as exc:
            row=dict(method=name,passed=False,error=str(exc),traceback=traceback.format_exc(),elapsed_seconds=time.time()-start)
        rows.append(row);atomic_json(root/'CPU_ROWS.json',rows)
    atomic_json(root/'CPU_COMPLETE.json',dict(rows=rows,encoder_identity=encoder.identity,
        counters=dict(encoder.counters),cached_feature_hits=hits,all_passed=all(r['passed'] for r in rows)))


def terminal(cfg,source,root):
    from otlab.adapter import frozen_xla_factory,schedule_identity
    from otlab.noise import NoiseTape
    from ot_next.identity import compatible_scheduler_metadata
    from ot_next.features import RGBClient
    from ot_next.e0 import mean_diversity,contact_sheet
    from PIL import Image
    old=Path(cfg['old_attempt']);cp=load_tensor(cfg['checkpoint']);tape=NoiseTape.from_bundle(load_tensor(cfg['noise']))
    ac=json.loads((old/'adapter_config.json').read_text());runtime=root/'runtime';runtime.mkdir()
    cache=Path('/dev/shm')/('ot_finish_'+str(os.getpid()));cache.mkdir()
    (runtime/'xla_cache').symlink_to(cache,target_is_directory=True);ac['runtime_output']=str(runtime)
    adapter=frozen_xla_factory(ac)
    prior=json.loads((old/'neural_runtime/environment/sampler0.json').read_text())['scheduler_config']
    if not compatible_scheduler_metadata(prior,dict(adapter.sampler.scheduler.config)):raise RuntimeError('scheduler numeric change')
    adapter.sampler.scheduler.register_to_config(_use_default_values=prior['_use_default_values'])
    expected=cp['meta']['schedule_sha']
    if schedule_identity(adapter.sampler)!=expected:raise RuntimeError('schedule hash differs')
    adapter.schedule_sha=expected;adapter.base_identity['schedule_sha']=expected;adapter.set_prompt(cp['meta']['prompt'])
    rows=[json.loads(x) for x in (old/'results/rows.jsonl').read_text().splitlines()];old_rows={r['method']:r for r in rows}
    baseline=load_tensor(old/'results'/old_rows['none']['job_id']/'terminal.pt')
    replay=adapter.advance(cp['source_xt'],50,100,tuple(range(8)),tape)
    reward,rgb=adapter.score(replay,'replay')
    if not (torch.equal(replay,baseline['latent']) and torch.equal(reward,baseline['rewards']) and torch.equal(rgb,baseline['rgb'])):
        raise RuntimeError('old none latent/pixel/reward replay must all match exactly')
    atomic_json(root/'replay_gate.json',dict(passed=True,latent_bitwise=True,pixels_bitwise=True,reward_bitwise=True,new_nfe=400))
    features=RGBClient(cfg['worker_python'],source.parents[1]/'ot_tts_lab_v2_reimpl/configs/worker_dinov2.yaml',root/'rgb_worker')
    from sd15_sinkhorn_tpu.reward import CLIPCPU
    clip=CLIPCPU(adapter.sampler.cfg,cfg['dcs_root']);results=[]
    def evaluate(name,data,new):
        f=features.encode(data['rgb'],name)
        c=clip.score(adapter.prompt,[Image.fromarray(x.permute(1,2,0).numpy()) for x in data['rgb']])
        cf=torch.tensor(c['clip_embeddings'])
        data=dict(data,dino=f,clip=cf)
        torch.save(data,root/(name+'_terminal.pt'))
        row=dict(method=name,mean_ir=float(data['rewards'].mean()),best_ir=float(data['rewards'].max()),
            paired_delta_none=float((data['rewards']-baseline['rewards']).mean()),
            clip_diversity=mean_diversity(cf),dino_diversity=mean_diversity(f),
            cold_nfe=2400 if name=='none' else old_rows[name]['cold_nfe'] if name in old_rows else 2424,
            new_suffix_nfe=400 if new else 0,endpoint_sha=tensor_sha(data['latent']))
        results.append(row);atomic_json(root/'RESULT_ROWS.json',results)
        contact_sheet(data['rgb'],[f'{name} {i} IR {float(data["rewards"][i]):.2f}' for i in range(8)],root/(name+'.png'))
    try:
        for name in ['none','p1_ot','p1_independent','p1_greedy']:
            data=load_tensor(old/'results'/old_rows[name]['job_id']/'terminal.pt');evaluate(name,data,False)
        cpu_rows=json.loads((root/'CPU_ROWS.json').read_text())
        for item in cpu_rows:
            if not item['passed']:continue
            name=item['method'];atomic_json(root/'status.json',dict(stage='original_suffix',method=name))
            edit=load_tensor(root/(name+'_edit.pt'))
            end=adapter.advance(edit['entry_states'],50,100,tuple(range(8)),tape)
            r,im=adapter.score(end,'terminal');evaluate(name,dict(latent=end,rgb=im,rewards=r),True)
        result=dict(passed=len(results)==9,rows=results,original_four_reused=True,old_checkpoint_and_tape=True,
            prompts=1,prefix_seeds=1,engineering_only=True,scientific_winner=None,
            counters=dict(adapter.counters),dino_images=features.count,dino_rpc_seconds=features.seconds,
            cpu_receipt=json.loads((root/'CPU_COMPLETE.json').read_text()),new_nfe=int(adapter.counters['denoise_sample_steps']),
            no_new_aux_unet=True,old_failure_cost_preserved=True,reserved_final_used=False)
        atomic_json(root/'COMPLETE.json',result);print('COMPLETE',json.dumps(result),flush=True)
    finally:
        features.close();(root/'xla_metrics.txt').write_text(adapter.sampler.rt.metrics())


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--cpu',action='store_true');args=p.parse_args()
    source=Path(args.protocol).resolve().parent;cfg=json.loads(Path(args.protocol).read_text());root=Path(cfg['root'])
    root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    if args.cpu:return cpu(cfg,source,root)
    if (root/'COMPLETE.json').exists():return
    start=time.time();atomic_json(root/'preregistered.json',cfg)
    try:
        log=(root/'cpu_worker.log').open('a')
        process=subprocess.Popen([cfg['worker_python'],'-B','-u','-m','ot_remainder.finish_old','--protocol',args.protocol,'--cpu'],stdout=log,stderr=subprocess.STDOUT)
        atomic_json(root/'cpu_pid.json',dict(pid=process.pid));code=process.wait();log.close()
        if code:raise RuntimeError('CPU proposal worker failed; retain completed edits')
        # CPU-only edit phase never owns a TPU, and never displaces another job.
        atomic_json(root/'status.json',dict(stage='wait_for_free_TPU',cpu_complete=True))
        while True:
            owner=subprocess.run(['fuser','/dev/vfio/0','/dev/vfio/1','/dev/vfio/2','/dev/vfio/3'],capture_output=True,text=True)
            if not owner.stdout.strip():break
            time.sleep(5)
        terminal(cfg,source,root)
        result=json.loads((root/'COMPLETE.json').read_text());result['elapsed_seconds']=time.time()-start
        atomic_json(root/'COMPLETE.json',result)
    except BaseException as exc:
        atomic_json(root/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),elapsed_seconds=time.time()-start));raise


if __name__=='__main__':main()
