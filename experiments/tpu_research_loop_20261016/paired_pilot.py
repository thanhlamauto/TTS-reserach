"""Bounded same-row diagnostic around the existing frozen live-XLA BFS engine."""
from pathlib import Path
import argparse, gc, hashlib, itertools, json, os, subprocess, sys, time, traceback

HERE = Path(__file__).resolve().parent
LEGACY = Path('/home/thanhlamtba31/TTS-reserach')
DCS = Path('/home/thanhlamtba31/static-bfs-dcs')


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(p)


def verify(root):
    assert all(sha(p)==h for p,h in read(HERE/'legacy_manifest.json').items()), 'legacy code/config changed'
    m=read(root/'source_manifest.json')
    assert all(sha(root/n)==h for n,h in m['frozen_files'].items()), 'frozen round config changed'
    assert sha(HERE/'paired_pilot.py')==m['pilot_source_sha256'], 'pilot code changed'
    assert read(root.parent.parent/'reward_tpu_gate.json')['passed']


def policies(plan):
    out=[]
    for c in plan['candidates']:
        s=c['semantics']
        if 'MANUAL8' in c['roles']:
            p=dict(kind='manual',id=c['source_policy_ids'][0],particles=8,resampling_steps=s['steps'],
                   base_temperature=10.,tempering='increase',gamma=.024,selection_mode='manual')
        else:
            p=dict(kind='static',id=c['source_policy_ids'][0],particles=8,resampling_steps=s['steps'],
                   temperature_by_step=list(map(list,zip(s['steps'],s['score_multipliers']))),selection_mode='raw_tau')
        out.append(p)
    assert len({p['id'] for p in out})==6
    return out


def load_engine(root, chip, stage):
    import torch, torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    sys.path.insert(0,str(LEGACY))
    from scripts.static_bfs_graph import adaptive_refinement_tpu as engine
    from scripts.diffusion_classical_search_t2i import tpu_runner as baseline
    from scripts.static_bfs_graph.runner import Runtime
    engine.verify=verify

    def loader(official, cache):
        sys.path.insert(0,str(official/'text_to_image/fkd_diffusers'))
        from image_reward_utils import rm_load
        model=rm_load('ImageReward-v1.0',device=xm.xla_device(),download_root=str(cache/'ImageReward'))
        model.eval();model.requires_grad_(False)
        assert all(p.device.type=='xla' and p.dtype==torch.float32 for p in model.parameters())

        class TPUReward:
            def score_batched(self, prompts, images):
                # Materialize all pending denoiser operations at their old precision.
                xm.mark_step();xm.wait_device_ops()
                torch_xla._XLAC._xla_set_use_full_mat_mul_precision(True)
                try:
                    with torch.no_grad(): value=model.score_batched(prompts,images)
                    xm.mark_step();xm.wait_device_ops()
                    return value
                finally:
                    torch_xla._XLAC._xla_set_use_full_mat_mul_precision(False)

        def hook(module,args,kwargs):
            def tensors(v):
                if torch.is_tensor(v): return [v]
                if isinstance(v,(list,tuple)): return [t for x in v for t in tensors(x)]
                if isinstance(v,dict): return [t for x in v.values() for t in tensors(x)]
                return []
            ts=tensors(args)+tensors(kwargs)
            assert ts and all(t.device.type=='xla' for t in ts), 'neural CPU input'
        for module in [model.blip.visual_encoder,model.blip.text_encoder,model.mlp]:
            module.register_forward_pre_hook(hook,with_kwargs=True)
        result=TPUReward();baseline._ACTIVE_REWARD=result
        return result

    baseline._load_reward=loader
    torch_xla._XLAC._xla_set_use_full_mat_mul_precision(False)
    eng=engine.Engine(root,chip,stage)
    return eng,engine


def worker(root, chip, stage, budget_root):
    from budget_guard import authorize,history,arm_child_deadline
    authorize(budget_root)
    leases=history(budget_root)
    active=[r for r in leases if r['event']=='RESERVE' and r['phase']=='exploration'
            and r['deadline_unix']>time.time() and not any(x['event']=='COMPLETE' and x['job_id']==r['job_id'] for x in leases)]
    assert len(active)==1
    arm_child_deadline(active[0]);verify(root)
    import importlib.metadata
    expected={'torch':'2.4.0+cpu','torch_xla':'2.4.0','transformers':'4.38.2','diffusers':'0.31.0.dev0'}
    assert {k:importlib.metadata.version(k) for k in expected}==expected
    eng,backend=load_engine(root,chip,stage)
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    plan=read(root/'preregistered.json');ps=policies(plan)
    try:
        if stage=='gates':
            eng.gates()
            manual=next(p for p in ps if p['kind']=='manual')
            explicit=dict(manual,kind='static',selection_mode='raw_tau',
                          temperature_by_step=[[i,10*((1+.024)**i-1)] for i in manual['resampling_steps']])
            a,_=eng.direct(plan['prompts'][0],plan['seeds'][0],manual)
            b,_=eng.direct(plan['prompts'][0],plan['seeds'][0],explicit)
            assert all(a[k]==b[k] for k in ['reward','scores','RGB_hashes','latent_hash','initial_latents_sha256'])
            assert [e['parent_indices'] for e in a['events']]==[e['parent_indices'] for e in b['events']]
            write(root/'correctness.json',dict(passed=True,legacy_gates='gates.json',manual_explicit_exact=True,
                  exact_prefix_shared_tree=True,verifier_device='TPU',verifier_matmul='HIGHEST scoped to completed verifier',
                  denoiser_precision='existing BF16 and DEFAULT matmul unchanged',confirmation_opened=False))
        else:
            assert read(root/'correctness.json')['passed']
            edges=plan['edges']['edges']
            jobs=[backend.job(e['src'],e['dst'],e['multiplier']) for e in edges.values()]
            units=list(itertools.product(plan['prompts'],plan['seeds']))
            for pid,seed in units[chip::4]:
                ref,_=eng.reference('A',pid,seed)
                eng.profile('A',pid,seed,jobs)
                for p in ps:
                    target=root/'full_policies'/f'p{pid}-s{seed}-{p["id"]}.json'
                    assert not target.exists(), 'new attempt does not overwrite prior rows'
                    before=dict(eng.account);started=time.perf_counter()
                    row,_=eng.direct(pid,seed,p)
                    xm.mark_step();xm.wait_device_ops()
                    assert [e['sampling_index'] for e in row['events']]==p['resampling_steps']
                    assert len(row['events'])==3 and len(row['scores'])==8 and row['reward']==max(row['scores'])
                    assert eng.account['actual_denoising_NFE']-before['actual_denoising_NFE']==800
                    assert row['initial_latents_sha256']==ref['initial_latents_sha256']
                    for e in row['events']:
                        beta=10*((1+.024)**e['sampling_index']-1) if p['kind']=='manual' else dict(p['temperature_by_step'])[e['sampling_index']]
                        assert abs(e['temperature']-beta)<1e-9
                    row.update(prompt=pid,seed=seed,policy=p,NFE=800,runtime_seconds=time.perf_counter()-started,
                               counters={k:eng.account[k]-before[k] for k in before})
                    write(target,row);eng.persist()
                print(json.dumps(dict(stage='paired_unit_complete',chip=chip,prompt=pid,seed=seed,policies=6,edges=16)),flush=True)
        xm.mark_step();xm.wait_device_ops()
        unsupported=[n for n in met.counter_names() if n.startswith('aten::') and n!='aten::_local_scalar_dense' and met.counter_value(n)]
        assert not unsupported, 'unexpected XLA host fallback: '+str(unsupported)
        eng.persist();write(root/'workers'/stage/f'chip{chip}.done.json',dict(success=True,account=eng.account,host_fallback_counters=unsupported))
        (root/f'{stage}_chip{chip}_xla_metrics.txt').write_text(met.metrics_report())
        # NumPy SSP allocation and CPU statistical bookkeeping are allowed; no neural CPU forward occurs.
    except BaseException as e:
        eng.persist();write(root/'failures'/f'{stage}-chip{chip}.json',dict(error=repr(e),traceback=traceback.format_exc()));raise


def freeze(root, budget_root):
    plan=read(HERE/'paired_preregistered.json');root.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(LEGACY))
    from scripts.static_bfs_graph.policy import StaticBFSPolicy
    from scripts.static_bfs_graph.runner import static_record
    pool=read(LEGACY/'modal_n8_gpu/train_pool.json')
    cfg=dict(protocol=plan['protocol'],seeds={'A':plan['seeds']},snapshots=list(range(5,100,5)))
    split=dict(A=plan['prompts'],B=[i for i in pool['train_indices'] if i not in plan['prompts']],C=[],
               TRAIN_prompts=pool['prompts'],confirmation_opened=False)
    ref=static_record(StaticBFSPolicy(tuple(plan['reference_steps']),tuple((i,8.) for i in plan['reference_steps']),8))
    for n,x in [('preregistered.json',plan),('config.json',cfg),('data_split.json',split),('reference_schedule.json',ref)]: write(root/n,x)
    write(root/'source_manifest.json',dict(frozen_files={n:sha(root/n) for n in ['preregistered.json','config.json','data_split.json','reference_schedule.json']},
          pilot_source_sha256=sha(HERE/'paired_pilot.py'),legacy_source_sha256=read(HERE/'legacy_manifest.json'),
          budget_root=str(budget_root),created_unix=time.time()))
    verify(root)


def child(root,chip,stage,budget_root):
    env=dict(os.environ,TPU_VISIBLE_CHIPS=str(chip),STATIC_BFS_CACHE_ROOT=f'/dev/shm/static-bfs-loop-r1/chip{chip}/cache')
    argv=[sys.executable,str(Path(__file__).resolve()),'--root',str(budget_root),'--worker',str(chip),'--stage',stage,'--run-root',str(root)]
    log=(root/f'{stage}_chip{chip}.log').open('a')
    return subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT),log


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--worker',type=int);p.add_argument('--stage');p.add_argument('--run-root',type=Path);a=p.parse_args()
    if a.worker is not None:return worker(a.run_root,a.worker,a.stage,a.root)
    # No model imports or accelerator access in this supervisor.
    owners=subprocess.run(['fuser','/dev/vfio/0','/dev/vfio/1','/dev/vfio/2','/dev/vfio/3'],capture_output=True,text=True)
    assert not owners.stdout.strip(),'devices owned; do not launch duplicate'
    root=a.root/'round1'/sha(Path(__file__))[:12]
    freeze(root,a.root);write(a.root/'paired_status.json',dict(state='gates',run_root=str(root),started_unix=time.time()))
    gate,log=child(root,0,'gates',a.root)
    try: assert gate.wait()==0,'full-pipeline correctness gates failed; inspect gate log'
    finally: log.close()
    assert read(root/'correctness.json')['passed']
    write(a.root/'paired_status.json',dict(state='paired_panel',run_root=str(root),prompts=6,seeds=4,policies=6,union_edges=16))
    procs=[child(root,k,'panel',a.root) for k in range(4)]
    try:
        while any(proc.poll() is None for proc,_ in procs):
            if any(proc.poll() not in [None,0] for proc,_ in procs):raise RuntimeError('panel worker failed')
            time.sleep(2)
        assert all(proc.returncode==0 for proc,_ in procs)
    finally:
        for proc,log in procs:
            if proc.poll() is None:proc.terminate()
            log.close()
    subprocess.run([sys.executable,str(HERE/'paired_analysis.py'),'--root',str(root)],check=True)
    write(a.root/'paired_status.json',dict(state='complete',run_root=str(root),completed_unix=time.time(),confirmation_opened=False))
    print('PAIRED_DIAGNOSTIC_COMPLETE',root,flush=True)


if __name__=='__main__':main()
