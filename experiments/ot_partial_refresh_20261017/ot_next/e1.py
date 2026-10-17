"""Four-chip, six fixed TRAIN units. No outcome-dependent expansion."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import shutil

import torch
from PIL import Image, ImageDraw

from otlab.adapter import frozen_xla_factory,schedule_identity
from otlab.io import atomic_json,file_sha,load_tensor,digest,tensor_sha
from .features import RGBClient
from .identity import compatible_scheduler_metadata
from .output_selection import quality_diversity_indices
from .r1 import execute


def worker(config,rank,source):
    root=Path(config['root'])
    output=root/f'chip{rank}'
    output.mkdir(exist_ok=False)
    begin=time.time()
    adapter=features=None
    def status(stage,**extra):
        x=dict(stage=stage,rank=rank,pid=os.getpid(),updated_unix=time.time(),elapsed_seconds=time.time()-begin)
        x.update(extra)
        atomic_json(output/'status.json',x)
        print('STAGE',json.dumps(x),flush=True)
    def interrupted(signum,frame):raise TimeoutError(f'worker interrupted {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        status('load_existing_backend')
        old=Path(config['old_root'])/'attempts/8074f94c847d'
        cfg=json.loads((old/'adapter_config.json').read_text())
        runtime=output/'runtime'
        runtime.mkdir()
        cache=Path('/dev/shm')/('ot_R1_E1_'+config['protocol_sha'][:12])/f'chip{rank}'
        cache.mkdir(parents=True,exist_ok=True)
        (runtime/'xla_cache').symlink_to(cache,target_is_directory=True)
        cfg['runtime_output']=str(runtime)
        adapter=frozen_xla_factory(cfg)
        prior=json.loads((old/'neural_runtime/environment/sampler0.json').read_text())['scheduler_config']
        if not compatible_scheduler_metadata(prior,dict(adapter.sampler.scheduler.config)):
            raise RuntimeError('numeric scheduler changed')
        adapter.sampler.scheduler.register_to_config(_use_default_values=prior['_use_default_values'])
        authenticated=schedule_identity(adapter.sampler)
        expected=json.loads((old/'replay.json').read_text())['records'][0]['adapter_identity']['schedule_sha']
        if authenticated!=expected:raise RuntimeError('old full schedule hash does not authenticate current arrays')
        adapter.schedule_sha=authenticated
        adapter.base_identity['schedule_sha']=authenticated
        adapter.set_prompt(adapter.prompt)
        features=RGBClient(str(Path(config['old_root'])/'worker_venv/bin/python'),
                           source.parents[1]/'ot_tts_lab_v2_reimpl/configs/worker_dinov2.yaml',output/'rgb_worker')
        atomic_json(output/'identity.json',dict(adapter=adapter.identity,features=features.identity,rank=rank,
                                               visible_chip=os.environ['TPU_VISIBLE_CHIPS']))
        units=[u for i,u in enumerate(config['units']) if i%4==rank]
        for unit in units:
            local=dict(config,r1_prompt_index=unit['prompt_id'],r1_smoke_seed=unit['seed'])
            local.pop('reuse_r1_prefix',None)
            local.pop('reuse_e0',None)
            adapter.set_prompt(unit['prompt'])
            case=output/f'p{unit["prompt_id"]:03d}'
            execute(adapter,features,local,case,config['merge_distance'],status)
        finish=dict(passed=True,rank=rank,unit_ids=[u['prompt_id'] for u in units],
                    elapsed_seconds=time.time()-begin,counters=dict(adapter.counters),
                    encoder_samples=features.count,encoder_rpc_seconds=features.seconds)
        atomic_json(output/'COMPLETE.json',finish)
        status('complete',**finish)
    except BaseException as exc:
        atomic_json(output/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),
                     counters=dict(adapter.counters) if adapter else {},elapsed_seconds=time.time()-begin))
        status('failed',error=str(exc))
        raise
    finally:
        if features is not None:features.close()
        if adapter is not None:(output/'xla_metrics.txt').write_text(adapter.sampler.rt.metrics())


def aggregate(config,root):
    calibration=config['calibration']
    rows=[]
    review={}
    output_maps=[]
    for unit in config['units']:
        case=list(root.glob(f'chip*/p{unit["prompt_id"]:03d}'))
        if len(case)!=1:raise RuntimeError('missing or duplicate case')
        case=case[0]
        for name in config['methods']:
            result=json.loads((case/(name+'.json')).read_text())
            data=load_tensor(case/(name+'.pt'))
            ir=result['top8_indices']
            qd=quality_diversity_indices(data['rewards'].numpy(),data['dino'].numpy(),calibration['quality_threshold'])
            views=dict(top8_ir=ir,quality_diversity=qd)
            shared=[]
            for view,indices in views.items():
                entries=[]
                for slot in indices:
                    sha=tensor_sha(data['rgb'][slot])
                    # Deduplicate exact pixels across methods and selectors.
                    review.setdefault((unit['prompt_id'],sha),dict(prompt_id=unit['prompt_id'],sha=sha,
                                                                   rgb=data['rgb'][slot]))
                    entries.append(sha)
                shared.append(dict(selector=view,slots=indices,image_sha256=entries))
            output_maps.append(dict(prompt_id=unit['prompt_id'],method=name,views=shared,
                                    case=str(case.relative_to(root)),terminal_file_sha256=file_sha(case/(name+'.pt'))))
            for view,indices in views.items():
                from .e0 import mean_diversity
                rows.append(dict(prompt_id=unit['prompt_id'],seed=unit['seed'],method=name,selector=view,
                                 mean_output_ir=float(data['rewards'][indices].mean()),
                                 best_output_ir=float(data['rewards'][indices].max()),
                                 output_dino_diversity=mean_diversity(data['dino'][indices]),
                                 output_clip_diversity=mean_diversity(data['clip'][indices]),
                                 good_distinct=None,semantic_state='PENDING_BLINDED_REVIEW',
                                 pool_mean_ir=result['mean_ir'],unique_ancestors=result['unique_ancestors'],
                                 cold_sampling_nfe=2400,cold_aux_unet_nfe=24))
    if len(rows)!=48 or len(output_maps)!=24:raise RuntimeError('incomplete six-prompt method/selector rectangle')
    # Outcome-independent ordering: opaque image IDs sorted by fixed SHA key.
    ordered=sorted(review.values(),key=lambda v:digest(dict(key=2026101703,prompt=v['prompt_id'],pixels=v['sha'])))
    ids={}
    for i,item in enumerate(ordered):ids[(item['prompt_id'],item['sha'])]=f'I{i:03d}'
    maps=[]
    for item in ordered:
        maps.append(dict(id=ids[(item['prompt_id'],item['sha'])],prompt_id=item['prompt_id'],sha=item['sha']))
    reviewdir=root/'blinded_review'
    reviewdir.mkdir()
    for unit in config['units']:
        items=[v for v in ordered if v['prompt_id']==unit['prompt_id']]
        # 192px square cells; method/reward/parent IDs never appear on grids.
        cols=6;w=192;h=214
        sheet=Image.new('RGB',(cols*w,((len(items)+cols-1)//cols)*h),'white')
        draw=ImageDraw.Draw(sheet)
        for j,item in enumerate(items):
            x,y=(j%cols)*w,(j//cols)*h
            sheet.paste(Image.fromarray(item['rgb'].permute(1,2,0).numpy()).resize((w,w)),(x,y))
            draw.text((x+5,y+w+3),ids[(item['prompt_id'],item['sha'])],fill='black')
        sheet.save(reviewdir/f'p{unit["prompt_id"]:03d}.png')
        # Full512 views are available for ambiguous cases, no new neural work.
        for item in items:
            Image.fromarray(item['rgb'].permute(1,2,0).numpy()).save(reviewdir/(ids[(item['prompt_id'],item['sha'])]+'.png'))
    atomic_json(root/'BLINDED_IMAGE_INDEX.json',maps)
    atomic_json(root/'OUTPUT_MAPS.json',output_maps)
    atomic_json(root/'PRE_SEMANTIC_ROWS.json',rows)
    return dict(passed=True,paired_prompts=6,methods=4,selectors=2,rows=48,
                unique_review_images=len(ordered),semantic_review_complete=False,
                cached_prediction_optimization_passed=False,final_evaluation_opened=False,
                scientific_winner=None,decision='NEURAL_COMPLETE_PENDING_BLINDED_PRIMARY_EVALUATION')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--protocol',required=True)
    p.add_argument('--worker',type=int)
    args=p.parse_args()
    path=Path(args.protocol).resolve()
    config=json.loads(path.read_text())
    config['protocol_sha']=file_sha(path)
    source=path.parent
    if args.worker is not None:
        return worker(config,args.worker,source)
    root=Path(config['root'])
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'campaign.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'NEURAL_COMPLETE.json').exists():
        print('ALREADY_NEURAL_COMPLETE',flush=True)
        return
    begin=time.time()
    processes=[]
    try:
        owners=subprocess.run(['fuser','/dev/vfio/0','/dev/vfio/1','/dev/vfio/2','/dev/vfio/3'],capture_output=True,text=True)
        if owners.stdout.strip():raise RuntimeError('devices occupied; do not launch duplicate')
        if shutil.disk_usage(root).free<1800*1024**2:raise RuntimeError('insufficient disk headroom for fixed six units')
        for f,sha in json.loads((source/'old_input_manifest.json').read_text()).items():
            if file_sha(f)!=sha:raise RuntimeError('frozen old input changed')
        old_cal=json.loads((source/'calibration.json').read_text())
        if old_cal!=config['calibration']:raise RuntimeError('calibration not frozen in protocol')
        atomic_json(root/'preregistered.json',dict(config=config,protocol_sha=file_sha(path),
                    semantic_rubrics=json.loads((source/'semantic_rubrics.json').read_text()),
                    started_unix=begin,owner_attested_no_charge=True,actual_billing_usd=None,
                    user_authorization='Calibrate on old pilot then run R1; six-prompt development pilot only'))
        atomic_json(root/'status.json',dict(stage='launch_four_workers',started_unix=begin,protocol_sha=file_sha(path)))
        for rank in range(4):
            log=(root/f'chip{rank}.log').open('w')
            env=dict(os.environ,TPU_VISIBLE_CHIPS=str(rank))
            process=subprocess.Popen([sys.executable,'-u','-m','ot_next.e1','--protocol',str(path),'--worker',str(rank)],
                                     env=env,stdout=log,stderr=subprocess.STDOUT)
            processes.append((rank,process,log))
        atomic_json(root/'worker_pids.json',{str(r):p.pid for r,p,l in processes})
        results=[]
        for rank,process,log in processes:
            results.append(dict(rank=rank,exit_code=process.wait()))
            log.close()
        atomic_json(root/'worker_exit_codes.json',results)
        if any(v['exit_code']!=0 for v in results):raise RuntimeError('at least one worker failed; preserve independent completions')
        summary=aggregate(config,root)
        summary.update(elapsed_seconds=time.time()-begin,worker_receipts=[json.loads((root/f'chip{r}/COMPLETE.json').read_text()) for r in range(4)])
        atomic_json(root/'NEURAL_COMPLETE.json',summary)
        atomic_json(root/'status.json',dict(stage='neural_complete_pending_semantic',**summary))
        print('NEURAL_COMPLETE',json.dumps(summary),flush=True)
    except BaseException as exc:
        atomic_json(root/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),elapsed_seconds=time.time()-begin))
        atomic_json(root/'status.json',dict(stage='failed',error=str(exc)))
        raise
    finally:
        for rank,process,log in processes:
            if process.poll() is None:process.terminate()
            if not log.closed:log.close()


if __name__=='__main__':main()
