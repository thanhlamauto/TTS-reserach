"""Bounded E0 then R1 smoke, with durable failures and no implicit expansion."""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import torch

from otlab.adapter import frozen_xla_factory
from otlab.io import atomic_json, file_sha, load_tensor, digest
from .features import RGBClient
from . import e0, r1


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',required=True)
    args=parser.parse_args()
    protocol_path=Path(args.protocol).resolve()
    config=json.loads(protocol_path.read_text())
    root=Path(config['root'])
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'campaign.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source_files=sorted(protocol_path.parent.glob('ot_next/*.py'))+[protocol_path,protocol_path.parent/'old_input_manifest.json']
    source_id=digest({str(p.relative_to(protocol_path.parent)):file_sha(p) for p in source_files})
    run=root/'attempts'/source_id[:12]
    if (run/'COMPLETE.json').exists():
        print('ALREADY_COMPLETE',run,flush=True)
        return
    run.mkdir(parents=True,exist_ok=False)
    begin=time.time()
    adapter,features=None,None
    def status(stage,**extra):
        receipt=dict(stage=stage,pid=os.getpid(),source_sha=source_id,run=str(run),
                     updated_unix=time.time(),elapsed_seconds=time.time()-begin)
        receipt.update(extra)
        atomic_json(run/'status.json',receipt)
        atomic_json(root/'status.json',receipt)
        print('STAGE',stage,json.dumps(extra),flush=True)
    def interrupted(signum,frame):
        raise TimeoutError(f'bounded campaign interrupted by signal{signum}')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        status('preflight')
        versions={k:importlib.metadata.version(k) for k in ['torch','torch_xla','transformers','diffusers','numpy']}
        if versions!={'torch':'2.4.0+cpu','torch_xla':'2.4.0','transformers':'4.38.2','diffusers':'0.31.0.dev0','numpy':'1.26.3'}:
            raise RuntimeError(f'frozen sampler environment changed:{versions}')
        owners=subprocess.run(['fuser','/dev/vfio/0','/dev/vfio/1','/dev/vfio/2','/dev/vfio/3'],capture_output=True,text=True)
        if owners.stdout.strip():raise RuntimeError('TPU owned; no conflicting or duplicate launch')
        if shutil.disk_usage(root).free < 2*1024**3:raise RuntimeError('less than2GiB disk headroom; do not launch')
        from otlab.runner import code_identity
        if code_identity()!='8074f94c847d20dab1b6b6604ba7f3730e3e2eab027dc950ac40b55b0741fa98':
            raise RuntimeError('actual prior otlab reconstruction source differs')
        manifest=json.loads((protocol_path.parent/'old_input_manifest.json').read_text())
        verified={p:file_sha(p)==sha for p,sha in manifest.items()}
        atomic_json(run/'INPUT_INTEGRITY.json',dict(passed=all(verified.values()),files=verified))
        if not all(verified.values()):raise RuntimeError('frozen E0 input mismatch')
        atomic_json(run/'preregistered.json',dict(config=config,protocol_sha=file_sha(protocol_path),source_sha=source_id,
                    versions=versions,actual_billing_usd=None,owner_attested_no_charge=True,
                    scientific_winner=None,heldout_used=False))
        old=Path(config['old_root'])
        old_attempt=old/'attempts/8074f94c847d'
        adapter_config=json.loads((old_attempt/'adapter_config.json').read_text())
        runtime=run/'neural_runtime'
        runtime.mkdir()
        cache=Path('/dev/shm')/('ot_partial_refresh_'+source_id[:12])
        cache.mkdir(exist_ok=True)
        (runtime/'xla_cache').symlink_to(cache,target_is_directory=True)
        adapter_config['runtime_output']=str(runtime)
        status('load_existing_backend')
        adapter=frozen_xla_factory(adapter_config)
        # Diffusers builds this one ignored metadata list from a Python set.
        # Its order varies across interpreters, although every numeric field and
        # schedule coefficient is identical. Reconstruct the old FULL hash first;
        # do not accept a hash alias unless it authenticates the current arrays.
        from .identity import compatible_scheduler_metadata
        from otlab.adapter import schedule_identity
        prior_environment=json.loads((old_attempt/'neural_runtime/environment/sampler0.json').read_text())
        previous=prior_environment['scheduler_config']
        current=dict(adapter.sampler.scheduler.config)
        if not compatible_scheduler_metadata(previous,current):
            raise RuntimeError('numeric or semantic scheduler config changed')
        old_schedule=json.loads((old_attempt/'replay.json').read_text())['records'][0]['adapter_identity']['schedule_sha']
        new_hash=adapter.schedule_sha
        adapter.sampler.scheduler.register_to_config(_use_default_values=previous['_use_default_values'])
        authenticated=schedule_identity(adapter.sampler)
        if authenticated!=old_schedule:
            raise RuntimeError('current timesteps/alphas/config do not reproduce original full hash')
        adapter.schedule_sha=authenticated
        adapter.base_identity['schedule_sha']=authenticated
        adapter.set_prompt(adapter.prompt)
        atomic_json(run/'SCHEDULER_METADATA_REPAIR.json',dict(
            original_process_hash=new_hash,authenticated_old_hash=authenticated,
            all_numeric_parameters_unchanged=True,only_default_key_list_order_normalized=True,
            full_hash_authenticates_current_timesteps_alphas_final_alpha_eta_config=True,
            old_order=previous['_use_default_values'],new_process_order=current['_use_default_values']))
        atomic_json(run/'adapter_identity.json',adapter.identity)
        worker_python=old/'worker_venv/bin/python'
        worker_config=protocol_path.parents[2]/'ot_tts_lab_v2_reimpl/configs/worker_dinov2.yaml'
        prior_config=Path('/home/thanhlamtba31/.orx/runs/44789850-fdb0-4ae6-819f-d8e71d3268c1/repo/ot_tts_lab_v2_reimpl/configs/worker_dinov2.yaml')
        if file_sha(worker_config)!=file_sha(prior_config):raise RuntimeError('worker configuration changed')
        features=RGBClient(str(worker_python),worker_config,run/'rgb_worker')
        atomic_json(run/'feature_identity.json',features.identity)
        e0_result=e0.execute(adapter,features,config,run/'E0',status)
        bank_features=load_tensor(run/'E0/features.pt')['bank_features'].double().numpy()
        distance=np.maximum(0,1-bank_features@bank_features.T)
        positive=distance[~np.eye(16,dtype=bool)]
        positive=positive[positive>1e-12]
        merge=max(float(np.quantile(positive,.1)) if len(positive) else 1e-6,1e-6)
        atomic_json(run/'R1_CALIBRATION.json',dict(merge_distance=merge,rule=config['merge_calibration'],
                    calibration_data='old16 bank images only',new_r1_outcomes_seen=False,
                    quality_threshold=None,semantic_checker=None,smoke_only=True))
        r1_result=r1.execute(adapter,features,config,run/'R1_smoke',merge,status)
        for p,sha in manifest.items():
            if file_sha(p)!=sha:raise RuntimeError('original evidence mutated during campaign')
        finish=dict(passed=True,prompts_E0=1,prompts_R1=1,R1_rows=4,new_missing_edits=10,
                    elapsed_seconds=time.time()-begin,all_counters=dict(adapter.counters),
                    encoder_samples=features.count,encoder_rpc_seconds=features.seconds,
                    actual_billing_usd=None,active_visible_chips=1,allocated_physical_chips=4,
                    scientific_winner=None,final_evaluation_opened=False,E1_opened=False,
                    decision='E0_COMPLETE_R1_ENGINEERING_PASS_PENDING_PRIMARY_METRIC_CALIBRATION')
        atomic_json(run/'COMPLETE.json',finish)
        status('complete',**finish)
    except BaseException as exc:
        atomic_json(run/'failure.json',dict(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),
                    elapsed_seconds=time.time()-begin,measured_counters=dict(adapter.counters) if adapter else {},
                    encoder_samples=features.count if features else 0))
        status('failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if features is not None:features.close()
        if adapter is not None:(run/'xla_metrics.txt').write_text(adapter.sampler.rt.metrics())


if __name__=='__main__':main()
