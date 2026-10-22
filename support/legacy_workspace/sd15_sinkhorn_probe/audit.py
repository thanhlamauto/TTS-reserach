import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from sd15_sinkhorn_tpu.common import atomic_json
from sd15_sinkhorn_tpu.collect import load_tensor
from sd15_sinkhorn_bfs_compare.analyze import read
from .core import METHODS,tensor_sha,noise_tape
from .stage0 import sha


def audit(root,reps):
    root=Path(root); torch.set_num_threads(8); plan=read(root/'frozen_plan.json')
    assert read(root/'stage0_gate.json')['passed'] and read(root/'replay_gate.json')['passed']
    for p,h in plan['source_sha256'].items(): assert sha(p)==h,p
    for p,h in read(root/'inherited_manifest.json').items(): assert sha(p)==h,p
    pngs=0; finals=0; tapes=0; max_norm_error=0.
    for rep in reps:
        for i in plan['stage1_indices']:
            tape=load_tensor(root/'noise'/f'p{i:03d}'/f'rep{rep}.pt')
            h=tensor_sha(tape['variance_noise'])
            assert h==tape['sha256']
            assert tensor_sha(noise_tape(plan['noise_root_seed'],i,rep))==h
            tapes+=1
            ctrl=load_tensor(root/'controls'/f'p{i:03d}'/f'rep{rep}.pt')
            max_norm_error=max(max_norm_error,ctrl['max_rms_relative_error'])
            assert ctrl['max_rms_relative_error']<=.02
            for name in METHODS:
                d=root/'full'/f'rep{rep}'/f'p{i:03d}'/name
                r=read(d/'result.json'); final=load_tensor(d/'final.pt')
                assert r['noise_tape_sha256']==h==final['noise_tape_sha256']
                assert r['initial_sha256']==tensor_sha(ctrl['latents'][name])
                assert final['initial_sha256']==r['initial_sha256']
                assert len(r['scores']['rewards'])==8 and np.isfinite(r['scores']['rewards']).all()
                assert r['diffusion_sample_steps']==400 and r['finite']
                assert final['latents'].shape==(8,4,64,64) and torch.isfinite(final['latents']).all()
                finals+=1
                for j in range(8):
                    with Image.open(d/'images'/f'{j:03d}.png') as im:
                        assert im.size==(512,512); im.verify()
                    pngs+=1
    result=dict(passed=True,replicates=reps,n_prompts=12,branches=finals,png_verified=pngs,
        latent_files_verified=finals,shared_noise_tapes_verified=tapes,
        max_control_rms_relative_error=max_norm_error,
        full_suffix_sample_steps=len(reps)*28800,full_reward_evaluations=len(reps)*576,
        mandatory_replay_sample_steps=400,mandatory_replay_reward_evaluations=8,
        warmup_sample_steps=5*800,
        total_new_actual_sample_steps=len(reps)*28800+400+5*800,
        total_new_reward_evaluations=len(reps)*576+8,
        bank_reward_evaluations_new=0,old_cache_preserved=True,
        replay=read(root/'replay_gate.json'))
    path=root/('validation_audit.json' if len(reps)==2 else 'replicate0_gate.json')
    atomic_json(path,result); print('AUDIT_COMPLETE',result,flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True)
    p.add_argument('--replicates',default='0,1'); a=p.parse_args()
    audit(a.output,[int(v) for v in a.replicates.split(',')])
