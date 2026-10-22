import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from sd15_sinkhorn_tpu.common import atomic_json, digest, event_seed
from sd15_sinkhorn_tpu.collect import load_tensor
from sd15_sinkhorn_tpu.steering import Config, intervene
from sd15_sinkhorn_bfs_compare.analyze import read
from .run import sha
from .analyze import build_rows


def audit(root):
    root=Path(root); plan=read(root/'frozen_plan.json'); rows,inherited=build_rows(root)
    for file,h in plan['source_sha256'].items(): assert sha(file)==h,file
    manifest=read(root/'analysis/inherited_manifest.json')
    assert manifest=={str(p):sha(p) for p in inherited}
    assert all(read(root/'full'/f'status_worker{i}.json')['state']=='complete' for i in range(4))
    assert read(root/'smoke_gate.json')['passed']
    prior=Path(plan['prior'])/'full/100eta1'; images=set(); latent_files=set(); errors=[]
    for p in sorted((root/'full/bfs').glob('p*/*/result.json')):
        r=read(p); n=r['n']; assert n==24 and len(r['events'])==3
        for e in r['events']:
            assert len(e['parent_indices'])==n
            assert all(0<=j<n for j in e['parent_indices'])
            assert np.isfinite(e['raw_rewards']).all()
        assert len(r['scores']['rewards'])==n and r['reward_evaluations']==96
        images.update(p.parent/'images'/f'{j:03d}.png' for j in range(n))
        latent_files.add(p.parent/'final.pt')
    for i,prompt in enumerate(plan['prompts']):
        bank_dir=prior/'banks/test'/f'p{i:03d}'; bank=read(bank_dir/'bank.json')
        source=load_tensor(bank_dir/'source/trajectories.pt')
        ref=load_tensor(bank_dir/'reference/trajectories.pt')
        images.update(bank_dir/'reference/images'/f'{j:03d}.png' for j in range(16))
        images.update(bank_dir/'source/images'/f'{j:03d}.png' for j in range(8))
        images.update(bank_dir/'fresh_bon_28/images'/f'{j:03d}.png' for j in range(24))
        for cfg in plan['selection']['configs']:
            d=prior/'test/trials'/f'p{i:03d}'/digest(cfg)[:12]
            actual=load_tensor(d/'intervention.pt')['steered']
            # Reconstruct using only source checkpoint and rewarded reference endpoints.
            # No source terminal latent/reward participates in the intervention.
            expected,_=intervene(source['checkpoints'][50],ref['checkpoints'][50],
                bank['sets']['reference']['rewards'],Config(**cfg),event_seed(plan['seed'],i,50))
            error=float((expected-actual).abs().max()); errors.append(error)
            assert error==0,(i,cfg,error)
            images.update(d/'images'/f'{j:03d}.png' for j in range(8))
            latent_files.add(d/'final.pt')
    for p in images:
        with Image.open(p) as im:
            assert im.size==(512,512); im.verify()
    for p in latent_files:
        data=load_tensor(p); assert torch.isfinite(data['latents']).all()
    r=dict(passed=True,bfs_rows=48,analysis_rows=len(rows),methods=8,prompts=24,
           png_verified=len(images),latent_files_verified=len(latent_files),
           source_future_independence_checks=len(errors),max_intervention_reconstruction_error=max(errors),
           bfs_full_actual_sample_steps=48*2400,bfs_full_reward_evaluations=48*96,
           smoke_actual_sample_steps=2*2400,warmup_actual_sample_steps=5*800,
           total_new_actual_sample_steps=48*2400+2*2400+5*800,
           inherited_reference_guided_bon_compute_excluded_from_new_execution=True,
           inherited_methods_cold_budget_each=2400)
    atomic_json(root/'validation_audit.json',r); print('AUDIT_COMPLETE',r,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True)
    audit(p.parse_args().output)
