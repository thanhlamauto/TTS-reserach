import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sd15_sinkhorn_tpu.common import atomic_json,initial_noise
from scripts.static_bfs_graph.adapter import event_seed
from .bfs import policy
from .run import OLD


def read(p): return json.loads(p.read_text())


def audit(root):
    root=Path(root); torch.set_num_threads(2)
    plan=read(root/'frozen_plan.json')
    repo=Path(__file__).resolve().parents[1]
    for rel,h in plan['source_sha256'].items():
        p=repo/rel if rel.startswith('sd15_sinkhorn_tpu/') else Path(__file__).parent/rel
        assert hashlib.sha256(p.read_bytes()).hexdigest()==h,('scientific source changed',p)
    assert read(root/'smoke_gate.json')['passed']
    bfs=list((root/'full').glob('*/*/bfs/p*/*/result.json'))
    assert len(bfs)==168,len(bfs)
    actual_bfs=bfs+list((root/'smoke').glob('*/*/bfs/p*/*/result.json'))
    bfs_nfe=0; reward_calls=0; clips=0; latent_files=0
    eta0_ancestors=[]
    for p in actual_bfs:
        r=read(p); n=r['n']; steps=r['steps']
        assert r['finite'] and len(r['scores']['rewards'])==n
        expected=policy(r['policy']['name'],steps)
        assert r['policy']['resampling_steps']==expected['resampling_steps']
        assert [e['sampling_index'] for e in r['events']]==expected['resampling_steps']
        for e,(idx,tau) in zip(r['events'],expected['temperature_by_step']):
            assert e['event_seed']==event_seed(r['seed'],r['prompt_index'],idx)
            assert math.isclose(e['temperature'],tau,rel_tol=1e-12)
            assert len(e['parent_indices'])==n and all(0<=i<n for i in e['parent_indices'])
            w=np.asarray(e['normalized_weights']); assert np.isfinite(w).all() and abs(w.sum()-1)<2e-6
        initial,pseed=initial_noise(r['seed'],r['prompt_index'],0,n)
        assert pseed==r['initial_seed']
        assert hashlib.sha256(initial.view(torch.uint8).numpy().tobytes()).hexdigest()==r['initial_noise_sha256']
        d=torch.load(p.parent/'final.pt',weights_only=True,map_location='cpu')
        assert d['latents'].shape==(n,4,64,64) and torch.isfinite(d['latents']).all()
        latent_files+=1; bfs_nfe+=r['diffusion_sample_steps']; reward_calls+=r['reward_evaluations']; clips+=n
        if steps==30: eta0_ancestors.append(r['unique_final_latents'])
    guided_nfe=0
    guided=list(root.glob('full/*/*/trials/p*/*/result.json'))
    assert len(guided)==48,len(guided)
    for p in guided+list(root.glob('smoke/*/*/trials/p*/*/result.json')):
        r=read(p); guided_nfe+=r['incremental_sample_steps']; reward_calls+=r['incremental_reward_evaluations']; clips+=r['n_images']
        x=torch.load(p.parent/'final.pt',weights_only=True,map_location='cpu')['latents']
        assert x.shape[1:]==(4,64,64) and torch.isfinite(x).all(); latent_files+=1
        assert r['bf16_finite'] and max(r['row_error'],r['column_error'])<1e-4
    banks=list(root.glob('full/*/banks/*/p*/bank.json'))
    assert len(banks)==24,len(banks)
    bank_nfe=0; bon_nfe=0; bon_physical=0; replay_nfe=0; replays=[]
    for p in banks+list(root.glob('smoke/*/banks/*/p*/bank.json')):
        bank=read(p)
        for name,s in bank['sets'].items():
            bank_nfe+=s['sample_steps']; reward_calls+=len(s['rewards']); clips+=len(s['rewards'])
            d=torch.load(p.parent/name/'trajectories.pt',weights_only=True,map_location='cpu')
            for x in [d['initial'],d['final'],*d['checkpoints'].values()]:
                assert x.shape[1:]==(4,64,64) and torch.isfinite(x).all()
            latent_files+=1
        for q in p.parent.glob('fresh_bon_*/result.json'):
            r=read(q); bon_nfe+=r['incremental_sample_steps']; bon_physical+=r['physical_padded_sample_steps']
            reward_calls+=r['reward_evaluations']; clips+=r['fresh_count']
            x=torch.load(q.parent/'final.pt',weights_only=True,map_location='cpu')['latents']
            assert x.shape[1:]==(4,64,64) and torch.isfinite(x).all(); latent_files+=1
        for q in p.parent.glob('replay_*.json'):
            r=read(q); assert r['passed']; replay_nfe+=r['sample_steps']; replays.append(r)
    gates=[read(p) for p in root.glob('*/**/gate_*worker*.json')]
    assert len(gates)==14,len(gates)
    assert all(g['passed'] for g in gates)
    segmented_nfe=sum(g['sample_steps'] for g in gates)
    warmups=[read(p) for p in root.glob('*/**/environment/warmup*.json')]
    assert len(warmups)==10,len(warmups)
    warm_nfe=sum(w['warmup_sample_steps'] for w in warmups)
    pngs=list(root.glob('**/images/*.png'))
    for p in pngs:
        with Image.open(p) as im:
            assert im.size==(512,512); im.verify()
    assert len(pngs)==clips,(len(pngs),clips)
    for rank in range(4):
        for phase in ['30eta0','100eta1']:
            assert read(root/'full'/phase/f'status_worker{rank}.json')['state']=='complete'
    # Same source checkpoints and original data are never rewritten by this campaign.
    assert read(OLD/'validation_audit.json')['passed']
    result=dict(passed=True,full_bfs_rows=len(bfs),new_guided_rows=len(guided),new_banks=len(banks),
                png_verified=len(pngs),latent_files_verified=latent_files,replay_checks=len(replays),
                max_replay_latent_abs=max(r['latent_max_abs'] for r in replays),
                segmentation_checks=len(gates),max_segmentation_latent_abs=max(g['latent_max_abs'] for g in gates),
                max_segmentation_image_mae=max(g['image_mae'] for g in gates),
                bfs_sample_steps=bfs_nfe,bank_sample_steps=bank_nfe,guided_suffix_sample_steps=guided_nfe,
                bon_logical_sample_steps=bon_nfe,bon_physical_sample_steps=bon_physical,
                replay_sample_steps=replay_nfe,segmentation_gate_sample_steps=segmented_nfe,warmup_sample_steps=warm_nfe,
                total_actual_sample_steps=bfs_nfe+bank_nfe+guided_nfe+bon_physical+replay_nfe+segmented_nfe+warm_nfe,
                image_reward_evaluations=reward_calls,clip_image_evaluations=clips,
                inherited_old_run_compute='Excluded from new execution totals; retained in per-method logical budgets.',
                eta0_mean_unique_final_latents=float(np.mean(eta0_ancestors)))
    atomic_json(root/'validation_audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();audit(a.output)
