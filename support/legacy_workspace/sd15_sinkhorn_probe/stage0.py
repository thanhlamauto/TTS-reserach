import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import torch

from sd15_sinkhorn_tpu.common import atomic_json,digest,event_seed,read_prompts
from sd15_sinkhorn_tpu.collect import load_tensor
from sd15_sinkhorn_tpu.sinkhorn import checked,coupling,barycenter,reference_scale,features,quadratic
from sd15_sinkhorn_tpu.steering import Config,targets,intervene
from sd15_sinkhorn_bfs_compare.analyze import read
from .core import diagnostics,controls

DCS=Path('/home/thanhlamtba31/static-bfs-dcs')
PRIOR=DCS/'experiments/sd15-sinkhorn-bfs-v1/full/100eta1'
SELECTION=DCS/'experiments/sd15-sinkhorn-v2/selection_frozen_before_test.json'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare(root,readme):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    selection=read(SELECTION); config=next(c for c in selection['configs'] if c['epsilon']==.1)
    assert config['method']=='A' and config['cost']=='raw' and config['lowpass']==1
    assert config['gamma_neg']==0 and config['jitter']==0 and config['checkpoint']==.5
    source_files=[p for package in ['sd15_sinkhorn_tpu','sd15_sinkhorn_bfs_compare','sd15_sinkhorn_probe'] for p in Path(package).glob('*.py')]
    source_files += [SELECTION,Path(readme)]
    plan=dict(prompts=read_prompts('test'),stage1_indices=list(range(12)),replicates=[0,1],
              selection=selection,config=config,prior=str(PRIOR),seed=20361015,
              steps=100,ddim_eta=1.,checkpoint=50,control_root_seed=9103101,noise_root_seed=8103101,
              methods=['none','A','mean_matched','minus_A','shuffle_matched','random_matched'],
              primary='paired source mean reward gain: A minus none and A minus each matched control',
              secondary=['pool best gain','pool top8 mean gain','candidate winner/top8 contribution','CLIP diversity'],
              analysis=dict(bootstrap=20000,seed=81031,unit='prompt; average suffix replicates within prompt',ties='bank first'),
              original_dtype='bfloat16',variance_noise_dtype='float32 (matches existing sampler arithmetic)',
              variance_noise_protocol='SeedSequence([8103101,prompt_index,replicate,particle_id,step_index]) uint64 modulo (2**63-1), CPU torch.randn FP32; no method key',
              stage0='Reconstruct float32 original solver on real stored checkpoints, retain returned plans without solver refinement; verify original steered tensor. Historical plans were not saved.',
              stage1='New suffix noise protocol: rerun all six methods. Replicate1 is a conditional suffix-noise stability check on the same bank/prefix.',
              max_control_rms_relative_error=.02,source_sha256={str(p):sha(p) for p in source_files})
    dest=root/'frozen_plan.json'
    if dest.exists() and read(dest)!=plan: raise RuntimeError('plan changed; choose a fresh namespace')
    atomic_json(dest,plan); shutil.copy2(readme,root/'TASK_README_VI.md')
    return plan


def trace(x,z,rewards,cfg,meta):
    assert cfg.method=='A' and cfg.cost=='raw' and cfg.lowpass==1 and cfg.gamma_neg==0
    xx,zz=checked(x),checked(z); scale=reference_scale(zz,cfg.cost)
    w,_,_,_=targets(rewards,cfg.mode,cfg.beta,cfg.top_fraction)
    u=torch.full_like(w,1/len(w)); a=torch.full((len(xx),),1/len(xx))
    c=quadratic(features(xx,cfg.cost),features(zz,cfg.cost))/scale
    pt=coupling(xx,zz,w,cfg.epsilon,scale,cfg.cost)
    ps=coupling(xx,zz,u,cfg.epsilon,scale,cfg.cost)
    direction=barycenter(pt,zz)-barycenter(ps,zz)
    pre=cfg.eta*direction; limit=cfg.trust_region*zz.square().mean().sqrt().clamp_min(1e-8)
    norm=pre.square().flatten(1).mean(1).sqrt()
    post=pre*(limit/norm.clamp_min(1e-12)).clamp_max(1)[:,None,None,None]
    steered=checked(xx+post)
    # This is the exact entry conversion used by Sampler.run. Stage1 passes this
    # BF16 tensor directly to the next model call; do not substitute an FP32 sum.
    actual_entry=steered.cpu().to(torch.bfloat16)
    return dict(X=x.clone(),Z=z.clone(),C=c,a=a,w=w,u=u,P_target=pt.p,P_self=ps.p,
        epsilon=cfg.epsilon,direction_used=direction,delta_preclip=pre,
        delta_postclip_fp32=post,X_after=actual_entry,clip_rms=limit,
        solver_target=dict(iterations=pt.iterations,row_error=pt.row_error,column_error=pt.column_error),
        solver_self=dict(iterations=ps.iterations,row_error=ps.row_error,column_error=ps.column_error),
        meta={**meta,'operator_variant':'raw_A','cost_scale':scale}),steered


def stage0(root):
    torch.set_num_threads(8); root=Path(root); plan=read(root/'frozen_plan.json')
    for p,h in plan['source_sha256'].items(): assert sha(p)==h,p
    diagnostics_rows=[]; inherited={}; max_error=0.; max_control_error=0.
    for i,prompt in enumerate(plan['prompts']):
        bank_dir=PRIOR/'banks/test'/f'p{i:03d}'; bank=read(bank_dir/'bank.json')
        assert bank['prompt']==prompt and bank['seed']==plan['seed']
        assert bank['n_reference']==16 and bank['n_source']==8
        assert bank['sampler']['steps']==100 and bank['sampler']['ddim_eta']==1
        source=load_tensor(bank_dir/'source/trajectories.pt')
        ref=load_tensor(bank_dir/'reference/trajectories.pt')
        for p in [bank_dir/'bank.json',bank_dir/'source/trajectories.pt',bank_dir/'reference/trajectories.pt']:
            inherited[str(p)]=sha(p)
        for config in plan['selection']['configs']:
            cfg=Config(**config); cid=digest(config)[:12]
            original=PRIOR/'test/trials'/f'p{i:03d}'/cid
            old=load_tensor(original/'intervention.pt')['steered']
            inherited[str(original/'intervention.pt')]=sha(original/'intervention.pt')
            meta=dict(prompt_index=i,prompt=prompt,run_id=str(root),original_run=str(PRIOR),
                model=bank['sampler']['model'],revision=bank['sampler']['revision'],
                prefix_steps_completed=50,next_sampling_index=50,
                timesteps=bank['timesteps'],timesteps_sha256=digest(bank['timesteps']),
                config=config,sampler_config=bank['sampler'],
                scheduler_config=read(PRIOR/'worker0/schedule.json')['scheduler'],
                dtype=str(source['checkpoints'][50].dtype),bank_identity=bank['identity'],
                source_seed=bank['sets']['source']['seed'],reference_seed=bank['sets']['reference']['seed'],
                provenance='Reconstructed original float32 OT at stored checkpoints; plans captured as returned, not refined.')
            d,y=trace(source['checkpoints'][50],ref['checkpoints'][50],bank['sets']['reference']['rewards'],cfg,meta)
            err=float((y-old).abs().max()); max_error=max(max_error,err)
            assert err==0,(i,cid,err)
            assert torch.equal(d['X_after'],old.to(torch.bfloat16))
            dest=root/'dumps'/f'p{i:03d}_{cid}.pt'; dest.parent.mkdir(parents=True,exist_ok=True)
            torch.save(d,dest); row=diagnostics(d); row['dump']=str(dest); diagnostics_rows.append(row)
            assert max(row['target_plan']['row_mass_error_max'],row['target_plan']['col_mass_error_max'],
                       row['self_plan']['row_mass_error_max'],row['self_plan']['col_mass_error_max'])<1e-5
            if i<12 and cfg.epsilon==.1:
                for rep in plan['replicates']:
                    seed=int(np.random.SeedSequence([plan['control_root_seed'],i,rep]).generate_state(1)[0])
                    out=controls(d,seed); max_control_error=max(max_control_error,out['max_rms_relative_error'])
                    path=root/'controls'/f'p{i:03d}'/f'rep{rep}.pt'; path.parent.mkdir(parents=True,exist_ok=True)
                    torch.save(out,path)
        print('STAGE0_PROMPT',i,flush=True)
    out=root/'stage0'; out.mkdir(exist_ok=True)
    (out/'audit.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in diagnostics_rows))
    summary=[]
    for eps in [.1,.3]:
        rr=[r for r in diagnostics_rows if r['epsilon']==eps]
        summary.append(dict(epsilon=eps,n_prompts=len(rr),
            common_energy_mean=float(np.mean([r['raw_field']['common_energy_fraction'] for r in rr])),
            common_energy_min=min(r['raw_field']['common_energy_fraction'] for r in rr),
            target_row_kl_mean=float(np.mean([r['target_plan']['row_dependency_kl'] for r in rr])),
            interaction_std_over_epsilon_mean=float(np.mean([r['interaction_std_over_epsilon'] for r in rr])),
            cast_lost_fraction_mean=float(np.mean([r['cast_nonzero_proposal_lost_fraction'] for r in rr])),
            cast_relative_l2_error_mean=float(np.mean([r['cast_relative_l2_error'] for r in rr])),
            actual_update_rms_mean=float(np.mean([r['actual_update_rms'] for r in rr])),
            clip_fraction_mean=float(np.mean([r['clip_fraction'] for r in rr]))))
    atomic_json(out/'summary.json',summary); atomic_json(root/'inherited_manifest.json',inherited)
    atomic_json(root/'stage0_gate.json',dict(passed=True,dumps=48,controls=24,
                original_steered_max_abs_error=max_error,max_control_rms_relative_error=max_control_error))
    print('STAGE0_COMPLETE',summary,flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True)
    p.add_argument('--prepare',action='store_true'); p.add_argument('--readme')
    a=p.parse_args()
    if a.prepare: prepare(a.output,a.readme)
    else: stage0(a.output)


if __name__=='__main__': main()
