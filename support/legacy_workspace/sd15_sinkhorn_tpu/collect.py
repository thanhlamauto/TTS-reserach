"""Immutable reference/source banks and a mandatory, model-level replay gate."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from .common import atomic_json,digest,initial_noise
from .sd15_sampler import checkpoint_step
from .reward import evaluate_images


def save_images(images,directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    for i,im in enumerate(images):
        if im.size!=(512,512): raise RuntimeError(f'bad image dimensions {im.size}')
        if np.asarray(im).std()<=1: raise RuntimeError('near uniform image')
        im.save(directory/f'{i:03d}.png')


def load_tensor(path):
    return torch.load(path,map_location='cpu',weights_only=True)


def collect_bank(sampler,reward,clip,directory,prompt,prompt_index,seed,n_ref,n_source):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    expected=dict(prompt=prompt,prompt_index=prompt_index,seed=seed,n_reference=n_ref,n_source=n_source,
                  sampler=sampler.cfg,timesteps=sampler.timesteps)
    ident=digest(expected)
    marker=directory/'bank.json'
    if marker.exists():
        record=json.loads(marker.read_text())
        if record['identity']!=ident: raise RuntimeError('cached bank protocol mismatch')
        return record
    steps=sorted({checkpoint_step(p,sampler.cfg['steps']) for p in sampler.cfg['checkpoints']})
    record=dict(**expected,identity=ident,sets={})
    for name,n,stream in [('source',n_source,0),('reference',n_ref,1)]:
        sub=directory/name; sub.mkdir(exist_ok=True)
        checkpoint_file=sub/'trajectories.pt'; score_file=sub/'scores.json'
        initial,pseed=initial_noise(seed,prompt_index,stream,n)
        if checkpoint_file.exists():
            data=load_tensor(checkpoint_file)
            if data['identity']!=ident: raise RuntimeError('partial bank identity mismatch')
            final=data['final']; elapsed=data['denoise_seconds']
        else:
            final,checkpoints,elapsed=sampler.run(prompt,initial,checkpoints=steps,noise_seed=pseed)
            data=dict(initial=initial,final=final,checkpoints=checkpoints,identity=ident,denoise_seconds=elapsed)
            torch.save(data,checkpoint_file)
        if score_file.exists():
            scored=json.loads(score_file.read_text())
        else:
            t=time.perf_counter(); images=sampler.decode(final); decode_s=time.perf_counter()-t
            save_images(images,sub/'images')
            scored=evaluate_images(prompt,images,reward,clip)
            scored['decode_seconds']=decode_s
            atomic_json(score_file,scored)
        record['sets'][name]=dict(seed=pseed,denoise_seconds=elapsed,sample_steps=n*sampler.cfg['steps'],
                                  unet_cfg_branch_evaluations=2*n*sampler.cfg['steps'],**scored)
    if record['sets']['source']['seed']==record['sets']['reference']['seed']:
        raise RuntimeError('reference/source seeds must be disjoint')
    atomic_json(marker,record)
    return record


def validate_replay(sampler,directory,bank,completed):
    path=Path(directory)/f'replay_{completed}.json'
    if path.exists():
        data=json.loads(path.read_text())
        if not data['passed'] or data['bank_identity']!=bank['identity']: raise RuntimeError('replay gate failed')
        return data
    source=load_tensor(Path(directory)/'source'/'trajectories.pt')
    final,_,seconds=sampler.run(bank['prompt'],source['checkpoints'][completed],start=completed,
                                noise_seed=bank['sets']['source']['seed'])
    expected=source['final']
    delta=final.float()-expected.float()
    # Decode both through the same fixed batch shape. Tolerance is measured on RGB uint8.
    a,b=sampler.decode(final),sampler.decode(expected)
    differences=np.stack([np.asarray(x).astype('float32')-np.asarray(y).astype('float32') for x,y in zip(a,b)])
    cfg=sampler.cfg
    passed=bool(torch.allclose(final.float(),expected.float(),atol=cfg['replay_latent_atol'],rtol=cfg['replay_latent_rtol'])
                and np.abs(differences).mean()<=cfg['replay_image_mae_max'])
    record=dict(passed=passed,bank_identity=bank['identity'],completed_updates=completed,
                sampling_index=completed-1,remaining_timesteps=sampler.timesteps[completed:],
                latent_max_abs=float(delta.abs().max()),latent_rms=float(delta.square().mean().sqrt()),
                image_mae=float(np.abs(differences).mean()),image_max_abs=float(np.abs(differences).max()),
                seconds=seconds,sample_steps=len(final)*(cfg['steps']-completed),
                tolerance=dict(atol=cfg['replay_latent_atol'],rtol=cfg['replay_latent_rtol'],
                               image_mae_max=cfg['replay_image_mae_max']))
    atomic_json(path,record); print('REPLAY',record,flush=True)
    if not passed: raise RuntimeError('model replay validation failed; sweep blocked')
    return record


if __name__=='__main__':
    from .sweep import main
    main(default_action='collect')
