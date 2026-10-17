"""Cold-budget paired TRAIN pilot on the unchanged SD1.5 XLA sampler."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from PIL import Image, ImageDraw

from otlab.adapter import frozen_xla_factory, schedule_identity
from otlab.export import historical_tape
from otlab.noise import NoiseTape, new_tape
from otlab.io import atomic_json, file_sha, tensor_sha, load_tensor
from ot_next.identity import compatible_scheduler_metadata
from ot_next.output_selection import quality_diversity_indices, cosine_distance
from noise_avoid.campaign import make_basis, CompactRGB
from noise_avoid.model import probe_actions
from .proposal import gradient, sample, checks


def difference(after, before):
    return {k: v-before.get(k, 0) for k, v in after.items()}


def sheet(rgb, path, labels):
    image = Image.new('RGB', (4*192, 2*214), 'white')
    draw = ImageDraw.Draw(image)
    for i, (im, text) in enumerate(zip(rgb, labels)):
        x, y = (i % 4)*192, (i // 4)*214
        image.paste(Image.fromarray(im.permute(1, 2, 0).numpy()).resize((192, 192)), (x, y))
        draw.text((x+4, y+195), text, fill='black')
    image.save(path, quality=90)


def measure(adapter, encoder, cfg, u, case, name, bundle, features, clip):
    count = len(bundle['rewards'])
    expected = 24 if name == 'bfs_graph' else 32
    assert count == expected
    top = torch.argsort(bundle['rewards'], descending=True, stable=True)[:8].tolist()
    qd = quality_diversity_indices(bundle['rewards'].numpy(), features.numpy(),
                                   cfg['calibration']['quality_threshold'])
    out = dict(latent=bundle['latent'], rgb=bundle['rgb'], rewards=bundle['rewards'],
               dino=features, clip=clip, root_ids=bundle['root_ids'], role=bundle['role'])
    torch.save(out, case/(name+'.pt'))
    views = {}
    for selector, ids in [('top8_ir', top), ('qd8', qd)]:
        dc, dd = cosine_distance(clip[ids].numpy()), cosine_distance(features[ids].numpy())
        views[selector] = dict(indices=ids, mean_ir=float(bundle['rewards'][ids].mean()),
            clip_diversity=float(dc[np.triu_indices(8, 1)].mean()),
            dino_diversity=float(dd[np.triu_indices(8, 1)].mean()),
            guided_count=sum(bundle['role'][i] == 'guided' for i in ids),
            scout_count=sum(bundle['role'][i].startswith('scout') for i in ids))
        sheet(bundle['rgb'][ids], case/f'{name}_{selector}.jpg',
              [f'{j+1} IR {float(bundle["rewards"][i]):.3f}' for j, i in enumerate(ids)])
    row = dict(method=name, unit_id=u['unit_id'], prompt_id=u['prompt_id'], seed=u['seed'],
        terminal_pool=count, best_ir=float(bundle['rewards'].max()),
        pool_mean_ir=float(bundle['rewards'].mean()), views=views,
        nominal_cold_sampling_nfe=2400, physical_cold_unet_sample_nfe=2400,
        auxiliary_unet_nfe=0, physical_padding_nfe=0,
        cold_cfg_branch_sample_evaluations=4800,
        cold_terminal_ir_scores=count, cold_proxy_ir_scores=72 if name == 'bfs_graph' else 0,
        root_count=24 if name == 'bfs_graph' else 12,
        unique_terminal_root_ancestors=len(set(bundle['root_ids'])),
        endpoint_sha=tensor_sha(bundle['latent']), bundle_sha256=file_sha(case/(name+'.pt')),
        scout_outputs_in_pool=True if name != 'bfs_graph' else None,
        semantic_success='NOT_EVALUATED_DO_NOT_INFER_FROM_REWARD')
    if name != 'bfs_graph':
        shared = [i for i, role in enumerate(bundle['role']) if role != 'guided']
        row['shared29_best_ir'] = float(bundle['rewards'][shared].max())
        row['guided_incremental_best_ir'] = row['best_ir']-row['shared29_best_ir']
        row['guided_best_ir'] = float(bundle['rewards'][20:23].max())
    atomic_json(case/(name+'.json'), row)
    return row


def run_unit(adapter, encoder, cfg, u, case, status, smoke):
    from sd15_sinkhorn_tpu.common import initial_noise
    from sd15_sinkhorn_bfs_compare.bfs import advance, policy, make_adapter
    from sd15_sinkhorn_tpu.reward import CLIPCPU
    case.mkdir(parents=True, exist_ok=False)
    begin = time.time()
    before = dict(adapter.counters)
    seed, pid = u['seed'], u['prompt_id']
    adapter.set_prompt(u['prompt'])
    initial, noise_seed = initial_noise(seed, pid, 0, 24)
    historical = historical_tape(adapter.sampler, noise_seed, 24, start=0)
    status('mean_prefix12', unit_id=u['unit_id'])
    prefix = adapter.advance(initial[:12], 0, 40, tuple(range(12)), historical)
    if smoke:
        status('original_full100_and_bfs_replay', unit_id=u['unit_id'])
        full, cps, _ = adapter.sampler.run(adapter.prompt, initial[:4], start=0,
                                          checkpoints=(40,), noise_seed=noise_seed)
        adapter.counters['original_full_replay_sample_steps'] += 400
        adapter.counters['unet_cfg_branch_sample_evaluations'] += 800
        resumed = adapter.advance(prefix[:4], 40, 100, tuple(range(4)), historical)
        direct, _, _ = advance(adapter.sampler, adapter.prompt, prefix[:4], 40, 100, noise_seed)
        adapter.counters['bfs_replay_sample_steps'] += 240
        adapter.counters['unet_cfg_branch_sample_evaluations'] += 480
        assert torch.equal(prefix[:4], cps[40]), 'original prefix40 drifted'
        assert torch.equal(full, resumed), 'original full100 vs resume drifted'
        assert torch.equal(full, direct), 'BFS fused continuation drifted'
        atomic_json(case/'REPLAY.json', dict(passed=True, cases=4,
            full100_resume_bitwise=True, bfs_resume_bitwise=True,
            sampling_nfe_extra=880, replay_separate_from_each_method_budget=True))
    z, _ = probe_actions(cfg['radius'], pid, seed)
    z = z[:18]
    sigmas = [float(adapter.sampler.coefficients[i][4]) for i in cfg['noise_indices']]
    basis, basis_info = make_basis(sigmas, seed, pid)
    scouts = torch.cat([new_tape(seed, pid, 100+k, (0,), (4,64,64), adapter.schedule_sha,
                                 start=40).variance_noise.repeat(1,9,1,1,1) for k in range(2)], dim=1)
    scouts[:5] = historical.variance_noise[40:45, :1] + torch.einsum(
        'jchwk,nk->jnchw', basis, torch.tensor(z, dtype=torch.float32))
    first_noise = torch.cat([scouts, historical.variance_noise[40:, 1:3]], dim=1)
    first_state = torch.cat([prefix[:1].repeat(18,1,1,1), prefix[1:3]])
    first_tape = NoiseTape(first_noise, 40, tuple(range(20)), adapter.schedule_sha)
    status('scout18_plus_independent2', unit_id=u['unit_id'])
    s45 = adapter.advance(first_state, 40, 45, tuple(range(20)), first_tape)
    assert torch.equal(s45[:9], s45[9:18]), 'A/B action-block mismatch'
    first_final = adapter.advance(s45, 45, 100, tuple(range(20)), first_tape)
    first_reward, first_rgb = adapter.score(first_final, 'terminal')
    g_tapes, g = gradient(first_reward[:18].numpy(), cfg['radius'])
    guided_z, uniform_z, audit = sample(cfg, seed, pid, g)
    common_tail = torch.cat([new_tape(seed, pid, 200+k, (0,), (4,64,64),
        adapter.schedule_sha, start=40).variance_noise for k in range(3)], dim=1)
    second_state = torch.cat([prefix[:1].repeat(3,1,1,1), prefix[3:12]])
    pools = {}
    second_outputs = []
    for name, actions in [('mean_gradient', guided_z), ('allocation_uniform', uniform_z)]:
        status('new_suffixes', unit_id=u['unit_id'], method=name)
        guided_noise = common_tail.clone()
        guided_noise[:5] = historical.variance_noise[40:45, :1] + torch.einsum(
            'jchwk,nk->jnchw', basis, torch.tensor(actions, dtype=torch.float32))
        noise = torch.cat([guided_noise, historical.variance_noise[40:, 3:12]], dim=1)
        tape = NoiseTape(noise, 40, tuple(range(12)), adapter.schedule_sha)
        current = adapter.advance(second_state, 40, 100, tuple(range(12)), tape)
        reward, rgb = adapter.score(current, 'terminal')
        second_outputs.append(current)
        pools[name] = dict(latent=torch.cat([first_final, current]),
            rewards=torch.cat([first_reward, reward]), rgb=torch.cat([first_rgb, rgb]),
            root_ids=[0]*18+[1,2]+[0]*3+list(range(3,12)),
            role=['scout_A']*9+['scout_B']*9+['independent']*2+['guided']*3+['independent']*9)
    assert torch.equal(second_outputs[0][3:], second_outputs[1][3:]), 'uniform control altered shared roots'
    if np.array_equal(guided_z, uniform_z):
        assert torch.equal(second_outputs[0], second_outputs[1]), 'flat filter differs from uniform'
    status('bfs_graph24', unit_id=u['unit_id'])
    record = policy('graph', 100)
    assert record['resampling_steps'] == cfg['bfs']['schedule']
    assert [t for k,t in record['temperature_by_step']] == cfg['bfs']['tau']
    def proxy_score(clean):
        values, _ = adapter.score(clean, 'proxy')
        return values
    search = make_adapter(24, record, seed, pid, proxy_score)
    x, start, ancestors = initial, 0, torch.arange(24)
    for event in record['resampling_steps']:
        x, x0, _ = advance(adapter.sampler, adapter.prompt, x, start, event+1, noise_seed)
        adapter.counters['bfs_denoise_sample_steps'] += 24*(event+1-start)
        adapter.counters['unet_cfg_branch_sample_evaluations'] += 48*(event+1-start)
        x, _ = search.resample(sampling_idx=event, latents=x, x0_preds=x0)
        log = search.events[-1]
        ancestors = ancestors[log['parent_indices']]
        log['unique_initial_ancestors'] = len(set(ancestors.tolist()))
        log['completed_updates'] = event+1
        start = event+1
    x, _, _ = advance(adapter.sampler, adapter.prompt, x, start, 100, noise_seed)
    adapter.counters['bfs_denoise_sample_steps'] += 24*(100-start)
    adapter.counters['unet_cfg_branch_sample_evaluations'] += 48*(100-start)
    reward, rgb = adapter.score(x, 'terminal')
    pools['bfs_graph'] = dict(latent=x, rewards=reward, rgb=rgb,
        root_ids=ancestors.tolist(), role=['bfs_terminal']*24)
    assert all(torch.isfinite(pools[n]['latent']).all() and torch.isfinite(pools[n]['rewards']).all()
               for n in pools)
    atomic_json(case/'ALLOCATION.json', dict(gradient_A_B=g_tapes.tolist(), gradient_mean=g.tolist(),
        guided_z=guided_z.tolist(), uniform_z=uniform_z.tolist(), proposal_audit=audit,
        basis=basis_info, scout_actions=z.tolist(), noise_seed=noise_seed,
        shared_initial_first12_sha=tensor_sha(initial[:12]),
        bfs_initial24_sha=tensor_sha(initial), shared_suffix9_bitwise=True,
        replayed_same_proposals=bool(np.array_equal(guided_z, uniform_z)),
        cfg=cfg['mean_allocation'], bfs_events=search.events,
        terminal_data_not_used_for_guidance=True, shared_scout_first20_sha=tensor_sha(first_final)))
    # Encode shared pixels once. Reuse is accounting-only; selectors are identical.
    status('terminal_features_and_common_selectors', unit_id=u['unit_id'])
    all_rgb = torch.cat([first_rgb, pools['mean_gradient']['rgb'][20:],
                        pools['allocation_uniform']['rgb'][20:], rgb])
    assert len(all_rgb) == 68
    f = encoder.encode(all_rgb, u['unit_id']+'_terminal68')
    if not hasattr(adapter, 'clip'):
        adapter.clip = CLIPCPU(adapter.sampler.cfg, cfg['dcs_root'])
    clip = adapter.clip.score(adapter.prompt,
        [Image.fromarray(im.permute(1,2,0).numpy()) for im in all_rgb])
    clip = torch.tensor(clip['clip_embeddings'])
    maps = dict(mean_gradient=list(range(32)),
                allocation_uniform=list(range(20))+list(range(32,44)),
                bfs_graph=list(range(44,68)))
    rows = [measure(adapter, encoder, cfg, u, case, name, pools[name], f[maps[name]], clip[maps[name]])
            for name in cfg['methods']]
    counters = difference(adapter.counters, before)
    actual_nfe = sum(counters.get(k,0) for k in ['denoise_sample_steps',
        'bfs_denoise_sample_steps','original_full_replay_sample_steps','bfs_replay_sample_steps'])
    assert actual_nfe == 5520 + (880 if smoke else 0), counters
    assert counters.get('denoise_padding_sample_steps', 0) == 0
    assert counters.get('aux_clean_sample_evaluations', 0) == 0
    assert counters['unet_cfg_branch_sample_evaluations'] == 2*actual_nfe
    assert counters['terminal_verifier_scores'] == 68 and counters['proxy_verifier_scores'] == 72
    result = dict(passed=True, unit=u, rows=rows, physical_sampling_nfe=actual_nfe,
        physical_padding_nfe=0, aux_unet_nfe=0, counters=counters,
        cold_all3_methods_sampling_nfe=7200, shared_execution_saving_nfe=1680,
        separate_replay_nfe=880 if smoke else 0,
        terminal_feature_images=68, elapsed_seconds=time.time()-begin)
    atomic_json(case/'COMPLETE.json', result)
    status('unit_complete', unit_id=u['unit_id'], actual_nfe=actual_nfe)


def worker(cfg, rank, source):
    out = Path(cfg['root'])/f'chip{rank}'
    out.mkdir(parents=True, exist_ok=True)
    begin = time.time()
    adapter = encoder = None
    def status(stage, **extra):
        row = dict(stage=stage, rank=rank, pid=os.getpid(), elapsed_seconds=time.time()-begin, **extra)
        atomic_json(out/'status.json', row)
        print(json.dumps(row), flush=True)
    def interrupted(signum, frame):
        raise TimeoutError('persistent deadline or cancellation')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        status('load_verified_backend')
        old = Path(cfg['old_root'])/'attempts/8074f94c847d'
        ac = json.loads((old/'adapter_config.json').read_text())
        runtime = out/'runtime'
        runtime.mkdir(exist_ok=True)
        # Only immutable-compatible XLA graphs are cached; no disk persistence.
        cache = Path('/dev/shm')/('mean_bfs_'+file_sha(source/'protocol.json')[:12])/f'chip{rank}'
        cache.mkdir(parents=True, exist_ok=True)
        if not (runtime/'xla_cache').exists():
            (runtime/'xla_cache').symlink_to(cache, target_is_directory=True)
        ac['runtime_output'] = str(runtime)
        adapter = frozen_xla_factory(ac)
        prior = json.loads((old/'neural_runtime/environment/sampler0.json').read_text())['scheduler_config']
        assert compatible_scheduler_metadata(prior, dict(adapter.sampler.scheduler.config))
        adapter.sampler.scheduler.register_to_config(_use_default_values=prior['_use_default_values'])
        expected = json.loads((old/'replay.json').read_text())['records'][0]['adapter_identity']['schedule_sha']
        assert schedule_identity(adapter.sampler) == expected
        adapter.schedule_sha = expected
        adapter.base_identity['schedule_sha'] = expected
        for path, expected_hash in cfg['bfs_source_hashes'].items():
            assert file_sha(path) == expected_hash, path
        encoder = CompactRGB(str(Path(cfg['old_root'])/'worker_venv/bin/python'),
            source.parents[1]/'ot_tts_lab_v2_reimpl/configs/worker_dinov2.yaml', out/'rgb_worker')
        atomic_json(out/'identity.json', dict(adapter=adapter.identity, encoder=encoder.identity,
            visible_chip=os.environ['TPU_VISIBLE_CHIPS'], bf16=True, backend_unchanged=True))
        units = cfg['units'][rank::4]
        for pos, u in enumerate(units):
            case = out/u['unit_id']
            if (case/'COMPLETE.json').exists():
                continue
            run_unit(adapter, encoder, cfg, u, case, status, smoke=(rank==0 and pos==0))
            if rank == 0 and pos == 0:
                atomic_json(Path(cfg['root'])/'SMOKE_COMPLETE.json', dict(passed=True, unit=u['unit_id'],
                    replay_cases=4, method_cold_nfe=2400, padding=0))
                while not (Path(cfg['root'])/'RELEASE_FULL.json').exists():
                    time.sleep(1)
        atomic_json(out/'COMPLETE.json', dict(passed=True, rank=rank,
            counters=dict(adapter.counters), elapsed_seconds=time.time()-begin,
            encoder_images=encoder.count))
    except BaseException as exc:
        atomic_json(out/'failure.json', dict(error=str(exc), traceback=traceback.format_exc(),
            counters={} if adapter is None else dict(adapter.counters)))
        raise
    finally:
        if encoder:
            encoder.close()
        if adapter:
            (out/'xla_metrics.txt').write_text(adapter.sampler.rt.metrics())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', required=True)
    p.add_argument('--worker', type=int)
    args = p.parse_args()
    source = Path(args.protocol).resolve().parent
    cfg = json.loads(Path(args.protocol).read_text())
    if args.worker is not None:
        return worker(cfg, args.worker, source)
    root = Path(cfg['root'])
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'campaign.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'COMPLETE.json').exists():
        print('ALREADY_COMPLETE', flush=True)
        return
    assert not list(root.glob('chip*')), 'Existing workers/partial data: inspect before repair/resume'
    assert shutil.disk_usage(root).free > 2*1024**3, 'Need2GiB free'
    assert not subprocess.run(['fuser']+[f'/dev/vfio/{r}' for r in range(4)],
        capture_output=True,text=True).stdout.strip(), 'TPU already owned'
    atomic_json(root/'STATIC_GATES.json', checks(cfg))
    atomic_json(root/'preregistered.json', cfg)
    atomic_json(root/'SOURCE_HASHES.json', {str(p.relative_to(source)):file_sha(p)
        for p in source.rglob('*') if p.is_file()})
    processes = []
    begin = time.time()
    def interrupted(signum, frame):
        raise TimeoutError('persistent campaign deadline')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for rank in range(4):
            if rank == 1:
                while not (root/'SMOKE_COMPLETE.json').exists():
                    assert processes[0][0].poll() is None, 'Single-chip smoke failed'
                    time.sleep(1)
                atomic_json(root/'RELEASE_FULL.json', dict(passed=True, released_unix=time.time()))
                print('SMOKE_PASS_RELEASE_FOUR_CHIPS', flush=True)
            log = (root/f'chip{rank}.log').open('a')
            child = subprocess.Popen([sys.executable,'-B','-u','-m','mean_bfs.campaign',
                '--protocol',args.protocol,'--worker',str(rank)],
                env=dict(os.environ,TPU_VISIBLE_CHIPS=str(rank)),stdout=log,stderr=subprocess.STDOUT)
            processes.append((child,log))
            atomic_json(root/'worker_pids.json',[p.pid for p,l in processes])
        exits = [p.wait() for p,l in processes]
        assert not any(exits), f'Worker failure {exits}'
        receipts = [json.loads(p.read_text()) for p in sorted(root.glob('chip*/p*/COMPLETE.json'))]
        assert len(receipts) == 12 and len({r['unit']['unit_id'] for r in receipts}) == 12
        nfe = sum(r['physical_sampling_nfe'] for r in receipts)
        assert nfe == cfg['neural_nfe_cap'] == 67120
        assert not list(root.glob('chip*/failure.json'))
        from .analyze import analyze
        analyze(root, cfg)
        result = dict(passed=True, units=12, prompts=6, seeds_per_prompt=2, methods=3,
            method_rows=36, physical_sampling_nfe=nfe, physical_padding_nfe=0,aux_unet_nfe=0,
            cfg_branch_sample_evaluations=2*nfe, replay_extra_sampling_nfe=880,
            cold_all3_methods_nfe=86400, common_prefix_scout_execution_savings=20160,
            occupied_worker_chip_hours=sum(json.loads((root/f'chip{r}/COMPLETE.json').read_text())[
                'elapsed_seconds'] for r in range(4))/3600,
            elapsed_seconds=time.time()-begin, physical_visible_chips=[0,1,2,3],
            reserved_final_used=False, actual_billing_usd=None, owner_attested_no_charge=True,
            scientific_scope='exploratory matched TRAIN pilot, not confirmation')
        atomic_json(root/'COMPLETE.json', result)
        print('COMPLETE', json.dumps(result), flush=True)
    except BaseException as exc:
        atomic_json(root/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),
            elapsed_seconds=time.time()-begin))
        raise
    finally:
        for child, log in processes:
            if child.poll() is None:
                child.terminate()
            log.close()


if __name__ == '__main__':
    main()
