"""One fixed, bounded TRAIN pilot; no reward-dependent expansion or tuning."""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import torch
import yaml

from .adapter import frozen_xla_factory
from .export import export_bank
from .io import atomic_json, file_sha, load_tensor, tensor_sha
from .runner import code_identity, make_plan, replay, run_plan
from .worker import WorkerClient


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--worker-python', required=True)
    args = p.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'campaign.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run = root / 'attempts' / code_identity()[:12]
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'COMPLETE.json').exists():
        print('ALREADY_COMPLETE', run, flush=True)
        return
    start = time.time()
    worker, adapter = None, None

    def status(stage, **extra):
        record = dict(stage=stage, pid=os.getpid(), source=code_identity(), updated_unix=time.time(),
                      elapsed_seconds=time.time() - start, run=str(run))
        record.update(extra)
        atomic_json(run / 'status.json', record)
        atomic_json(root / 'status.json', record)
        print('STAGE', stage, json.dumps(extra), flush=True)

    def stop(signum, frame):
        raise TimeoutError(f'bounded campaign interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    try:
        status('preflight')
        expected = {'torch': '2.4.0+cpu', 'torch_xla': '2.4.0', 'transformers': '4.38.2', 'diffusers': '0.31.0.dev0'}
        versions = {k: importlib.metadata.version(k) for k in expected}
        if versions != expected:
            raise RuntimeError(f'frozen sampler environment changed: {versions}')
        owners = subprocess.run(['fuser', '/dev/vfio/0', '/dev/vfio/1', '/dev/vfio/2', '/dev/vfio/3'],
                                capture_output=True, text=True)
        if owners.stdout.strip():
            raise RuntimeError('TPU devices are owned; do not launch duplicate/conflicting work')
        from sd15_sinkhorn_tpu.common import load_config, read_prompts
        from sd15_sinkhorn_tpu.collect import collect_bank
        from sd15_sinkhorn_tpu.reward import CLIPCPU
        legacy = Path('/home/thanhlamtba31/TTS-reserach')
        source_paths = list((legacy / 'sd15_sinkhorn_tpu').glob('*.py'))
        source_paths += [legacy / 'scripts/diffusion_classical_search_t2i/tpu_runner.py']
        legacy_sha = {str(f): file_sha(f) for f in source_paths}
        cfg = {**load_config(), 'steps': 100, 'ddim_eta': 1., 'seed': 20261016}
        prompt = read_prompts('train')[0]
        prereg = dict(encoder='facebook/dinov2-base', revision='f9e44c814b77203eaa57a6bdbbd535f21ede1415',
                      prompt=prompt, prompt_index=0, split='TRAIN', seed=20261016, source8=8, bank16=16,
                      methods=9, sampler=cfg, legacy_sha256=legacy_sha, wall_cap_seconds=3600,
                      active_tpu_chips=1, allocated_tpu_chips=4, reward_dependent_expansion=False)
        if (run / 'preregistered.json').exists() and json.loads((run / 'preregistered.json').read_text()) != prereg:
            raise RuntimeError('frozen preregistration differs')
        atomic_json(run / 'preregistered.json', prereg)
        worker_cfg = Path(__file__).parents[1] / 'configs/worker_dinov2.yaml'
        worker = WorkerClient(args.worker_python, worker_cfg, run / 'worker_requests')
        atomic_json(run / 'worker_identity.json', worker.identity)
        runtime = run / 'neural_runtime'
        runtime.mkdir(exist_ok=True)
        tmpcache = Path('/dev/shm') / ('otlab_dinov2_' + code_identity()[:12])
        tmpcache.mkdir(exist_ok=True)
        if not (runtime / 'xla_cache').exists():
            (runtime / 'xla_cache').symlink_to(tmpcache, target_is_directory=True)
        adapter_cfg = dict(legacy_pythonpath=[str(legacy)], legacy_source_hashes=legacy_sha,
                           runtime_output=str(runtime), dcs_root='/home/thanhlamtba31/static-bfs-dcs',
                           sampler=cfg, prompt=prompt)
        atomic_json(run / 'adapter_config.json', adapter_cfg)
        status('load_frozen_sampler')
        adapter = frozen_xla_factory(adapter_cfg)
        atomic_json(run / 'sampler_environment.json', dict(versions=versions, identity=adapter.identity))
        clip = CLIPCPU(cfg, '/home/thanhlamtba31/static-bfs-dcs')
        status('new_train_bank')
        bankdir = root / 'bank_train_seed20261016' / 'p000'
        bank = collect_bank(adapter.sampler, adapter.reward, clip, bankdir, prompt, 0, 20261016, 16, 8)
        atomic_json(run / 'bank_receipt.json', dict(identity=bank['identity'], base_nfe=2400, directory=str(bankdir)))
        export = run / 'export'
        if not (export / 'checkpoint.pt').exists():
            export_bank(bankdir, adapter, export)
        cp = load_tensor(export / 'checkpoint.pt')
        clean_rgb = adapter.sampler.decode(cp['source_clean'].to(torch.bfloat16))
        for i, image in enumerate(clean_rgb):
            image.save(export / f'source_current_clean_{i:03d}.png')
        status('actual_vae_encoder_gradient_gate')
        preflight = worker.call('preflight', cp, {})
        atomic_json(run / 'gradient_preflight.json', preflight)
        if not preflight['passed']:
            raise RuntimeError('actual VAE/DINOv2 gradient preflight failed; do not sample methods')
        planpath = run / 'plan'
        plan = json.loads((planpath / 'plan.json').read_text()) if (planpath / 'plan.json').exists() else make_plan(export / 'manifest.jsonl', planpath)
        status('trusted_and_duplicate_replay')
        proof = replay(plan, lambda _: adapter, run / 'replay.json')
        status('nine_paired_methods')
        rows = run_plan(plan, proof, lambda _: adapter, worker, run / 'results', root / 'cache',
                        execution_identity=dict(worker=worker.identity, legacy=legacy_sha, encoder_config_sha=file_sha(worker_cfg)))
        worker.close()
        worker = None
        status('terminal_evaluation')
        evaluate = [args.worker_python, '-u', '-m', 'otlab.evaluate', '--root', str(run / 'results'),
                    '--bank', str(bankdir), '--config', str(worker_cfg), '--out', str(run / 'dino_terminal.json')]
        subprocess.run(evaluate, check=True)
        from PIL import Image
        import numpy as np
        clip_rows = []
        for row in rows:
            terminal = load_tensor(run / 'results' / row['job_id'] / 'terminal.pt')
            images = [Image.fromarray(rgb.permute(1, 2, 0).numpy()) for rgb in terminal['rgb']]
            scored = clip.score(prompt, images)
            f = np.asarray(scored['clip_embeddings'])
            distance = 1 - f @ f.T
            clip_rows.append(dict(method=row['method'], clip_alignment_mean=float(np.mean(scored['clip_alignment'])),
                                  clip_diversity=float(distance[~np.eye(len(f), dtype=bool)].mean()), **scored))
        atomic_json(run / 'clip_terminal.json', clip_rows)
        if len(rows) != 9 or len({r['method'] for r in rows}) != 9 or not proof['passed']:
            raise RuntimeError('pilot completeness integrity failed')
        for path, expected_sha in legacy_sha.items():
            if file_sha(path) != expected_sha:
                raise RuntimeError('legacy source changed while running')
        elapsed = time.time() - start
        finish = dict(passed=True, rows=9, prompts=1, conditional_replicates=1,
                      scientific_winner=None, evidence='engineering smoke; single TRAIN prompt',
                      elapsed_seconds=elapsed, active_chip_hours=elapsed / 3600,
                      allocated_slice_chip_hours=4 * elapsed / 3600, actual_billing_usd=None,
                      new_tpu_counters=dict(adapter.counters), worker_environment=str(root / 'worker_environment.freeze.txt'))
        atomic_json(run / 'COMPLETE.json', finish)
        status('complete', **finish)
    except BaseException as exc:
        atomic_json(run / 'failure.json', dict(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()))
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if worker is not None:
            worker.close()
        if adapter is not None:
            (run / 'xla_metrics.txt').write_text(adapter.sampler.rt.metrics())


if __name__ == '__main__':
    main()
