import argparse
import hashlib
import json
from pathlib import Path

from sd15_sinkhorn_tpu.common import atomic_json, load_config, read_prompts
from sd15_sinkhorn_tpu.reward import ImageRewardCPU, CLIPCPU
from sd15_sinkhorn_tpu.sd15_sampler import Sampler
from sd15_sinkhorn_tpu.tpu_runtime import Runtime
from sd15_sinkhorn_bfs_compare.run import bfs_trial, reconfigure, DCS

PRIOR = DCS/'experiments/sd15-sinkhorn-bfs-v1'
SELECTION = DCS/'experiments/sd15-sinkhorn-v2/selection_frozen_before_test.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    files = [p for package in ['sd15_sinkhorn_tpu','sd15_sinkhorn_bfs_compare',
                              'sd15_sinkhorn_budget2400']
             for p in Path(package).glob('*.py')]
    files += [Path('sd15_sinkhorn_tpu/config.yaml'), SELECTION]
    return {str(p): sha(p) for p in files}


def prepare(root):
    root = Path(root)
    plan = dict(budget=2400, steps=100, ddim_eta=1., prompts=read_prompts('test'),
                seed=load_config()['seed']+100000, prior=str(PRIOR),
                selection=json.loads(SELECTION.read_text()),
                bfs=[['manual',24],['graph',24]], output_images=8,
                primary='mean best terminal ImageReward per prompt',
                secondary=['mean reward of 8 output images','CLIP image diversity'],
                sinkhorn_budget='16*100 reference + 8*50 source prefix + 8*50 guided suffix',
                source_final_excluded=True,
                reuse='Existing frozen guided outputs, reference scores and first24 fresh BoN draws. No new hyperparameter selection.',
                recycled='top8 from reference16+guided8 only; no unguided source endpoints',
                bootstrap=dict(draws=20000,seed=81031,unit='paired prompt'),
                source_sha256=sources())
    dest=root/'frozen_plan.json'
    if dest.exists() and json.loads(dest.read_text())!=plan:
        raise RuntimeError('frozen plan differs; use a new namespace')
    atomic_json(dest,plan)
    return plan


def run(args):
    root=Path(args.output)
    plan=json.loads((root/'frozen_plan.json').read_text())
    for file,digest in plan['source_sha256'].items():
        if sha(file)!=digest: raise RuntimeError(f'source changed: {file}')
    if not args.smoke:
        assert json.loads((root/'smoke_gate.json').read_text())['passed']
    mode='smoke' if args.smoke else 'full'
    rt=Runtime(root/mode/'runtime'/f'worker{args.rank}',args.rank)
    cfg={**load_config(),'steps':100,'ddim_eta':1.}
    sampler=Sampler(rt,cfg,DCS)
    reconfigure(sampler,cfg,rt.output)
    reward=ImageRewardCPU(DCS); clip=CLIPCPU(cfg,DCS)
    prompts=read_prompts('train')[:1] if args.smoke else plan['prompts']
    seed=cfg['seed'] if args.smoke else plan['seed']
    assigned=[i for i in range(len(prompts)) if i%args.workers==args.rank]
    try:
        sampler.warmup(prompts[assigned[0]])
        for i in assigned:
            for name,n in plan['bfs']:
                r=bfs_trial(sampler,reward,clip,prompts[i],i,seed,n,name,
                            root/mode/'bfs'/f'p{i:03d}'/f'{name}_n{n}')
                assert r['diffusion_sample_steps']==2400 and r['finite']
                assert len(r['events'])==3 and r['n']==24
                assert all(len(e['parent_indices'])==24 for e in r['events'])
            atomic_json(root/mode/f'status_worker{args.rank}.json',
                        dict(state='running',last_prompt=i,assigned=assigned))
        atomic_json(root/mode/f'status_worker{args.rank}.json',
                    dict(state='complete',assigned=assigned,
                         reward_evaluations=reward.evaluations))
        if args.smoke:
            atomic_json(root/'smoke_gate.json',dict(passed=True,n=24,steps=100,eta=1,
                        policies=['manual','graph'],prompt_split='train'))
    finally:
        (rt.output/'xla_metrics.txt').write_text(rt.metrics())


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True)
    p.add_argument('--prepare',action='store_true'); p.add_argument('--smoke',action='store_true')
    p.add_argument('--rank',type=int,default=0); p.add_argument('--workers',type=int,default=1)
    args=p.parse_args()
    if args.prepare: prepare(args.output)
    else: run(args)


if __name__=='__main__': main()
