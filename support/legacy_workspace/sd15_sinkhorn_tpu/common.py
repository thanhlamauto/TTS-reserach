import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def load_config(path=None):
    import yaml
    path = Path(path) if path else Path(__file__).with_name('config.yaml')
    cfg = yaml.safe_load(path.read_text())
    return cfg


def initial_noise(seed, prompt_index, stream, n):
    # The source stream preserves the existing SeedSequence([trial_seed,prompt_index]).
    key = [seed,prompt_index] if stream == 0 else [seed,prompt_index,stream]
    pseed = int(np.random.SeedSequence(key).generate_state(1,dtype=np.uint32)[0])
    gen = torch.Generator(device='cpu').manual_seed(pseed)
    # Draw one at a time: values stay invariant when N or microbatch size changes.
    noise = torch.stack([torch.randn((4,64,64),generator=gen) for _ in range(n)])
    return noise.to(torch.bfloat16), pseed


def event_seed(seed, prompt_index, checkpoint):
    return int(np.random.SeedSequence([seed,prompt_index,checkpoint,7341]).generate_state(1)[0])


def source_hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py'))}


def read_prompts(split):
    return [s.strip() for s in Path(__file__).with_name(f'prompts_{split}.txt').read_text().splitlines()
            if s.strip() and not s.startswith('#')]
