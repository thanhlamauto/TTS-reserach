"""Terminal-only feature evaluation in a fresh process after steering ends."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import yaml

from .features import TensorFeatures
from .io import atomic_json, load_tensor


def diversity(features):
    f = torch.nn.functional.normalize(features.double(), dim=1)
    mask = ~torch.eye(len(f), dtype=torch.bool)
    return float((1 - f @ f.T)[mask].mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--bank', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(cfg.get('cpu_threads', 4))
    encoder = TensorFeatures(cfg)
    root, bank = Path(args.root), Path(args.bank)
    bankrecord = json.loads((bank / 'bank.json').read_text())
    images = [torch.from_numpy(np.asarray(Image.open(f).convert('RGB')).copy()).permute(2, 0, 1)
              for f in sorted((bank / 'reference/images').glob('*.png'))]
    if len(images) != 16:
        raise RuntimeError('terminal bank images missing')
    with torch.no_grad():
        bf = torch.cat([encoder.encode_rgb(im[None]) for im in images])
        rows = []
        for line in (root / 'rows.jsonl').read_text().splitlines():
            r = json.loads(line)
            d = load_tensor(root / r['job_id'] / 'terminal.pt')
            sf = torch.cat([encoder.encode_rgb(im[None]) for im in d['rgb']])
            rewards = torch.cat([torch.tensor(bankrecord['sets']['reference']['rewards']), d['rewards']])
            top = torch.argsort(rewards, descending=True, stable=True)[:8]
            rows.append(dict(method=r['method'], source_dino_diversity=diversity(sf),
                             pool_top8_dino_diversity=diversity(torch.cat([bf, sf])[top]),
                             source_terminal_features=sf.tolist(), pool_top8_indices=top.tolist()))
    atomic_json(args.out, dict(encoder_identity=encoder.identity, rows=rows,
                              independent_quality_metric=False, no_feedback_to_steering=True))


if __name__ == '__main__':
    main()
