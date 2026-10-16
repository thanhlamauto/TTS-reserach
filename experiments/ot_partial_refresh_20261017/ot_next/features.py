"""RGB-only DINO evaluation worker using the exact prior worker environment."""
import argparse
import contextlib
import json
import subprocess
import sys
from pathlib import Path
import time

import torch
import yaml
from otlab.io import load_tensor


class RGBClient:
    def __init__(self, python, config, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log = (self.directory / 'stderr.log').open('a')
        self.process = subprocess.Popen([python, '-u', '-m', 'ot_next.features', '--config', str(config)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True)
        ready = self.process.stdout.readline()
        if not ready or not json.loads(ready).get('ready'):
            self.close()
            raise RuntimeError('RGB worker initialization failed')
        self.identity = json.loads(ready)['identity']
        self.calls = 0
        self.count = 0
        self.seconds = 0.

    def encode(self, rgb, label):
        self.calls += 1
        self.count += len(rgb)
        request = self.directory / f'{self.calls:03d}_{label}.input.pt'
        output = request.with_name(request.name.replace('.input.', '.output.'))
        if request.exists() or output.exists():
            raise RuntimeError('new attempt required: worker request namespace already exists')
        torch.save({'rgb': rgb}, request)
        start = time.perf_counter()
        self.process.stdin.write(json.dumps(dict(input=str(request), output=str(output))) + '\n')
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        self.seconds += time.perf_counter() - start
        if not response or not json.loads(response).get('ok'):
            raise RuntimeError(f'RGB worker failed: {response}; inspect {self.log.name}')
        return load_tensor(output)['features']

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.log.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(cfg.get('cpu_threads', 4))
    with contextlib.redirect_stdout(sys.stderr):
        from otlab.features import TensorFeatures
        encoder = TensorFeatures(cfg)
    print(json.dumps(dict(ready=True, identity=encoder.identity)), flush=True)
    for line in sys.stdin:
        try:
            req = json.loads(line)
            data = load_tensor(req['input'])
            if set(data) != {'rgb'} or data['rgb'].dtype != torch.uint8 or data['rgb'].ndim != 4:
                raise ValueError('closed RGB-only interface; no rewards, bank or source outcomes')
            start = time.perf_counter()
            with torch.no_grad(), contextlib.redirect_stdout(sys.stderr):
                features = torch.cat([encoder.encode_rgb(im[None]) for im in data['rgb']])
            torch.save(dict(features=features, seconds=time.perf_counter()-start), req['output'])
            print(json.dumps(dict(ok=True)), flush=True)
        except Exception as exc:
            print(json.dumps(dict(ok=False, error=str(exc))), flush=True)


if __name__ == '__main__':
    main()
