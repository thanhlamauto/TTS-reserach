import importlib.metadata
import inspect
import os
import platform
import socket
import time
from pathlib import Path
from .common import atomic_json, source_hashes


def versions():
    result = {}
    for name in ('torch','torch_xla','diffusers','transformers','image-reward','numpy','Pillow','huggingface-hub'):
        try: result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name] = None
    return result


class Runtime:
    def __init__(self, output, rank=0):
        import torch
        import torch_xla
        import torch_xla.runtime as xr
        import torch_xla.core.xla_model as xm
        self.output = Path(output); self.output.mkdir(parents=True,exist_ok=True)
        self.rank = rank
        cache = self.output/'xla_cache'/f'worker{rank}'
        cache.mkdir(parents=True,exist_ok=True)
        xr.initialize_cache(str(cache),readonly=False)
        self.device = xm.xla_device()
        if xr.device_type() != 'TPU':
            raise RuntimeError(f'TPU required; got {xr.device_type()}')
        torch.set_num_threads(int(os.environ.get('SINKHORN_CPU_THREADS','8')))
        self.xm = xm
        self.info = dict(versions=versions(),hostname=socket.gethostname(),python=platform.python_version(),
                         device=str(self.device),device_type=xr.device_type(),
                         addressable_devices=xr.addressable_runtime_device_count(),
                         global_devices=xr.global_runtime_device_count(),rank=rank,
                         visible_chips=os.environ.get('TPU_VISIBLE_CHIPS'),
                         torch_xla_launch_available=hasattr(torch_xla,'launch'),
                         code_sha256=source_hashes(),start_unix=time.time())
        atomic_json(self.output/'environment'/f'worker{rank}.json',self.info)
        print(self.info,flush=True)

    def sync(self):
        self.xm.mark_step(); self.xm.wait_device_ops()

    def metrics(self):
        import torch_xla.debug.metrics as m
        return m.metrics_report()


def launch(fn, args=()):
    """No runtime initialization in parent; XLA 2.4 uses documented xmp.spawn."""
    import torch_xla
    if hasattr(torch_xla,'launch'):
        torch_xla.launch(fn,args=args)
    else:
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(fn,args=args,nprocs=None,start_method='spawn')


if __name__ == '__main__':
    import argparse
    import torch
    p = argparse.ArgumentParser(); p.add_argument('--output',required=True)
    a = p.parse_args(); rt = Runtime(a.output)
    x = torch.ones(4,device=rt.device); rt.sync()
    print('TPU_SUM',x.sum().cpu().item(),flush=True)
