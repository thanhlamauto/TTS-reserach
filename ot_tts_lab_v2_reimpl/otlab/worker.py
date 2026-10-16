"""Long-lived encoder worker with a closed, source-outcome-free file API."""
import argparse
import contextlib
import json
import subprocess
import sys
from pathlib import Path
from collections import Counter

import torch
import yaml

from .io import load_tensor, validate_checkpoint
from .p1 import propose, select
from .proximal import proximal_edit


class AlgorithmWorker:
    def __init__(self, encoder, identity):
        self.encoder, self.identity = encoder, identity
        self.counters = encoder.counters

    def call(self, operation, bundle, config):
        if operation == "p1-select":
            if set(bundle) != {"states", "rgb", "proxy_reward", "active", "next_index"}:
                raise ValueError("invalid preview bundle")
            if bundle["next_index"] != 54:
                raise ValueError("fixed first-pass preview must be at index54")
            active = bundle["active"]
            # One encoder/VAE worker, one feature microbatch at a time.
            f = torch.cat([self.encoder.encode_rgb(im[None]) for im in bundle["rgb"][active]])
            features = torch.zeros(*active.shape, f.shape[-1], dtype=f.dtype)
            features[active] = f
            result = select(bundle["proxy_reward"], features, active,
                            config.get("selector", "ot"), config.get("penalty", .1))
            ids = torch.tensor(result["choices"])
            result["entry_states"] = bundle["states"][torch.arange(len(ids)), ids]
            return result
        cp = validate_checkpoint(bundle)
        if cp["meta"]["vae_id"] != self.identity["vae_id"]:
            raise ValueError("surrogate VAE identity mismatch")
        if operation == "preflight":
            from .features import gradient_preflight
            return gradient_preflight(self.encoder, cp["source_clean"])
        if operation == "p1-propose":
            with torch.no_grad():
                sf = torch.cat([self.encoder.encode_clean(x[None]) for x in cp["source_clean"]])
                bf = torch.cat([self.encoder.encode_clean(x[None]) for x in cp["bank_clean"]])
            return propose(cp, sf, bf, config.get("bound", .025), config.get("factor", 4))
        if operation == "proximal":
            return proximal_edit(cp, self.encoder.encode_clean, **config)
        if operation == "terminal-features":
            raise ValueError("terminal outcomes are not accepted by the steering API")
        raise ValueError("unknown worker operation")


class WorkerClient:
    def __init__(self, python, config, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.log = (self.directory / "worker.stderr.log").open("a")
        self.process = subprocess.Popen([python, "-m", "otlab.worker", "--config", str(config), "--serve"],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True)
        self.index = 0
        self.counters = Counter()
        ready = self.process.stdout.readline()
        if not ready or not json.loads(ready).get("ready"):
            self.close()
            raise RuntimeError(f"worker initialization failed; inspect {self.log.name}")
        self.identity = json.loads(ready)["identity"]

    def call(self, operation, bundle, config):
        self.index += 1
        input_path = self.directory / f"{self.index:06d}.input.pt"
        output_path = self.directory / f"{self.index:06d}.output.pt"
        torch.save(bundle, input_path)
        request = dict(op=operation, input=str(input_path.resolve()), output=str(output_path.resolve()), config=config)
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        response_line = self.process.stdout.readline()
        if not response_line:
            raise RuntimeError(f"encoder worker exited; inspect {self.log.name}")
        response = json.loads(response_line)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "worker failure"))
        result = load_tensor(output_path)
        self.counters.update(result.get("worker_counters", {}))
        return result

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
    p.add_argument("--config", required=True)
    p.add_argument("--request")
    p.add_argument("--serve", action="store_true")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    with contextlib.redirect_stdout(sys.stderr):
        if cfg.get("mode") == "synthetic_test_only":
            from .toy import ToyEncoder
            encoder = ToyEncoder()
            encoder.identity = dict(mode="synthetic_test_only", torch=torch.__version__)
        else:
            from .features import TensorFeatures
            encoder = TensorFeatures(cfg)
        worker = AlgorithmWorker(encoder, dict(vae_id=cfg["vae"]["identity"]))
    if args.serve:
        print(json.dumps(dict(ready=True, identity=encoder.identity)), flush=True)
    lines = sys.stdin if args.serve else [Path(args.request).read_text()]
    for line in lines:
        try:
            req = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                before = dict(encoder.counters)
                result = worker.call(req["op"], load_tensor(req["input"]), req.get("config", {}))
                result["worker_counters"] = {k: v - before.get(k, 0) for k, v in encoder.counters.items()}
                Path(req["output"]).parent.mkdir(parents=True, exist_ok=True)
                torch.save(result, req["output"])
            print(json.dumps(dict(ok=True, output=req["output"])), flush=True)
        except Exception as exc:
            print(json.dumps(dict(ok=False, error=f"{type(exc).__name__}: {exc}")), flush=True)
            if not args.serve:
                raise


if __name__ == "__main__":
    main()
