import argparse
import json
from pathlib import Path

import torch
import yaml

from .adapter import factory
from .io import atomic_json, digest, load_tensor
from .runner import make_plan, replay, run_plan


def demo(output):
    from .noise import new_tape
    from .toy import ToyAdapter, ToyEncoder, trusted_loop
    from .worker import AlgorithmWorker
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    encoder = ToyEncoder()
    worker = AlgorithmWorker(encoder, dict(vae_id="toy"))
    rows, adapters = [], {}
    for prompt in range(2):
        adapter = adapters[prompt] = ToyAdapter(prompt)
        g = torch.Generator().manual_seed(100 + prompt)
        source = torch.randn(8, 2, 8, 8, generator=g).to(torch.bfloat16)
        bank = torch.randn(16, 2, 8, 8, generator=g)
        cp = dict(schema="otlab.checkpoint.v2", meta=dict(
            prompt=f"SYNTHETIC {prompt}", prompt_id=prompt, bank_id=f"bank{prompt}", prefix_id=f"prefix{prompt}",
            next_index=50, clean_prediction_index=50, schedule_sha=adapter.schedule_sha,
            model_id="toy", model_revision="toy-v1", vae_id="toy", state_dtype=str(source.dtype),
            provenance="SYNTHETIC integration fixture, no pretrained model", cold_initial_aux_nfe=24),
            source_xt=source, source_clean=adapter.predict_clean(source, 50), bank_clean=bank.float(),
            bank_reward=-(bank - .25).square().flatten(1).mean(1), signal_scale=.5)
        directory = root / f"input{prompt}"
        directory.mkdir()
        torch.save(cp, directory / "checkpoint.pt")
        for rep in range(2):
            tape = new_tape(20261016, prompt, rep, tuple(range(8)), (2, 8, 8), adapter.schedule_sha)
            torch.save(tape.bundle(), directory / f"noise{rep}.pt")
            endpoint = trusted_loop(source, 50, 100, range(8), tape)
            torch.save({"latent": endpoint}, directory / f"trusted{rep}.pt")
            rows.append(dict(prompt_id=prompt, bank_id=f"bank{prompt}", prefix_id=f"prefix{prompt}", replicate=rep,
                             checkpoint=str((directory / "checkpoint.pt").resolve()),
                             noise=str((directory / f"noise{rep}.pt").resolve()),
                             trusted_none=str((directory / f"trusted{rep}.pt").resolve())))
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    plan = make_plan(manifest, root / "plan")
    adapter_for = lambda cp: adapters[cp["meta"]["prompt_id"]]
    proof = replay(plan, adapter_for, root / "replay.json")
    result = run_plan(plan, proof, adapter_for, worker, root / "run", root / "cache",
                      execution_identity={"mode": "synthetic", "encoder": "toy-v1"}, evaluate_rgb=encoder.encode_rgb)
    summary = dict(passed=True, jobs=len(result), prompts=2, suffix_replicates=2, configurations=9,
                   evidence_kind="synthetic_only", pretrained_quality_claim=False,
                   implementation="independent_reimplementation", worker_counters=dict(encoder.counters))
    atomic_json(root / "summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--out", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--manifest", required=True)
    pl.add_argument("--out", required=True)
    for name in ["replay", "run"]:
        q = sub.add_parser(name)
        q.add_argument("--plan", required=True)
        q.add_argument("--adapter", required=True)
        q.add_argument("--adapter-config", required=True)
        q.add_argument("--out", required=True)
        if name == "run":
            q.add_argument("--worker-python", required=True)
            q.add_argument("--worker-config", required=True)
            q.add_argument("--replay-proof", required=True)
            q.add_argument("--cache", required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.command == "demo":
        print(json.dumps(demo(args.out)))
        return
    if args.command == "plan":
        plan = make_plan(args.manifest, args.out)
        print(json.dumps(dict(jobs=len(plan["jobs"]), plan_sha256=plan["plan_sha256"])))
        return
    plan = json.loads(Path(args.plan).read_text())
    cfg = yaml.safe_load(Path(args.adapter_config).read_text())
    adapters = {}
    shared = []

    def adapter_for(cp):
        key = digest(dict(prompt=cp["meta"]["prompt"], config=cfg))
        if shared and hasattr(shared[0], "set_prompt"):
            shared[0].set_prompt(cp["meta"]["prompt"])
            return shared[0]
        if key not in adapters:
            adapters[key] = factory(args.adapter, {**cfg, "prompt": cp["meta"]["prompt"]})
            shared.append(adapters[key])
        return adapters[key]

    if args.command == "replay":
        print(json.dumps(replay(plan, adapter_for, args.out)))
    else:
        from .worker import WorkerClient
        from .io import file_sha
        from .runner import verify_plan
        verify_plan(plan)
        proof = json.loads(Path(args.replay_proof).read_text())
        if not proof["passed"] or proof["plan_sha256"] != plan["plan_sha256"]:
            raise ValueError("replay proof missing/stale")
        client = WorkerClient(args.worker_python, args.worker_config,
                              Path(args.out).parent / (Path(args.out).name + ".worker_requests"))
        try:
            rows = run_plan(plan, proof, adapter_for, client, args.out, args.cache,
                            execution_identity={"worker_config_sha": file_sha(args.worker_config),
                                                "worker_python": args.worker_python,
                                                "worker_identity": client.identity, "adapter_config": cfg})
            print(json.dumps(dict(completed=len(rows))))
        finally:
            client.close()


if __name__ == "__main__":
    main()
