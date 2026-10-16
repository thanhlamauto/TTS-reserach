"""Paired checkpoint runner. P1 controls share proposals and scored previews."""
from collections import Counter
import itertools
import json
from pathlib import Path
import time

import torch

from .analysis import prompt_bootstrap, terminal_metrics
from .io import atomic_json, digest, file_sha, load_tensor, tensor_sha, validate_checkpoint
from .noise import NoiseTape


def code_identity():
    return digest({p.name: file_sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def smoke_configs():
    return [dict(name="none", method="none"),
            *[dict(name="p1_" + s, method="p1", selector=s, penalty=.1 if s != "independent" else 0.)
              for s in ["ot", "independent", "greedy"]],
            dict(name="p2", method="p2", loss="sinkhorn"),
            dict(name="feature_mean", method="p2", loss="mean"),
            dict(name="p2_shuffle", method="p2", loss="sinkhorn", shuffle=True),
            dict(name="p3", method="p3", loss="sliced", slices=32),
            dict(name="p3_shuffle", method="p3", loss="sliced", slices=32, shuffle=True)]


def make_plan(manifest, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = Path(manifest).resolve()
    inputs, seen = [], set()
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        for key in ["checkpoint", "noise", "trusted_none"]:
            if key in row:
                p = Path(row[key])
                p = p if p.is_absolute() else manifest.parent / p
                row[key] = str(p.resolve())
                row[key + "_sha256"] = file_sha(p)
        cp = validate_checkpoint(load_tensor(row["checkpoint"]))
        if len(cp["source_xt"]) != 8 or len(cp["bank_clean"]) != 16:
            raise ValueError("fixed first pass requires source8/reference16")
        if cp["meta"]["model_id"] != "toy":
            if (cp["meta"]["model_id"] != "runwayml/stable-diffusion-v1-5" or
                    cp["meta"]["model_revision"] != "451f4fe16113bff5a5d2269ed5ad43b0592e9a14" or
                    tuple(cp["source_xt"].shape[1:]) != (4, 64, 64)):
                raise ValueError("fixed first pass supports the pinned SD1.5 VP states only")
        tape = NoiseTape.from_bundle(load_tensor(row["noise"]))
        if cp["meta"]["next_index"] != 50 or tape.start_index != 50:
            raise ValueError("fixed first pass requires actual checkpoint/tape at50")
        if tape.schedule_sha != cp["meta"]["schedule_sha"] or tape.source_ids != tuple(range(len(cp["source_xt"]))):
            raise ValueError("checkpoint/tape schedule or source identity mismatch")
        for key in ["prompt_id", "bank_id", "prefix_id"]:
            if row[key] != cp["meta"][key]:
                raise ValueError("manifest identity mismatch")
        identity = tuple(row[k] for k in ["prompt_id", "bank_id", "prefix_id", "replicate"])
        if identity in seen:
            raise ValueError("duplicate manifest conditional unit")
        seen.add(identity)
        inputs.append(row)
    jobs = [{"input": row, "config": config} for row in inputs for config in smoke_configs()]
    for job in jobs:
        job["job_id"] = digest(job)
    plan = dict(schema="otlab.reimplementation.plan.v1", code_sha256=code_identity(), jobs=jobs,
                source="independent reconstruction of supplied prose; missing original package")
    plan["plan_sha256"] = digest(plan)
    atomic_json(output / "plan.json", plan)
    (output / "jobs.jsonl").write_text("".join(json.dumps(j, sort_keys=True) + "\n" for j in jobs))
    return plan


def checked_inputs(row):
    for key in ["checkpoint", "noise", "trusted_none"]:
        if key in row and file_sha(row[key]) != row[key + "_sha256"]:
            raise ValueError(f"frozen input changed: {key}")
    return validate_checkpoint(load_tensor(row["checkpoint"])), NoiseTape.from_bundle(load_tensor(row["noise"]))


def verify_plan(plan):
    copied = dict(plan)
    expected = copied.pop("plan_sha256")
    if digest(copied) != expected or plan["code_sha256"] != code_identity():
        raise ValueError("plan/code changed; create a new plan namespace")


def replay(plan, adapter_for, output):
    verify_plan(plan)
    records, seen = [], set()
    for job in plan["jobs"]:
        row = job["input"]
        key = digest(row)
        if key in seen:
            continue
        seen.add(key)
        cp, tape = checked_inputs(row)
        adapter = adapter_for(cp)
        if adapter.schedule_sha != tape.schedule_sha:
            raise ValueError("adapter schedule mismatch")
        if adapter.identity.get("backend") == "frozen_sd15_xla":
            for k in ["model_id", "model_revision"]:
                if cp["meta"][k] != adapter.identity[k]:
                    raise ValueError("checkpoint/adapter model identity mismatch")
        ids = tuple(range(len(cp["source_xt"])))
        first = adapter.advance(cp["source_xt"], 50, 100, ids, tape)
        second = adapter.advance(cp["source_xt"], 50, 100, ids, tape)
        r1, im1 = adapter.score(first, "replay")
        r2, im2 = adapter.score(second, "replay")
        duplicate = torch.equal(first, second) and torch.equal(im1, im2) and torch.equal(r1, r2)
        trusted = load_tensor(row["trusted_none"])["latent"] if "trusted_none" in row else None
        trusted_pass = trusted is not None and torch.equal(first, trusted)
        trusted_pixels_pass = False
        if trusted is not None:
            rt, it = adapter.score(trusted, "replay")
            trusted_pixels_pass = torch.equal(r1, rt) and torch.equal(im1, it)
        record = dict(input_sha256=key, duplicate_bitwise=duplicate, trusted_latent_bitwise=trusted_pass,
                      trusted_pixels_reward_exact=trusted_pixels_pass, endpoint_sha256=tensor_sha(first),
                      adapter_identity=adapter.identity,
                      passed=duplicate and trusted_pass and trusted_pixels_pass)
        records.append(record)
        if not record["passed"]:
            atomic_json(output, dict(passed=False, records=records, plan_sha256=plan["plan_sha256"]))
            raise RuntimeError("replay gate failed; no method run authorized")
    proof = dict(passed=True, plan_sha256=plan["plan_sha256"], code_sha256=code_identity(), records=records)
    atomic_json(output, proof)
    return proof


class Cache:
    def __init__(self, root, execution):
        self.root, self.execution = Path(root), execution
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, stage, key, compute):
        identity = digest(dict(stage=stage, key=key, execution=self.execution, code=code_identity()))
        path, receipt = self.root / (identity + ".pt"), self.root / (identity + ".json")
        if receipt.exists():
            if not path.exists() or json.loads(receipt.read_text())["sha256"] != file_sha(path):
                raise ValueError("cached artifact corrupted")
            return load_tensor(path), True
        value = compute()
        tmp = path.with_suffix(".tmp")
        torch.save(value, tmp)
        tmp.replace(path)
        atomic_json(receipt, dict(sha256=file_sha(path), identity=identity, stage=stage))
        return value, False


def counters_diff(now, before):
    return {k: v - before.get(k, 0) for k, v in now.items() if v - before.get(k, 0)}


def cold_accounting(method, n=8, active_proposals=None, initial_aux=None, physical_batch=None):
    if method not in {"none", "p1", "p2", "p3"}:
        raise ValueError("unknown method for cost accounting")
    active = n if active_proposals is None else active_proposals
    if active < n:
        raise ValueError("each source requires an active no-op")
    extra = (active - n) * 4 + active if method == "p1" else 0
    lower = 2400 + extra
    aux = 0 if method == "none" else initial_aux
    exact = lower + aux if aux is not None else None
    padding = None
    if physical_batch is not None:
        if 8 % physical_batch or 16 % physical_batch:
            raise ValueError("base recipe padding must be measured explicitly")
        padding = ((-active) % physical_batch) * 5 if method == "p1" else 0
    return dict(cold_nfe=exact, cold_nfe_lower_bound=lower, method_initial_aux_nfe=aux,
                cold_padding_nfe=padding,
                cold_nfe_with_padding=(exact + padding) if exact is not None and padding is not None else None)


def run_plan(plan, proof, adapter_for, worker, output, cache_dir, execution_identity, evaluate_rgb=None):
    verify_plan(plan)
    if not proof["passed"] or proof["plan_sha256"] != plan["plan_sha256"] or proof["code_sha256"] != code_identity():
        raise ValueError("absent/stale replay proof")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "plan.json", plan)
    cache = Cache(cache_dir, execution_identity)
    results, baselines = [], {}
    for job in plan["jobs"]:
        begin = time.perf_counter()
        row, cfg = job["input"], job["config"]
        cp, tape = checked_inputs(row)
        adapter = adapter_for(cp)
        matching = [r for r in proof["records"] if r["input_sha256"] == digest(row)]
        if not matching or matching[0]["adapter_identity"] != adapter.identity:
            raise ValueError("execution adapter differs from replay gate")
        before = dict(adapter.counters)
        worker_before = dict(worker.counters)
        n, shape = len(cp["source_xt"]), cp["source_xt"].shape[1:]
        ids = tuple(range(n))
        conditional_key = digest(row)
        common = dict(checkpoint=row["checkpoint_sha256"], noise=row["noise_sha256"],
                      prompt=cp["meta"]["prompt"], adapter=adapter.identity)
        bound = cfg.get("bound", .025)
        logs, hits = {}, {}
        entry, start = cp["source_xt"], 50
        if cfg["method"] == "p1":
            proposal_key = dict(checkpoint=row["checkpoint_sha256"], bound=bound, factor=4)
            proposed, hits["propose"] = cache.get("p1-propose", proposal_key,
                lambda: worker.call("p1-propose", cp, dict(bound=bound, factor=4)))

            def preview_compute():
                active = proposed["active"]
                source_ids = torch.arange(n)[:, None].expand(n, 3)[active].tolist()
                selected = proposed["entry_states"][active]
                future = adapter.advance(selected, 50, 54, source_ids, tape)
                clean = adapter.predict_clean(future, 54)
                reward, rgb = adapter.score(clean, "proxy")
                states = torch.zeros(n, 3, *shape, dtype=future.dtype)
                images = torch.zeros(n, 3, *rgb.shape[1:], dtype=rgb.dtype)
                scores = torch.zeros(n, 3)
                states[active], images[active], scores[active] = future, rgb, reward
                return dict(states=states, rgb=images, proxy_reward=scores, active=active, next_index=54)

            preview, hits["preview"] = cache.get("p1-preview", dict(**common, proposal=proposal_key), preview_compute)
            selection = worker.call("p1-select", preview, dict(selector=cfg["selector"], penalty=cfg["penalty"]))
            entry, start = selection["entry_states"], 54
            logs = {"selection": {k: v for k, v in selection.items() if k != "entry_states"},
                    "active_proposals": int(preview["active"].sum()), "trust": proposed["trust"],
                    "proposal_worker_counters": proposed.get("worker_counters")}
        elif cfg["method"] in {"p2", "p3"}:
            prox_config = {k: cfg[k] for k in ["loss", "slices", "shuffle"] if k in cfg}
            prox_config.update(bound=bound, factor=4, steps=2)
            edited, hits["proximal"] = cache.get("proximal", dict(checkpoint=row["checkpoint_sha256"], config=prox_config),
                                                 lambda: worker.call("proximal", cp, prox_config))
            entry = edited["entry_states"]
            logs = {k: v for k, v in edited.items() if k != "entry_states"}

        def suffix_compute():
            final = adapter.advance(entry, start, 100, ids, tape)
            rewards, rgb = adapter.score(final, "terminal")
            return dict(latent=final, rewards=rewards, rgb=rgb)

        terminal, hits["suffix"] = cache.get("suffix", dict(**common, entry=tensor_sha(entry), start=start), suffix_compute)
        if cfg["method"] == "none":
            baselines[conditional_key] = terminal["rewards"]
        if conditional_key not in baselines:
            raise ValueError("none must run before each paired comparison")
        features = evaluate_rgb(terminal["rgb"]) if evaluate_rgb else None
        metrics = terminal_metrics(terminal["rewards"], cp["bank_reward"], features, baselines[conditional_key])
        aux = cp["meta"].get("cold_initial_aux_nfe")
        active_count = logs.get("active_proposals", n)
        physical_batch = adapter.sampler.batch if hasattr(adapter, "sampler") else None
        cold = cold_accounting(cfg["method"], n, active_count, aux, physical_batch)
        record = dict(job_id=job["job_id"], method=cfg["name"], config=cfg,
                      **{k: row[k] for k in ["prompt_id", "bank_id", "prefix_id", "replicate"]},
                      **metrics, measured_counters=counters_diff(adapter.counters, before),
                      measured_worker_counters=counters_diff(worker.counters, worker_before), **cold,
                      cold_nfe_excludes_padding=True, inherited_aux_nfe=aux,
                      feature_compute_included_in_nfe=False, cache_hits=hits,
                      wall_seconds=time.perf_counter() - begin, logs=logs,
                      endpoint_sha256=tensor_sha(terminal["latent"]),
                      state_dtype=str(entry.dtype), package_provenance="independent_reimplementation")
        branch = output / job["job_id"]
        branch.mkdir()
        torch.save(terminal, branch / "terminal.pt")
        atomic_json(branch / "result.json", record)
        results.append(record)
        with (output / "rows.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        atomic_json(output / "status.json", dict(state="running", complete=len(results), total=len(plan["jobs"])))
    names = [c["name"] for c in smoke_configs()]
    contrasts = []
    for method, control in itertools.combinations(names, 2):
        for field in ["source_mean", "pool_best", "pool_top8_mean"]:
            contrasts.append(dict(method=method, control=control, field=field,
                                  **prompt_bootstrap(results, method, control, field)))
    atomic_json(output / "paired_contrasts.json", contrasts)
    atomic_json(output / "status.json", dict(state="complete", complete=len(results), total=len(plan["jobs"])))
    return results
