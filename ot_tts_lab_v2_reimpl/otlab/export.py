"""Export current-state clean predictions from a verified historical bank.

This file never relabels a 30-step bank as a 100-step bank. Source terminal
outcomes are retained only by the controller for a trusted replay comparison.
"""
import json
from pathlib import Path

import torch

from .io import atomic_json, digest, file_sha, load_tensor, validate_checkpoint
from .noise import NoiseTape


def historical_tape(sampler, noise_seed, n, start=50):
    if n % sampler.batch:
        raise ValueError("historical tape export requires complete original microbatches")
    all_steps = []
    for index in range(start, 100):
        chunks = []
        for offset in range(0, n, sampler.batch):
            generator = torch.Generator().manual_seed(noise_seed + index * 100003 + offset)
            chunks.append(torch.randn(sampler.batch, 4, 64, 64, generator=generator, dtype=torch.float32))
        all_steps.append(torch.cat(chunks))
    from .adapter import schedule_identity
    return NoiseTape(torch.stack(all_steps), start, tuple(range(n)), schedule_identity(sampler))


def export_bank(bank_dir, adapter, output):
    bank_dir, output = Path(bank_dir), Path(output)
    bank = json.loads((bank_dir / "bank.json").read_text())
    cfg = bank["sampler"]
    if cfg["steps"] != 100 or cfg["ddim_eta"] != 1 or cfg["cfg_scale"] != 7.5:
        raise ValueError("existing bank is not the requested DDIM100 eta1 CFG7.5 recipe")
    if bank["n_reference"] != 16 or bank["n_source"] != 8 or bank["timesteps"] != adapter.sampler.timesteps:
        raise ValueError("bank counts or original schedule mismatch")
    if bank["prompt"] != adapter.prompt:
        raise ValueError("bank prompt mismatch")
    if cfg["model"] != adapter.identity["model_id"] or cfg["revision"] != adapter.identity["model_revision"]:
        raise ValueError("bank model identity mismatch")
    source = load_tensor(bank_dir / "source/trajectories.pt")
    reference = load_tensor(bank_dir / "reference/trajectories.pt")
    if source["identity"] != bank["identity"] or reference["identity"] != bank["identity"]:
        raise ValueError("bank tensor identity mismatch")
    if 50 not in source["checkpoints"] or 50 not in reference["checkpoints"]:
        raise ValueError("actual index50 state absent")
    sx, bx = source["checkpoints"][50], reference["checkpoints"][50]
    before = dict(adapter.counters)
    clean = adapter.predict_clean(sx, 50)
    bank_clean = adapter.predict_clean(bx, 50)
    meta = dict(prompt=bank["prompt"], prompt_id=bank["prompt_index"], bank_id=bank["identity"],
                prefix_id=digest({"bank": bank["identity"], "completed": 50}), next_index=50,
                clean_prediction_index=50, schedule_sha=adapter.schedule_sha,
                model_id=cfg["model"], model_revision=cfg["revision"],
                vae_id=f'{cfg["model"]}@{cfg["revision"]}/vae', state_dtype=str(sx.dtype),
                provenance="actual current-state UNet evaluation, original schedule index50",
                cold_initial_aux_nfe=24)
    cp = dict(schema="otlab.checkpoint.v2", meta=meta, source_xt=sx, source_clean=clean,
              bank_clean=bank_clean, bank_reward=torch.tensor(bank["sets"]["reference"]["rewards"], dtype=torch.float32),
              signal_scale=float(adapter.sampler.coefficients[50][0]))
    validate_checkpoint(cp)
    output.mkdir(parents=True, exist_ok=False)
    torch.save(cp, output / "checkpoint.pt")
    tape = historical_tape(adapter.sampler, bank["sets"]["source"]["seed"], len(sx))
    torch.save(tape.bundle(), output / "noise.pt")
    # Trusted old endpoint is controller-only; it never enters checkpoint.pt.
    torch.save({"latent": source["final"]}, output / "trusted_none.pt")
    row = dict(prompt_id=meta["prompt_id"], bank_id=meta["bank_id"], prefix_id=meta["prefix_id"],
               replicate=0, checkpoint=str((output / "checkpoint.pt").resolve()),
               noise=str((output / "noise.pt").resolve()),
               trusted_none=str((output / "trusted_none.pt").resolve()),
               inherited_bank_dir=str(bank_dir.resolve()))
    (output / "manifest.jsonl").write_text(json.dumps(row) + "\n")
    atomic_json(output / "export_audit.json", dict(
        source_hashes={str(p): file_sha(p) for p in [bank_dir / "bank.json", bank_dir / "source/trajectories.pt",
                                                   bank_dir / "reference/trajectories.pt"]},
        current_clean_index=50, inherited_base_nfe=2400,
        export_counters={k: v - before.get(k, 0) for k, v in adapter.counters.items()},
        source_terminal_outcomes_in_worker_input=False))
    return row
