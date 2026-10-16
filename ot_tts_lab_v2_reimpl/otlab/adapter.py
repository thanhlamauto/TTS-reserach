"""A thin adapter over the existing frozen SD1.5 XLA Sampler.

No Diffusers/Transformers/XLA imports occur until the real factory is requested.
The working model loader, coefficients, BF16 states and VAE are not replaced.
"""
from collections import Counter
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .io import digest, file_sha


def factory(path, config):
    module, attribute = path.split(":", 1)
    return getattr(importlib.import_module(module), attribute)(config)


def schedule_identity(sampler):
    return digest(dict(timesteps=sampler.timesteps,
                       alphas=sampler.scheduler.alphas_cumprod.tolist(),
                       final_alpha=float(sampler.scheduler.final_alpha_cumprod),
                       scheduler=dict(sampler.scheduler.config), eta=sampler.cfg["ddim_eta"],
                       prediction_type=sampler.scheduler.config.prediction_type))


class FrozenXlaAdapter:
    def __init__(self, sampler, reward, prompt, update, identity):
        if sampler.cfg["steps"] != 100 or sampler.cfg["ddim_eta"] != 1 or sampler.cfg["cfg_scale"] != 7.5:
            raise ValueError("fixed first pass requires DDIM100 eta1 CFG7.5")
        self.sampler, self.reward, self.prompt, self.update = sampler, reward, prompt, update
        self.schedule_sha = schedule_identity(sampler)
        self.base_identity = {**identity, "schedule_sha": self.schedule_sha}
        self.set_prompt(prompt)
        self.counters = Counter()

    def set_prompt(self, prompt):
        self.prompt = prompt
        self.identity = {**self.base_identity, "prompt_sha256": digest(prompt)}

    @torch.no_grad()
    def _chunks(self, xt, index):
        s = self.sampler
        embed = s.embedding(self.prompt)
        for offset in range(0, len(xt), s.batch):
            chunk = xt[offset:offset + s.batch]
            count = len(chunk)
            if count < s.batch:
                chunk = torch.cat([chunk, chunk[-1:].repeat(s.batch - count, 1, 1, 1)])
            x = chunk.to(device=s.rt.device, dtype=torch.bfloat16)
            timestep = torch.tensor(s.timesteps[index], dtype=torch.int64, device=s.rt.device)
            pred = s.pipe.unet(torch.cat([x, x]), timestep, encoder_hidden_states=embed, return_dict=False)[0]
            uncond, cond = pred.chunk(2)
            eps = uncond.float() + s.cfg["cfg_scale"] * (cond.float() - uncond.float())
            yield offset, count, x, eps, s.coefficients[index].to(s.rt.device)

    @torch.no_grad()
    def advance(self, xt, start, end, source_ids, noise_tape):
        if not 0 <= start <= end <= 100 or len(xt) != len(source_ids):
            raise ValueError("invalid original-schedule interval")
        if noise_tape.schedule_sha != self.schedule_sha:
            raise ValueError("noise/scheduler identity mismatch")
        if xt.dtype != torch.bfloat16:
            raise ValueError("frozen sampler state must be BF16")
        state = xt.clone()
        for index in range(start, end):
            pieces = []
            noise = noise_tape.at(index, source_ids)
            for offset, count, x, eps, coeff in self._chunks(state, index):
                n = noise[offset:offset + count]
                if count < self.sampler.batch:
                    n = torch.cat([n, n[-1:].repeat(self.sampler.batch - count, 1, 1, 1)])
                y = self.update(x, eps, coeff, n.to(x.device)).to(torch.bfloat16)
                self.sampler.rt.xm.mark_step()
                pieces.append(y[:count].cpu())
                self.counters["denoise_sample_steps"] += count
                self.counters["denoise_padding_sample_steps"] += self.sampler.batch - count
                self.counters["unet_cfg_branch_sample_evaluations"] += 2 * self.sampler.batch
            state = torch.cat(pieces)
        self.sampler.rt.sync()
        return state

    @torch.no_grad()
    def predict_clean(self, xt, index):
        if not 0 <= index < 100:
            raise ValueError("cannot predict clean beyond original schedule")
        pieces = []
        for _, count, x, eps, coeff in self._chunks(xt, index):
            clean = (x.float() - coeff[1] * eps) / coeff[0]
            self.sampler.rt.xm.mark_step()
            pieces.append(clean[:count].cpu())
            self.counters["aux_clean_sample_evaluations"] += count
            self.counters["aux_padding_sample_evaluations"] += self.sampler.batch - count
            self.counters["unet_cfg_branch_sample_evaluations"] += 2 * self.sampler.batch
        self.sampler.rt.sync()
        return torch.cat(pieces).float()

    @torch.no_grad()
    def score(self, clean, tag):
        if tag not in {"proxy", "terminal", "replay"}:
            raise ValueError("invalid verifier tag")
        images = self.sampler.decode(clean.to(torch.bfloat16))
        values = self.reward.score(self.prompt, images)
        rgb = torch.from_numpy(np.stack([np.asarray(im).copy() for im in images])).permute(0, 3, 1, 2)
        self.counters[tag + "_verifier_calls"] += 1
        self.counters[tag + "_verifier_scores"] += len(images)
        self.counters["vae_terminal_decode_samples"] += len(images)
        return torch.tensor(values, dtype=torch.float32), rgb


def frozen_xla_factory(config):
    # Existing helper modules live outside this newly reconstructed package.
    # Their exact source hashes are required, so a changed external dependency
    # cannot silently hit the same replay/cache identity.
    if not config.get("legacy_source_hashes"):
        raise ValueError("pin the existing sampler source hashes before real execution")
    for path, expected in config["legacy_source_hashes"].items():
        if file_sha(path) != expected:
            raise ValueError(f"legacy source changed: {path}")
    for path in reversed(config["legacy_pythonpath"]):
        sys.path.insert(0, path)
    from sd15_sinkhorn_tpu.sd15_sampler import Sampler, ddim_update
    from sd15_sinkhorn_tpu.tpu_runtime import Runtime
    from sd15_sinkhorn_tpu.reward import ImageRewardCPU
    rt = Runtime(config["runtime_output"], rank=0)
    sampler = Sampler(rt, config["sampler"], config["dcs_root"])
    reward = ImageRewardCPU(config["dcs_root"])
    identity = dict(backend="frozen_sd15_xla", legacy_source_hashes=config["legacy_source_hashes"],
                    model_id=config["sampler"]["model"], model_revision=config["sampler"]["revision"],
                    state_dtype="torch.bfloat16", package_provenance="independent_reimplementation")
    return FrozenXlaAdapter(sampler, reward, config["prompt"], ddim_update, identity)
