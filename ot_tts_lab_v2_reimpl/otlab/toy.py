"""Synthetic integration fixture. This is never a substitute for SD1.5 evidence."""
from collections import Counter

import torch
import torch.nn.functional as F

from .io import digest


def pixels(x):
    rgb = torch.cat([x[:, :2], (x[:, :1] - x[:, 1:2])], dim=1)
    return torch.sigmoid(rgb)


class ToyEncoder:
    def __init__(self):
        self.counters = Counter()

    def encode_clean(self, x):
        return self.encode_pixels(pixels(x))

    def encode_rgb(self, x):
        return self.encode_pixels(x.float() / 255)

    def encode_pixels(self, x):
        self.counters["feature_forward_samples"] += len(x)
        pooled = F.adaptive_avg_pool2d(x, (2, 2)).flatten(1)
        return F.normalize(torch.cat([pooled, pooled.square()], dim=1), dim=-1)


def trusted_loop(initial, start, end, source_ids, tape):
    # Independent driver used to provide the trusted synthetic endpoint.
    result = initial.clone()
    for k in range(start, end):
        prediction = .02 * torch.tanh(result.float())
        result = (.995 * result.float() - prediction + .01 * tape.at(k, source_ids)).to(initial.dtype)
    return result


class ToyAdapter:
    def __init__(self, prompt_id):
        self.schedule_sha = digest({"toy_schedule": 100, "eta": 1})
        self.identity = dict(backend="synthetic_only", version=1, prompt_id=prompt_id,
                             schedule_sha=self.schedule_sha)
        self.counters = Counter()

    def advance(self, xt, start, end, source_ids, noise_tape):
        if noise_tape.schedule_sha != self.schedule_sha:
            raise ValueError("toy schedule mismatch")
        x = xt.clone()
        for k in range(start, end):
            x = (.995 * x.float() - .02 * torch.tanh(x.float()) + .01 * noise_tape.at(k, source_ids)).to(xt.dtype)
            self.counters["denoise_sample_steps"] += len(x)
        return x

    def predict_clean(self, xt, index):
        self.counters["aux_clean_sample_evaluations"] += len(xt)
        return xt.float() / (.8 + .002 * index)

    def score(self, clean, tag):
        reward = -(clean.float() - .25).square().flatten(1).mean(1)
        rgb = (pixels(clean.float()) * 255).round().to(torch.uint8)
        self.counters[tag + "_verifier_scores"] += len(clean)
        self.counters[tag + "_verifier_calls"] += 1
        return reward, rgb
