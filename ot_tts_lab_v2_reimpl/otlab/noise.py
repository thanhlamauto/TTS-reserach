"""Source-keyed FP32 DDIM tapes. New tapes define a new paired protocol."""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class NoiseTape:
    variance_noise: torch.Tensor
    start_index: int
    source_ids: tuple
    schedule_sha: str

    def __post_init__(self):
        x = self.variance_noise
        if x.dtype != torch.float32 or x.ndim != 5 or x.device.type != "cpu" or not torch.isfinite(x).all():
            raise ValueError("noise tape must be finite FP32 CPU [steps,N,C,H,W]")
        if not self.schedule_sha or len(self.source_ids) != x.shape[1] or len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("invalid schedule/source identities")
        if not 0 <= self.start_index < 100 or self.start_index + len(x) != 100:
            raise ValueError("tape must end at original step 100")
        self.lookup = {int(s): i for i, s in enumerate(self.source_ids)}

    def at(self, index, source_ids):
        if not self.start_index <= index < self.start_index + len(self.variance_noise):
            raise ValueError("noise index outside frozen tape")
        try:
            ids = [self.lookup[int(s)] for s in source_ids]
        except KeyError as exc:
            raise ValueError("source absent from noise tape") from exc
        return self.variance_noise[index - self.start_index, ids]

    def bundle(self):
        return dict(variance_noise=self.variance_noise, start_index=self.start_index,
                    source_ids=list(self.source_ids), schedule_sha=self.schedule_sha)

    @classmethod
    def from_bundle(cls, b):
        return cls(b["variance_noise"], b["start_index"], tuple(b["source_ids"]), b["schedule_sha"])


def new_tape(seed, prompt_id, replicate, source_ids, shape, schedule_sha, start=50):
    slices = []
    for index in range(start, 100):
        parts = []
        for source in source_ids:
            key = np.random.SeedSequence([seed, prompt_id, replicate, int(source), index])
            s = int(key.generate_state(1, dtype=np.uint64)[0]) % (2**63 - 1)
            parts.append(torch.randn(shape, generator=torch.Generator().manual_seed(s), dtype=torch.float32))
        slices.append(torch.stack(parts))
    return NoiseTape(torch.stack(slices), start, tuple(source_ids), schedule_sha)
