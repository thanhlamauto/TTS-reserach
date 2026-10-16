"""Immutable inputs, explicit checkpoint contract, and atomic output helpers."""
import hashlib
import json
import os
from pathlib import Path

import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(x):
    x = x.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str((str(x.dtype), tuple(x.shape))).encode())
    h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load_tensor(path):
    return torch.load(path, map_location="cpu", weights_only=True)


CHECKPOINT_FIELDS = {"schema", "meta", "source_xt", "source_clean", "bank_clean",
                     "bank_reward", "signal_scale"}
META_FIELDS = {"prompt", "prompt_id", "bank_id", "prefix_id", "next_index",
               "clean_prediction_index", "schedule_sha", "model_id", "model_revision",
               "vae_id", "state_dtype", "provenance", "cold_initial_aux_nfe"}


def validate_checkpoint(cp):
    # A closed schema prevents source outcomes from entering the steering worker.
    if set(cp) != CHECKPOINT_FIELDS or cp["schema"] != "otlab.checkpoint.v2":
        raise ValueError("checkpoint schema/fields invalid; terminal source outcomes forbidden")
    meta = cp["meta"]
    required = META_FIELDS - {"provenance", "cold_initial_aux_nfe"}
    if set(meta) - META_FIELDS or not required <= set(meta):
        raise ValueError("invalid metadata or unapproved outcome field")
    index = meta["next_index"]
    if type(index) is not int or not 0 <= index < 100 or meta["clean_prediction_index"] != index:
        raise ValueError("stale clean prediction or invalid next index")
    for key in ("bank_id", "prefix_id", "schedule_sha", "model_id", "model_revision", "vae_id"):
        if not isinstance(meta[key], str) or not meta[key]:
            raise ValueError(f"missing identity: {key}")
    x, clean, bank, reward = (cp[k] for k in ("source_xt", "source_clean", "bank_clean", "bank_reward"))
    if x.ndim != 4 or len(x) == 0 or x.shape != clean.shape or bank.ndim != 4 or len(bank) == 0:
        raise ValueError("checkpoint shapes invalid")
    if x.shape[1:] != bank.shape[1:] or reward.shape != (len(bank),):
        raise ValueError("bank/source mismatch")
    if clean.dtype != torch.float32 or bank.dtype != torch.float32 or reward.dtype != torch.float32:
        raise ValueError("clean predictions and bank rewards must be FP32")
    if x.dtype not in (torch.bfloat16, torch.float32) or meta["state_dtype"] != str(x.dtype):
        raise ValueError("state dtype/provenance mismatch")
    if any(t.device.type != "cpu" or not torch.isfinite(t).all() for t in (x, clean, bank, reward)):
        raise ValueError("checkpoint requires finite CPU tensors")
    scale = float(cp["signal_scale"])
    if not 0 < scale <= 1:
        raise ValueError("invalid VP signal scale")
    return cp


def new_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path
