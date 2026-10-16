"""Separate FP32 VAE/DINO surrogate. Imports models only inside the worker."""
from collections import Counter
import importlib.metadata
from pathlib import Path

import torch
import torch.nn.functional as F

from .io import file_sha


class TensorFeatures:
    def __init__(self, config):
        if importlib.metadata.version("transformers") != "4.56.2":
            raise ValueError("the isolated feature worker requires Transformers4.56.2")
        from diffusers import AutoencoderKL
        from transformers import AutoImageProcessor, AutoModel
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        self.counters = Counter()
        enc = config["encoder"]
        if not enc.get("revision") or len(enc["revision"]) != 40:
            raise ValueError("encoder requires an immutable revision")
        local = config.get("local_files_only", True)
        self.processor = AutoImageProcessor.from_pretrained(enc["model_id"], revision=enc["revision"],
                                                           local_files_only=local)
        self.model = AutoModel.from_pretrained(enc["model_id"], revision=enc["revision"],
                                              local_files_only=local, torch_dtype=torch.float32,
                                              attn_implementation="eager").eval().to(self.device)
        self.vae = AutoencoderKL.from_pretrained(config["vae"]["model_id"], subfolder="vae",
                                                local_files_only=local, torch_dtype=torch.float32).eval().to(self.device)
        self.model.requires_grad_(False)
        self.vae.requires_grad_(False)
        families = {"dinov3": ("dinov3_vit", 16), "dinov2": ("dinov2", 14)}
        family = enc.get("family", "dinov3")
        if family not in families:
            raise ValueError("encoder family must be explicitly audited")
        model_type, patch_size = families[family]
        if self.model.config.model_type != model_type or self.model.config.hidden_size != 768:
            raise ValueError(f"fixed pilot requires the declared {family} base encoder")
        if self.model.config.patch_size != patch_size:
            raise ValueError("wrong encoder patch size")
        p = self.processor
        if int(p.resample) != 3 or not p.do_resize or not p.do_normalize:
            raise ValueError("unsupported preprocessing; audit before changing it")
        size = p.size
        if "height" in size and "width" in size:
            self.size = (size["height"], size["width"])
        elif "shortest_edge" in size:
            self.size = (size["shortest_edge"],) * 2
        else:
            raise ValueError("unknown processor resize contract")
        self.mean = torch.tensor(p.image_mean, device=self.device)[None, :, None, None]
        self.std = torch.tensor(p.image_std, device=self.device)[None, :, None, None]
        from huggingface_hub import try_to_load_from_cache
        local_encoder = Path(enc["model_id"])
        encoder_weight = local_encoder / "model.safetensors" if local_encoder.is_dir() else \
            try_to_load_from_cache(enc["model_id"], "model.safetensors", revision=enc["revision"])
        vae_weight = Path(config["vae"]["model_id"]) / "vae/diffusion_pytorch_model.safetensors"
        if not isinstance(encoder_weight, (str, Path)) or not Path(encoder_weight).is_file() or not vae_weight.is_file():
            raise ValueError("fingerprint the exact local safetensors files before real execution")
        self.identity = dict(mode="pretrained", encoder_family=family, encoder_revision=enc["revision"],
                             encoder_sha256=file_sha(encoder_weight), vae_sha256=file_sha(vae_weight),
                             surrogate_dtype="float32", device=str(self.device),
                             versions={name: importlib.metadata.version(name) for name in
                                       ["torch", "transformers", "diffusers", "numpy", "Pillow"]})

    def preprocess(self, rgb01):
        if rgb01.ndim != 4 or rgb01.shape[1] != 3 or rgb01.shape[-2] != rgb01.shape[-1]:
            raise ValueError("first pass supports square NCHW RGB only")
        p = F.interpolate(rgb01, size=self.size, mode="bicubic", align_corners=False, antialias=True)
        if getattr(self.processor, "do_center_crop", False):
            crop = self.processor.crop_size
            h, w = crop["height"], crop["width"]
            top, left = (p.shape[-2] - h) // 2, (p.shape[-1] - w) // 2
            if top < 0 or left < 0:
                raise ValueError("crop exceeds resized image")
            p = p[:, :, top:top + h, left:left + w]
        return (p - self.mean) / self.std

    def encode_pixels(self, rgb01):
        self.counters["feature_forward_samples"] += len(rgb01)
        output = self.model(pixel_values=self.preprocess(rgb01.to(self.device)))
        return F.normalize(output.last_hidden_state[:, 0].float(), dim=-1).to("cpu")

    def encode_rgb(self, rgb):
        if rgb.dtype != torch.uint8:
            raise ValueError("P1 features require the exact scored uint8 pixels")
        return self.encode_pixels(rgb.float() / 255)

    def encode_clean(self, clean):
        self.counters["vae_surrogate_forward_samples"] += len(clean)
        if clean.requires_grad:
            def count_backward(g):
                self.counters["feature_backward_samples"] += len(clean)
                return g
            clean.register_hook(count_backward)
        latent = clean.to(device=self.device, dtype=torch.float32) / self.vae.config.scaling_factor
        decoded = self.vae.decode(latent, return_dict=False)[0]
        return self.encode_pixels((decoded / 2 + .5).clamp(0, 1))


def gradient_preflight(encoder, clean, seed=314159):
    if torch.is_inference_mode_enabled():
        raise RuntimeError("gradient preflight cannot run in inference mode")
    x = clean[:1].detach().float().requires_grad_(True)
    f = encoder.encode_clean(x)
    repeat = encoder.encode_clean(x.detach())
    cotangent = torch.randn(f.shape, generator=torch.Generator().manual_seed(seed))
    loss = (f * cotangent).sum()
    gradient = torch.autograd.grad(loss, x)[0]
    direction = torch.randn(x.shape, generator=torch.Generator().manual_seed(seed + 1))
    direction = direction / direction.norm()
    analytic = float((gradient * direction).sum())
    finite_differences = []
    with torch.no_grad():
        for h in [.02, .01, .005]:
            fp = (encoder.encode_clean(x + h * direction) * cotangent).sum()
            fm = (encoder.encode_clean(x - h * direction) * cotangent).sum()
            fd = float((fp - fm) / (2 * h))
            finite_differences.append(dict(h=h, value=fd, relative_error=abs(fd - analytic) / max(abs(analytic), 1e-7)))
    repeat_error = float((f.detach() - repeat.detach()).abs().max())
    passed = bool(torch.isfinite(gradient).all() and gradient.norm() > 0 and
                  repeat_error <= 1e-6 and min(d["relative_error"] for d in finite_differences) <= .15)
    return dict(passed=passed, repeat_max_abs=repeat_error, directional_analytic=analytic,
                finite_differences=finite_differences, gradient_norm=float(gradient.norm()),
                note="Numerical smoke only; no reward-quality conclusion")
