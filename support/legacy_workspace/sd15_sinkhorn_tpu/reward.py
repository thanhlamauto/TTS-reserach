"""Reward interface: implement score(prompt, images) to substitute a verifier."""
import time
from pathlib import Path
from typing import Protocol
import numpy as np
import torch


class Reward(Protocol):
    def score(self,prompt,images): ...


class ImageRewardCPU:
    def __init__(self,dcs_root):
        from scripts.diffusion_classical_search_t2i.tpu_runner import _load_reward
        root=Path(dcs_root)
        self.model=_load_reward(root/'official',root/'cache')
        self.evaluations=0

    @torch.inference_mode()
    def score(self,prompt,images):
        values=[float(x) for x in self.model.score_batched([prompt]*len(images),images)]
        if len(values)!=len(images) or not np.isfinite(values).all():
            raise RuntimeError('invalid ImageReward scores')
        self.evaluations+=len(images)
        return values


class CLIPCPU:
    def __init__(self,cfg,dcs_root):
        from transformers import CLIPModel,CLIPProcessor
        kw=dict(revision=cfg['clip_revision'],cache_dir=str(Path(dcs_root)/'cache'/'sinkhorn_clip'),local_files_only=True)
        self.model=CLIPModel.from_pretrained(cfg['clip_model'],**kw).eval()
        self.processor=CLIPProcessor.from_pretrained(cfg['clip_model'],**kw)

    @torch.inference_mode()
    def score(self,prompt,images):
        image_features=[]
        for offset in range(0,len(images),8):
            data=self.processor(images=images[offset:offset+8],return_tensors='pt')
            f=self.model.get_image_features(**data)
            image_features.append(torch.nn.functional.normalize(f,dim=-1))
        im=torch.cat(image_features)
        data=self.processor(text=[prompt],padding=True,truncation=True,return_tensors='pt')
        txt=torch.nn.functional.normalize(self.model.get_text_features(**data),dim=-1)
        return dict(clip_alignment=(im@txt.T).flatten().tolist(),clip_embeddings=im.tolist())


def prefetch(cfg,dcs_root):
    # Public supplementary CLIP weights only. SD1.5 and ImageReward stay local-only.
    from huggingface_hub import snapshot_download
    return snapshot_download(cfg['clip_model'],revision=cfg['clip_revision'],
                             cache_dir=str(Path(dcs_root)/'cache'/'sinkhorn_clip'),
                             allow_patterns=['config.json','pytorch_model.bin','preprocessor_config.json',
                                             'tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json'])


def evaluate_images(prompt,images,reward,clip):
    t=time.perf_counter(); scores=reward.score(prompt,images); reward_s=time.perf_counter()-t
    t=time.perf_counter(); clip_data=clip.score(prompt,images); clip_s=time.perf_counter()-t
    return dict(rewards=scores,**clip_data,reward_seconds=reward_s,clip_seconds=clip_s)


if __name__=='__main__':
    import argparse
    from .common import load_config
    p=argparse.ArgumentParser(); p.add_argument('--dcs-root',required=True)
    a=p.parse_args(); print(prefetch(load_config(),a.dcs_root))
