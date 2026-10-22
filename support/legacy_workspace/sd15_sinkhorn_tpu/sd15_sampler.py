"""Original DDIM schedule, BF16 UNet/VAE, FP32 DDIM arithmetic.

Scalar timestep/coefficient tensors are device inputs, avoiding a distinct UNet
graph for every Python timestep. No gradients, training or model modification.
"""
import time
from pathlib import Path
import torch
from .common import atomic_json


def checkpoint_step(progress, steps):
    import math
    return int(math.ceil(progress*steps))


def ddim_coefficients(scheduler, timestep, eta):
    t = int(timestep)
    prev = t-scheduler.config.num_train_timesteps//scheduler.num_inference_steps
    at = scheduler.alphas_cumprod[t].float()
    ap = scheduler.alphas_cumprod[prev].float() if prev >= 0 else scheduler.final_alpha_cumprod.float()
    sigma = eta*(((1-ap)/(1-at)*(1-at/ap)).clamp_min(0)).sqrt()
    return torch.stack([at.sqrt(),(1-at).sqrt(),ap.sqrt(),(1-ap-sigma*sigma).clamp_min(0).sqrt(),sigma])


def ddim_update(x, eps, coefficients, noise=None):
    c = coefficients.unbind()
    x0 = (x.float()-c[1]*eps.float())/c[0]
    y = c[2]*x0+c[3]*eps.float()
    if noise is not None: y = y+c[4]*noise.float()
    return y


class Sampler:
    def __init__(self, rt, cfg, dcs_root):
        from scripts.diffusion_classical_search_t2i import tpu_runner as baseline
        self.rt,self.cfg = rt,cfg
        manifest = {'models':{'sd15':{'id':cfg['model'],'revision':cfg['revision'],
                                    'execution_dtype':'bfloat16','use_safetensors':True}}}
        start = time.perf_counter()
        self.pipe = baseline._load_pipeline('sd15',manifest,Path(dcs_root)/'official',Path(dcs_root)/'cache',rt.device)
        self.pipe.unet.requires_grad_(False); self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.scheduler = self.pipe.scheduler
        s = self.scheduler.config
        if s.prediction_type != 'epsilon' or s.clip_sample or s.thresholding:
            raise ValueError('custom DDIM loop supports epsilon prediction without clipping/thresholding')
        self.scheduler.set_timesteps(cfg['steps'],device='cpu')
        self.timesteps = [int(t) for t in self.scheduler.timesteps]
        self.coefficients = [ddim_coefficients(self.scheduler,t,cfg['ddim_eta']) for t in self.timesteps]
        self.embeddings = {}
        self.batch = cfg['batch_size']
        rt.sync()
        atomic_json(rt.output/'environment'/f'sampler{rt.rank}.json',dict(
            load_seconds=time.perf_counter()-start,scheduler_config=dict(s),timesteps=self.timesteps,
            cfg_scale=cfg['cfg_scale'],steps=cfg['steps'],ddim_eta=cfg['ddim_eta'],
            checkpoint_semantics='state AFTER ceil(progress*steps) updates; next index equals completed updates',
            unet_dtype='bfloat16',scheduler_arithmetic='float32, state rounded to bfloat16 after each update',
            safety_checker='not invoked: raw model output used for research evaluation'))

    @torch.no_grad()
    def embedding(self,prompt):
        if prompt not in self.embeddings:
            tok = self.pipe.tokenizer(['',prompt],padding='max_length',max_length=self.pipe.tokenizer.model_max_length,
                                      truncation=True,return_tensors='pt')
            enc = self.pipe.text_encoder(tok.input_ids.to(self.rt.device))[0]
            self.embeddings[prompt] = torch.cat([enc[:1].repeat(self.batch,1,1),enc[1:].repeat(self.batch,1,1)])
            self.rt.sync()
        return self.embeddings[prompt]

    @torch.no_grad()
    def run(self,prompt,initial,start=0,checkpoints=(),noise_seed=0):
        if not 0 <= start < len(self.timesteps): raise ValueError('invalid resume index')
        initial = initial.cpu().to(torch.bfloat16)
        if not torch.isfinite(initial).all(): raise ValueError('nonfinite starting state')
        embed = self.embedding(prompt)
        saved = {k:[] for k in checkpoints}; final = []
        begin = time.perf_counter()
        for offset in range(0,len(initial),self.batch):
            chunk = initial[offset:offset+self.batch]; count = len(chunk)
            if count < self.batch: chunk = torch.cat([chunk,chunk[-1:].repeat(self.batch-count,1,1,1)])
            x = chunk.to(self.rt.device)
            for k in range(start,len(self.timesteps)):
                timestep = torch.tensor(self.timesteps[k],dtype=torch.int64,device=self.rt.device)
                model_input = torch.cat([x,x])
                pred = self.pipe.unet(model_input,timestep,encoder_hidden_states=embed,return_dict=False)[0]
                uncond,cond = pred.chunk(2)
                eps = uncond.float()+self.cfg['cfg_scale']*(cond.float()-uncond.float())
                coeff = self.coefficients[k].to(self.rt.device)
                noise = None
                if self.cfg['ddim_eta'] > 0:
                    # Stateless per-step CPU RNG also makes stochastic replay exact.
                    generator = torch.Generator().manual_seed(noise_seed+k*100003+offset)
                    noise = torch.randn(chunk.shape,generator=generator).to(self.rt.device)
                x = ddim_update(x,eps,coeff,noise).to(torch.bfloat16)
                self.rt.xm.mark_step()
                if k+1 in saved:
                    saved[k+1].append(x[:count].cpu())
            final.append(x[:count].cpu())
            print(f'denoised start={start} batch={offset}:{offset+count} steps={len(self.timesteps)-start}',flush=True)
        self.rt.sync()
        result = torch.cat(final)
        if not torch.isfinite(result).all(): raise RuntimeError('nonfinite generated latents')
        return result,{k:torch.cat(v) for k,v in saved.items()},time.perf_counter()-begin

    @torch.no_grad()
    def decode(self,latents):
        images=[]
        for offset in range(0,len(latents),self.batch):
            x=latents[offset:offset+self.batch]; count=len(x)
            if count < self.batch: x=torch.cat([x,x[-1:].repeat(self.batch-count,1,1,1)])
            y=self.pipe.vae.decode(x.to(self.rt.device)/self.pipe.vae.config.scaling_factor,return_dict=False)[0]
            self.rt.sync()
            y=(y[:count].float().cpu()/2+0.5).clamp(0,1)
            images.extend(self.pipe.image_processor.numpy_to_pil(y.permute(0,2,3,1).numpy()))
        return images

    def warmup(self,prompt):
        # Full original schedule exercised twice: first includes compile, second is warm.
        x=torch.zeros(self.batch,4,64,64,dtype=torch.bfloat16)
        t=time.perf_counter(); y,_,_=self.run(prompt,x); self.decode(y)
        first=time.perf_counter()-t
        metrics_first=self.rt.metrics()
        t=time.perf_counter(); y,_,_=self.run(prompt,x); self.decode(y)
        second=time.perf_counter()-t
        rec=dict(first_pass_seconds=first,steady_pass_seconds=second,
                 compile_overhead_estimate_seconds=max(0,first-second),
                 note='difference estimate includes other first-use costs; exact XLA CompileTime in metrics text',
                 warmup_sample_steps=2*self.batch*self.cfg['steps'])
        atomic_json(self.rt.output/'environment'/f'warmup{self.rt.rank}.json',rec)
        (self.rt.output/'environment'/f'xla_warmup{self.rt.rank}.txt').write_text(metrics_first+'\n'+self.rt.metrics())
        print('WARMUP',rec,flush=True)
        return rec
