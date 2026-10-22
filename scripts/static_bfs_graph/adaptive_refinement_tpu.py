"""Exact TPU DDIM replay and shared-tree edge/split profiling for adaptive nodes."""
import os, time, random, math, json, copy, argparse, traceback
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
import torch_xla.core.xla_model as xm
from .adaptive_refinement import *
from .adapter import StaticGraphBFSAdapter
from .runner import Runtime
from .tpu_reference_compare import checked_inputs
from scripts.diffusion_classical_search_t2i import tpu_runner as baseline

DCS=Path('/home/thanhlamtba31/static-bfs-dcs')
def th(t):return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()
def args_for(rec,pid,seed):
    return {'lmbda':rec.get('base_temperature',10.),'num_particles':8,'use_smc':True,'guidance_reward_fn':'ImageReward','time_steps':100,'potential_type':'max','tempering_schedule':rec.get('tempering','constant'),'resampling':'ssp','gamma':rec.get('gamma'),'resampling_steps':rec['resampling_steps'],'selection_mode':'manual' if rec['kind']=='manual' else 'raw_tau','temperature_by_step':rec.get('temperature_by_step',[]),'graph_trial_seed':seed,'graph_prompt_index':pid}

class Engine:
    def __init__(self,root,chip,stage):
        verify(root);self.root=root;self.chip=chip;self.stage=stage;self.cfg=read(root/'config.json');self.split=read(root/'data_split.json')
        _,cfg,old=checked_inputs();cfg={**cfg,'particles':8,'protocol_version':self.cfg['protocol']}
        prompts=[{'prompt':self.split['TRAIN_prompts'].get(str(i))} for i in range(100)]
        self.runtime=Runtime(root/'runtime'/stage,DCS,self.cfg['seeds']['A'][0],chip,cfg,old,prompts)
        self.pipe=self.runtime.pipe;self.reward=self.runtime.reward;self.ref=read(root/'reference_schedule.json')
        self.account={'actual_denoising_NFE':0,'actual_verifier_candidate_scores':0,'actual_verifier_batch_calls':0,'reference_trajectories':0,'logical_observations_new':0,'worker_load_seconds':self.runtime.load_seconds}
        self.live_reference=None
        self.start=time.time();self.ledger=root/'work_ledgers'/f'{stage}-chip{chip}-{os.getpid()}.json'
    def persist(self):dump(self.ledger,{**self.account,'stage':self.stage,'chip':self.chip,'wall_seconds_including_load':time.time()-self.start+self.runtime.load_seconds})
    def scores(self,images,pid):
        self.account['actual_verifier_candidate_scores']+=8;self.account['actual_verifier_batch_calls']+=1
        v=[float(s) for s in self.reward.score_batched([self.split['TRAIN_prompts'][str(pid)]]*8,images)]
        assert len(v)==8 and all(math.isfinite(s) for s in v)
        return v
    def capture(self,idx,latents,embeds,generator,adapter):
        xm.mark_step();xm.wait_device_ops()
        return {'idx':idx,'latents':latents.detach().clone(),'embeds':embeds.detach().clone(),'previous':adapter.previous.detach().clone(),'events':copy.deepcopy(adapter.events),'generator':generator.get_state().clone(),'torch':torch.get_rng_state().clone(),'numpy':np.random.get_state(),'python':random.getstate(),'xla':xm.get_rng_state(self.pipe.device),'timesteps':self.pipe.scheduler.timesteps.detach().clone().cpu()}
    @torch.no_grad()
    def direct(self,pid,seed,rec,snapshot=False,pre_index=None):
        pseed=baseline._prompt_seed(seed,pid);random.seed(pseed);np.random.seed(pseed);baseline._seed_numba(pseed);torch.manual_seed(pseed);xm.set_rng_state(pseed,self.pipe.device)
        generator=torch.Generator(device='cpu').manual_seed(pseed);pipe=self.pipe
        initial=pipe.prepare_latents(8,pipe.unet.config.in_channels,pipe.unet.config.sample_size*pipe.vae_scale_factor,pipe.unet.config.sample_size*pipe.vae_scale_factor,pipe.unet.dtype,pipe.device,generator)
        xm.mark_step();initial_state={'generator':generator.get_state().clone(),'torch':torch.get_rng_state().clone(),'numpy':np.random.get_state(),'python':random.getstate(),'xla':xm.get_rng_state(pipe.device)}
        captured={};embeds=None;last=None;pre={}
        import fkd_pipeline_sd as module
        engine=self
        class Tracking(StaticGraphBFSAdapter):
            def resample(self,*,sampling_idx,latents,x0_preds):
                if sampling_idx==pre_index:pre['latent_hash']=th(latents);pre['previous_hash']=th(self.previous);pre['events']=copy.deepcopy(self.events)
                return super().resample(sampling_idx=sampling_idx,latents=latents,x0_preds=x0_preds)
        module.FKD=Tracking
        def callback(_,idx,t,kw):
            nonlocal embeds,last
            xm.mark_step();engine.account['actual_denoising_NFE']+=8
            if idx==0:embeds=kw['prompt_embeds'].detach().clone()
            if snapshot and idx in engine.cfg['snapshots']:captured[idx]=engine.capture(idx,kw['latents'],kw['prompt_embeds'],generator,baseline._LAST_ADAPTER)
            if idx==99:last=kw['latents'].detach().clone()
            return kw
        output=pipe([self.split['TRAIN_prompts'][str(pid)]]*8,num_inference_steps=100,eta=1.,generator=generator,latents=initial.clone(),fkd_args=args_for(rec,pid,seed),output_type='pil',callback_on_step_end=callback,callback_on_step_end_tensor_inputs=['latents','prompt_embeds'])
        xm.mark_step();xm.wait_device_ops();adapter=baseline._LAST_ADAPTER
        events=copy.deepcopy(adapter.events);assert [e['sampling_index'] for e in events]==rec['resampling_steps']
        self.account['actual_verifier_candidate_scores']+=len(events)*8;self.account['actual_verifier_batch_calls']+=len(events)
        images=list(output.images);v=self.scores(images,pid)
        row={'reward':max(v),'scores':v,'latent_hash':th(last),'RGB_hashes':[hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest() for im in images],'events':events,'initial_latents_sha256':th(initial),'pre':pre}
        if snapshot:
            captured[-1]={'idx':-1,'latents':initial.detach().clone(),'embeds':embeds,'previous':torch.zeros(8,dtype=torch.float32),'events':[],'timesteps':pipe.scheduler.timesteps.detach().clone().cpu(),**initial_state}
            assert set(captured)==set(self.cfg['snapshots'])|{-1}
        return row,captured
    def reference(self,cohort,pid,seed):
        # CPU BF16 archives cannot reproduce the live XLA state. Rebuild once
        # per unit/process, check the canonical endpoint, and reuse live clones.
        key=(cohort,pid,seed)
        if self.live_reference is not None and self.live_reference[0]==key:
            return self.live_reference[1:]
        self.live_reference=None
        base=self.root/'references'/cohort/f'p{pid}-s{seed}';record=base.with_suffix('.json');snap=base.with_suffix('.pt')
        old=None
        if record.exists():
            old=read(record);assert snap.exists() and sha(snap)==old['snapshots_sha256']
            assert old.get('checkpoint_mode')=='live_xla_archive_only','legacy CPU replay checkpoint is not valid'
        row,states=self.direct(pid,seed,self.ref,True)
        self.account['reference_trajectories']+=1
        if old is not None:
            assert all(row[k]==old[k] for k in ['scores','RGB_hashes','latent_hash','events','initial_latents_sha256']),'regenerated reference endpoint mismatch'
            row=old
        else:
            archive={i:{k:(v.detach().cpu() if isinstance(v,torch.Tensor) else v) for k,v in state.items()} for i,state in states.items()}
            snap.parent.mkdir(parents=True,exist_ok=True);tmp=snap.with_suffix('.pt.tmp');torch.save(archive,tmp);os.replace(tmp,snap)
            row.update({'snapshots_sha256':sha(snap),'prompt':pid,'seed':seed,'checkpoint_mode':'live_xla_archive_only','latent_hash_mode':'live_float_expansion'})
            dump(record,row)
        self.persist();self.live_reference=(key,row,states);return row,states
    def restore(self,snap,rec,pid,seed):
        assert snap['latents'].device.type=='xla','CPU archives are not exact replay states'
        torch.set_rng_state(snap['torch']);np.random.set_state(snap['numpy']);random.setstate(snap['python']);xm.set_rng_state(snap['xla'],self.pipe.device)
        generator=torch.Generator(device='cpu');generator.set_state(snap['generator'])
        self.pipe.scheduler.set_timesteps(100,device=self.pipe.device);assert torch.equal(self.pipe.scheduler.timesteps.cpu(),snap['timesteps'])
        from fkd_pipeline_sd import latent_to_decode
        def reward_fn(images):
            pil=list(self.pipe.image_processor.postprocess(images,output_type='pil'));return torch.tensor(self.scores(pil,pid),device=images.device)
        adapter=StaticGraphBFSAdapter(latent_to_decode_fn=lambda x:latent_to_decode(model=self.pipe,output_type='pil',latents=x),reward_fn=reward_fn,device=self.pipe.device,**args_for(rec,pid,seed))
        adapter.previous=snap['previous'].clone().to(self.pipe.device);adapter.events=copy.deepcopy(snap['events'])
        self.pipe._guidance_scale=7.5;self.pipe._guidance_rescale=0.;self.pipe._cross_attention_kwargs=None
        return snap['latents'].clone().to(self.pipe.device),snap['embeds'].to(self.pipe.device),generator,adapter
    @torch.no_grad()
    def resume_pipeline(self,snap,rec,pid,seed,stop=None,collect=(),reward_cache=None,collect_reward=False):
        # Keep the official loop, tensor lifetimes and verifier synchronization.
        # An equivalent handwritten DDIM loop is not bitwise equivalent on XLA.
        import fkd_pipeline_sd as module
        engine=self;offset=snap['idx']+1;captured={};cache={};last={}
        lat,_,generator,_=self.restore(snap,rec,pid,seed);last['latents']=lat
        original_retrieve=module.retrieve_timesteps
        class StopPrefix(Exception):pass
        class Resume(StaticGraphBFSAdapter):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                self.previous=snap['previous'].clone().to(engine.pipe.device);self.events=copy.deepcopy(snap['events'])
                self.cached_event_count=0
                original_reward=self.reward_fn
                if reward_cache is not None or collect_reward:
                    def checked_reward(decoded):
                        # Preserve the original VAE + CPU postprocessing boundary;
                        # cache only ImageReward, after checking all eight pixels.
                        images=list(engine.pipe.image_processor.postprocess(decoded,output_type='pil'))
                        hashes=[hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest() for im in images]
                        if reward_cache is not None:
                            assert hashes==reward_cache['RGB_hashes'],'cached verifier image mismatch'
                            raw=torch.tensor(reward_cache['raw']).to(decoded.device);self.cached_event_count+=1
                        else:
                            raw=original_reward(decoded)
                            cache.update({'RGB_hashes':hashes,'raw':[float(x) for x in raw.detach().cpu().tolist()]})
                        return raw
                    self.reward_fn=checked_reward
            def resample(self,*,sampling_idx,latents,x0_preds):
                return super().resample(sampling_idx=sampling_idx+offset,latents=latents,x0_preds=x0_preds)
        module.FKD=Resume
        def retrieve(*args,**kwargs):
            times,n=original_retrieve(*args,**kwargs);return times[offset:],n
        module.retrieve_timesteps=retrieve
        def callback(_,i,t,kw):
            idx=i+offset;xm.mark_step();engine.account['actual_denoising_NFE']+=8
            if idx in collect:captured[idx]=engine.capture(idx,kw['latents'],kw['prompt_embeds'],generator,baseline._LAST_ADAPTER)
            if idx==99:last['latents']=kw['latents'].detach().clone()
            if idx==stop:raise StopPrefix()
            return kw
        output=None
        try:
            output=self.pipe([self.split['TRAIN_prompts'][str(pid)]]*8,num_inference_steps=100,eta=1.,generator=generator,latents=lat,fkd_args=args_for(rec,pid,seed),output_type='pil',callback_on_step_end=callback,callback_on_step_end_tensor_inputs=['latents','prompt_embeds'])
        except StopPrefix:
            assert stop is not None
        finally:
            module.retrieve_timesteps=original_retrieve
        adapter=baseline._LAST_ADAPTER
        new_events=len(adapter.events)-len(snap['events'])-adapter.cached_event_count
        self.account['actual_verifier_candidate_scores']+=8*new_events;self.account['actual_verifier_batch_calls']+=new_events
        if stop is not None:
            assert stop in captured and output is None
            return captured,cache
        assert output is not None
        images=list(output.images);v=self.scores(images,pid)
        assert [e['sampling_index'] for e in adapter.events]==rec['resampling_steps']
        row={'reward':max(v),'scores':v,'latent_hash':th(last['latents']),'RGB_hashes':[hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest() for im in images],'events':copy.deepcopy(adapter.events)}
        return row
    @torch.no_grad()
    def trunk(self,src,states,pid,seed,dests):
        prefix={e['sampling_index']:e['temperature'] for e in states[src]['events']};rec=static_record(StaticBFSPolicy(tuple(prefix),tuple(prefix.items()),8))
        # Fork immediately BEFORE each destination update. Recompute that one
        # update in the official loop so verifier fusion/materialization agrees.
        targets={d-1 for d in dests if d!=100}|({99} if 100 in dests else set())
        captured={src:states[src]};later=targets-{src}
        if later:
            extra,_=self.resume_pipeline(states[src],rec,pid,seed,stop=max(later),collect=later)
            captured.update(extra)
        return {d:(captured[99 if d==100 else d-1],None) for d in dests}
    @torch.no_grad()
    def cached(self,snap,x0,rec,pid,seed):
        idx=snap['idx']+1
        _,cache=self.resume_pipeline(snap,rec,pid,seed,stop=idx,collect=[idx],collect_reward=True)
        assert set(cache)=={'RGB_hashes','raw'}
        return cache
    @torch.no_grad()
    def suffix(self,snap,rec,pid,seed,cache=None):
        if cache is not None:
            idx=snap['idx']+1
            states,_=self.resume_pipeline(snap,rec,pid,seed,stop=idx,collect=[idx],reward_cache=cache)
            snap=states[idx]
        return self.resume_pipeline(snap,rec,pid,seed)
    @torch.no_grad()
    def profile(self,cohort,pid,seed,jobs):
        rr,states=self.reference(cohort,pid,seed);missing=[]
        for j in jobs:
            p=unitpath(self.root,cohort,pid,seed,j)
            if p.exists():assert read(p)['job']==j
            else:missing.append(j)
        groups=defaultdict(list)
        for j in missing:groups[j['src']].append(j)
        for src,group in groups.items():
            dests=sorted({j['mid'] if j['mid'] is not None else j['dst'] for j in group});trunk=self.trunk(src,states,pid,seed,dests)
            cache={}
            for j in group:
                pol=intervention(self.root,j['src'],j['dst'],j['tau'],j['mid'],j['mtau']);rec=static_record(pol);first=j['mid'] if j['mid'] is not None else j['dst'];snap,x0=trunk[first]
                if first!=100 and first not in cache:cache[first]=self.cached(snap,x0,rec,pid,seed)
                row=self.suffix(snap,rec,pid,seed,cache.get(first));row.update({'regret':rr['reward']-row['reward'],'job':j,'prompt':pid,'seed':seed,'reference_sha256':sha(self.root/'reference_schedule.json'),'reference_reward':rr['reward'],'policy':rec})
                dump(unitpath(self.root,cohort,pid,seed,j),row);self.account['logical_observations_new']+=1;self.persist()
    def gates(self):
        pid=self.split['A'][0];seed=self.cfg['seeds']['A'][0]
        off,_=self.direct(pid,seed,self.ref,False);on,states=self.direct(pid,seed,self.ref,True)
        assert all(off[k]==on[k] for k in ['scores','RGB_hashes','latent_hash','events','initial_latents_sha256']),'snapshot on/off mismatch'
        for i,s in states.items():assert [e['sampling_index'] for e in s['events']]==[t for t in self.ref['resampling_steps'] if t<=i]
        rr,_=self.reference('gate',pid,seed)
        base=job(20,60,8.);splits=[job(20,60,8.,30,t) for t in TAUS];self.profile('gate',pid,seed,[base]+splits)
        bpol=static_record(intervention(self.root,20,60,8.));independent,_=self.direct(pid,seed,bpol,False,30)
        saved=read(unitpath(self.root,'gate',pid,seed,base));assert all(saved[k]==independent[k] for k in ['scores','RGB_hashes','latent_hash','events']),'coarse replay differs from full pipeline'
        prefixes=[]
        for j in splits:
            rec=static_record(intervention(self.root,20,60,8.,30,j['mtau']));direct,_=self.direct(pid,seed,rec,False,30);shared=read(unitpath(self.root,'gate',pid,seed,j))
            assert all(shared[k]==direct[k] for k in ['scores','RGB_hashes','latent_hash','events']),'tau-fork/split independent mismatch'
            assert direct['pre']==independent['pre'],'split prefix differs before midpoint';prefixes.append(direct['pre'])
        # Exercise a source and destination absent from reference and an END branch.
        for j in [job(-1,20,8.),job(50,75,2.),job(90,100,None),job(80,100,None,95,8.)]:
            self.profile('gate',pid,seed,[j]);rec=static_record(intervention(self.root,j['src'],j['dst'],j['tau'],j['mid'],j['mtau']));direct,_=self.direct(pid,seed,rec,False);shared=read(unitpath(self.root,'gate',pid,seed,j));assert all(direct[k]==shared[k] for k in ['scores','RGB_hashes','latent_hash','events'])
        for nodes in [G0,FINE]:
            for es in all_paths(nodes,TAUS,3):assert len(policy_from_path(es,8).resampling_steps)==3
        assert len(records()['MANUAL8']['resampling_steps'])==3
        assert set(map(int,self.split['TRAIN_prompts']))==set(self.split['A']+self.split['B'])
        dump(self.root/'gates.json',{'passed':True,'gates_1_to_8':True,'snapshot_on_off_all8_exact':True,'coarse_independent_exact':True,'three_tau_split_independent_exact':True,'new_source_destination_and_END_exact':True,'TRAIN_only':True,'wall_seconds_excluding_model_load':time.time()-self.start})

def worker(root,stage,chip):
    eng=Engine(root,chip,stage)
    try:
        if stage=='gates':eng.gates()
        elif stage=='audit':
            frozen=read(root/'frozen_policies.json');unique={p['id']:p for p in frozen.values()};units=list(itertools.product(eng.split['C'],eng.cfg['seeds']['C']))
            for pid,seed in units[chip::4]:
                for policy in unique.values():
                    path=root/'audit'/f'p{pid}-s{seed}-{policy["id"]}.json'
                    if path.exists():assert read(path)['policy']==policy;continue
                    row,_=eng.direct(pid,seed,policy);assert len(row['events'])==3;dump(path,{'prompt':pid,'seed':seed,'policy':policy,**row});eng.persist()
        else:
            req=read(root/'tasks'/f'{stage}.json');cohort=req['cohort'];units=list(itertools.product(eng.split[cohort],eng.cfg['seeds'][cohort]))
            for pid,seed in units[chip::4]:eng.profile(cohort,pid,seed,req['jobs'])
        eng.persist();dump(root/'workers'/stage/f'chip{chip}.done.json',{'success':True,'account':eng.account})
    except BaseException as exc:
        eng.persist();dump(root/'failures'/f'{stage}-chip{chip}.json',{'exception':repr(exc),'traceback':traceback.format_exc()});raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--stage',required=True);p.add_argument('--chip',type=int,required=True);a=p.parse_args();worker(a.run_root,a.stage,a.chip)
