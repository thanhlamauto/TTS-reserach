"""Same cached experiment with eta16 binding caps; cold scouts remain charged."""
import argparse,fcntl,json,os,shutil,signal,subprocess,time,traceback,urllib.request,tarfile
from pathlib import Path
import torch
import numpy as np
from PIL import Image,ImageDraw
from otlab.io import atomic_json,file_sha,tensor_sha,load_tensor
from otlab.adapter import frozen_xla_factory,schedule_identity
from otlab.export import historical_tape
from otlab.noise import NoiseTape
from ot_next.identity import compatible_scheduler_metadata
from noise_avoid.campaign import CompactRGB
from cov_repair.core import direction,capped_state,matched_random,tests,rms
from cov_repair.analyze import measure,contrast


def restore(cfg,root):
    cache=Path(cfg['cache_root']);cache.mkdir(parents=True,exist_ok=True)
    receipt=cfg['parent_preservation'];archive=cache.parent/'covariance_parent_verified_download.tar.zst'
    url='https://huggingface.co/datasets/'+receipt['hf_repo']+'/resolve/'+receipt['immutable_revision']+'/'+receipt['hf_path']
    if not archive.exists() or archive.stat().st_size!=receipt['bytes'] or file_sha(archive)!=receipt['sha256']:
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url,timeout=90) as src,archive.open('wb') as dst:shutil.copyfileobj(src,dst,8*1024*1024)
                assert archive.stat().st_size==receipt['bytes'] and file_sha(archive)==receipt['sha256'];break
            except Exception:
                if attempt==2:raise
                time.sleep(5*(attempt+1))
    expected=cfg['parent_manifest'];seen=set();p=subprocess.Popen(['zstd','-q','-dc',str(archive)],stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=p.stdout,mode='r|') as tar:
            for member in tar:
                if not member.name.startswith('experiment/'):continue
                assert member.isfile() and member.name in expected
                rel=Path(member.name).relative_to('experiment');assert not rel.is_absolute() and '..' not in rel.parts
                target=cache/rel;target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
                assert target.stat().st_size==expected[member.name]['bytes'] and file_sha(target)==expected[member.name]['sha256']
                seen.add(member.name)
        p.stdout.close();assert p.wait()==0
    finally:
        if p.poll() is None:p.terminate();p.wait()
    assert json.loads((cache/'preregistered.json').read_text())==cfg['parent_protocol']
    assert file_sha(cache/'preregistered.json')==cfg['parent_protocol_sha256']
    for item in cfg['source_cases'].values():assert file_sha(cache/item['case']/'raw.pt')==item['raw_sha256']
    atomic_json(root/'RESTORE.json',dict(passed=True,archive_sha256=receipt['sha256'],immutable_revision=receipt['immutable_revision'],experiment_members_verified=len(seen),all12_raw_hashes_verified=True,cache_in_ram_only=True))
    return cache


def prepare(cfg,cache,root):
    prepared=[];active=[];gate_records=[]
    for u in cfg['units']:
        oldcase=cache/cfg['source_cases'][u['unit_id']]['case'];data=load_tensor(oldcase/'raw.pt')
        assert len(data['rewards'])==52 and len(data['parent'])==4
        oldlogs=json.loads((oldcase/'DIRECTION.json').read_text());logs=[];states=data['parent'].repeat(4,1,1,1)
        for parent in range(4):
            z=data['scout_states'][8*parent:8*(parent+1)]
            v,info=direction(data['parent'][parent],z,data['rewards'][4+8*parent:12+8*parent],float(data['rewards'][parent]),cfg)
            assert info['active']==oldlogs[parent]['active']
            assert abs(info['ungated_covariance_rms']-oldlogs[parent]['ungated_covariance_rms'])<1e-10
            info.update(parent=parent,proposals={})
            key=int(np.random.SeedSequence([u['seed'],u['prompt_id'],parent,210156]).generate_state(1)[0])
            for j,(ratio,cn,rn) in enumerate([(.25,'cov025','random025'),(.5,'cov050','random050')]):
                y,log=capped_state(data['parent'][parent],v,ratio*info['r_Q'],cfg['eta'])
                random,rlog=matched_random(data['parent'][parent],z,log['actual_rms'],torch.Generator().manual_seed(key))
                info['proposals'][cn]=log;info['proposals'][rn]=rlog
                states[(2*j)*4+parent]=y;states[(2*j+1)*4+parent]=random
            if info['active']:
                a,b=info['proposals']['cov025'],info['proposals']['cov050']
                assert a['actual_rms']>.23*info['r_Q'] and b['actual_rms']>.46*info['r_Q']
                assert 1.9<b['actual_rms']/a['actual_rms']<2.1
                assert not torch.equal(states[parent],states[8+parent])
                active.append(dict(unit_id=u['unit_id'],parent=parent))
            else:
                assert all(torch.equal(states[k*4+parent],data['parent'][parent]) for k in range(4))
            logs.append(info);gate_records.append(dict(unit_id=u['unit_id'],**info))
        prepared.append((u,data,logs,states))
    assert active==cfg['expected_active']
    atomic_json(root/'CAP_STATE_GATES.json',dict(passed=True,eta=16,active=active,all48_signal_gates_unchanged=True,
        both_caps_bind_on_all_active=True,actual_large_small_ratio_between_1_9_and_2_1=True,records=gate_records))
    return prepared


def render(data,logs,out):
    rgb,reward=data['rgb'],data['rewards'];p=next(d['parent'] for d in logs if d['active'])
    scout=4+p*8+int(reward[4+p*8:12+p*8].argmax())
    ids=[p,scout]+[36+k*4+p for k in range(4)];labels=['baseline','best scout','cov .25rQ','random .25rQ','cov .5rQ','random .5rQ']
    image=Image.new('RGB',(6*192,224),'white');draw=ImageDraw.Draw(image)
    for k,(i,label) in enumerate(zip(ids,labels)):
        image.paste(Image.fromarray(rgb[i].permute(1,2,0).numpy()).resize((192,192)),(k*192,0))
        draw.text((k*192+3,195),f'{label} IR{float(reward[i]):.3f}',fill='black')
    image.save(out,quality=92)


def analysis(cfg,root,rows,active_details,begin,adapter,encoder):
    summaries=[];metrics=['primary_incremental_best_ir','pool_best_ir','top8_ir_mean','top8_clip_diversity','accepted','corrected_in_top8','contribution_parents','quality_novel_proxy_added']
    for name in cfg['methods']:
        rr=[r for r in rows if r['method']==name];assert len(rr)==12
        summaries.append(dict(method=name,eta=16,cold_nfe=rr[0]['cold_sampling_nfe'],**{m:float(np.mean([r[m] for r in rr])) for m in metrics}))
    contrasts=[contrast(rows,'cov025','random025'),contrast(rows,'cov050','random050')]
    for item in contrasts:item['holm_p']=min(1.,item['exact_one_sided_p']*2)
    result=dict(eta=16,summaries=summaries,contrasts=contrasts,active_parent_results=active_details,
        interpretation='Only2active parents on1prompt. Reused same development cohort, not independent confirmation. Both caps now bind. Full cohort includes46identity no-ops.',
        accepted_covariance_total={n:sum(r['accepted'] for r in rows if r['method']==n) for n in ['cov025','cov050']},
        decision='EXPLORATORY_ONLY_NO_FINALIST',scout_cost_charged=True)
    atomic_json(root/'RESULTS.json',result)
    import csv
    for filename,data in [('per_unit.csv',rows),('summary.csv',summaries),('contrasts.csv',contrasts),('active_parents.csv',active_details)]:
        with (root/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    lines=['# Eta16 — binding caps, paired covariance versus random','',
        'Đã chạy cùng6 TRAIN prompt×2seed×4parent. Ensemble K8, checkpoint50, threshold/gate/acceptance/tape giữ nguyên. Chỉ tăng eta1→16. Hai cap đều chạm trên2parent active;46parent giữ nguyên theo gate cũ.',
        '', '| Parent | Method | Actual RMS | ΔIR so baseline | IR so best scout | Accepted |', '|---|---|---:|---:|---:|---|']
    for r in active_details:lines.append(f"| {r['unit_id']} /{r['parent']} | {r['method']} | {r['actual_rms']:.6f} | {r['delta_ir_vs_base']:+.6f} | {r['delta_ir_vs_best_scout']:+.6f} | {r['accepted']} |")
    lines+=['','## Chi phí gồm scout','',
        'Mỗi corrected recipe lạnh:4×[prefix50+8×prior5+(baseline1+scout8+correction1)×suffix50]=2360NFE/prompt-seed. No-correction=2160NFE. Chi phí scout1600NFE/unit gồm160 tạo ensemble+1280scout suffix? Xem diễn giải dưới.',
        '', 'Chi tiết chính xác: prefix4×50=200; tạo ensemble4×8×5=160; scout4×8×50=1600; baseline4×50=200; correction4×50=200. Tổng2360. Scout construction+rollout=1760NFE/unit, luôn được tính cho cả covariance và random. Dòng diễn giải chi phí chi tiết này là bảng accounting có thẩm quyền.',
        '', 'Cả5recipe×12units có cold nominal139200NFE. Cache dùng lại25920NFE prefix/prior/scout/baseline;9200NFE correction slots là identity no-op dùng lại endpoint đã kiểm; chỉ8candidate thay đổi×50=400NFE chạy mới. Replay4baseline suffix×50=200NFE được tách riêng. Tổng neural mới600NFE,1200CFG branch sample evals,0padding/aux. Không lấy600 thay cho chi phí2360 khi so method.',
        '', 'Tám correction mới có8terminal IR scores và8DINO/CLIP encodes; replay thêm4IR/pixels;432common terminal records được dùng lại với nguồn/hash. Không chấm lại scout miễn phí rồi bỏ chi phí khỏi recipe.',
        '', 'η16 chọn từ norm hướng cũ: power-of2 kế tiếp trênmax(.5rQ/RMS(v)); không dùng reward mới để chọn. Random giữ RNG direction cũ trong ensemble span, khớp actual BF16 RMS≤2%, không lớn hơn covariance. Zero gradients không bị ép lên cap.',
        '', 'Primary và gates giữ như pilot trước; mọi scout luôn có quyền vào pool. Các raw correction dù bị reject vẫn được log riêng. Hai active parents đều thuộc prompt6 nên không có evidence rộng để nominate hoặc mở allocation sweep; bootstrap6prompt chỉ mô tả cohort có5prompt no-op.',
        '', 'Cached parent/scout data được khôi phục từ immutable HF archive và kiểm checksum toàn archive/từngmember. StackSD1.5/DDIM100 eta1 CFG7.5 BF16 không đổi. Gate replay baseline phải khớp bitwise latent/pixels/reward4/4 trước khi chạy correction. Một chip TPU hiện có, không GPU thuê.',
        '', 'Raw mới ởp*/delta.pt; PARENT_CACHE_DEPENDENCY.json định danh archive raw cũ để dựng lại full52candidate pools. Không nhân bản raw cũ trong archive mới.']
    # Keep only the correct explicit NFE breakdown in the report.
    lines=[s for s in lines if not s.startswith('Mỗi corrected recipe lạnh:')]
    (root/'REPORT_VI.md').write_text('\n'.join(lines)+'\n')
    c=dict(adapter.counters)
    assert c['denoise_sample_steps']==600 and c['unet_cfg_branch_sample_evaluations']==1200
    assert c.get('denoise_padding_sample_steps',0)==0 and c.get('aux_clean_sample_evaluations',0)==0
    assert c['terminal_verifier_scores']==8 and c['replay_verifier_scores']==4 and encoder.count==8
    atomic_json(root/'COMPLETE.json',dict(passed=True,eta=16,units=12,parents=48,active_parents=2,cold_nfe_per_corrected_recipe_unit=2360,
        actual_new_sampling_nfe=600,new_candidate_sampling_nfe=400,replay_nfe=200,cold_all_recipes_nfe=139200,
        reused_common_cold_nfe=25920,cached_identity_slots_nfe=9200,counters=c,new_feature_images=8,visible_chips=[0],
        elapsed_seconds=time.time()-begin,actual_billing_usd=None,reserved_final_used=False,decision=result['decision']))
    print('COMPLETE',json.dumps(result),flush=True)


def main():
    arg=argparse.ArgumentParser();arg.add_argument('--protocol',required=True);args=arg.parse_args()
    source=Path(args.protocol).resolve().parent;cfg=json.loads(Path(args.protocol).read_text());root=Path(cfg['root']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'COMPLETE.json').exists():print('ALREADY_COMPLETE');return
    assert not list(root.glob('p*')),'Existing attempt: inspect before resume'
    assert not subprocess.run(['fuser']+[f'/dev/vfio/{r}' for r in range(4)],capture_output=True,text=True).stdout.strip(),'TPU owned'
    begin=time.time();adapter=encoder=None
    def status(stage,**kw):
        r=dict(stage=stage,elapsed_seconds=time.time()-begin,pid=os.getpid(),**kw);atomic_json(root/'status.json',r);print(json.dumps(r),flush=True)
    def stop(sig,frame):raise TimeoutError('Persistent900s deadline reached')
    signal.signal(signal.SIGTERM,stop)
    try:
        atomic_json(root/'preregistered.json',cfg);atomic_json(root/'STATIC_GATES.json',tests())
        atomic_json(root/'SOURCE_HASHES.json',{str(p.relative_to(source)):file_sha(p) for p in source.rglob('*') if p.is_file()})
        status('restore_hashed_parent_cache_from_hf');cache=restore(cfg,root)
        atomic_json(root/'PARENT_CACHE_DEPENDENCY.json',cfg['parent_preservation'])
        status('prepare_all48_parent_cap_states_cpu');prepared=prepare(cfg,cache,root)
        status('load_original_backend_chip0');old=Path(cfg['old_root'])/'attempts/8074f94c847d';ac=json.loads((old/'adapter_config.json').read_text());ac['runtime_output']=str(root/'runtime')
        adapter=frozen_xla_factory(ac);prior=json.loads((old/'neural_runtime/environment/sampler0.json').read_text())['scheduler_config']
        assert compatible_scheduler_metadata(prior,dict(adapter.sampler.scheduler.config))
        adapter.sampler.scheduler.register_to_config(_use_default_values=prior['_use_default_values'])
        expected=json.loads((old/'replay.json').read_text())['records'][0]['adapter_identity']['schedule_sha']
        assert schedule_identity(adapter.sampler)==expected;adapter.schedule_sha=expected;adapter.base_identity['schedule_sha']=expected
        encoder=CompactRGB(str(Path(cfg['old_root'])/'worker_venv/bin/python'),source.parents[1]/'ot_tts_lab_v2_reimpl/configs/worker_dinov2.yaml',root/'rgb_worker')
        atomic_json(root/'identity.json',dict(adapter=adapter.identity,encoder=encoder.identity,visible_chip=os.environ['TPU_VISIBLE_CHIPS']))
        u,data,_,_=prepared[0];adapter.set_prompt(u['prompt']);history=historical_tape(adapter.sampler,int(data['noise_seed']),4,start=0)
        status('original_cached_baseline_replay4');y=adapter.advance(data['parent'],50,100,tuple(range(4)),history);r,rgb=adapter.score(y,'replay')
        assert torch.equal(y,data['final'][:4]) and torch.equal(rgb,data['rgb'][:4]) and torch.equal(r,data['rewards'][:4]),'Cached baseline replay mismatch'
        atomic_json(root/'REPLAY.json',dict(passed=True,cases=4,latent_bitwise=True,pixels_bitwise=True,reward_exact=True,replay_nfe=200))
        from sd15_sinkhorn_tpu.reward import CLIPCPU
        adapter.clip=CLIPCPU(adapter.sampler.cfg,cfg['dcs_root']);rows=[];details=[]
        for u,data,logs,states in prepared:
            case=root/u['unit_id'];case.mkdir();adapter.set_prompt(u['prompt'])
            assert data['schedule_sha']==adapter.schedule_sha
            full=dict(final=data['final'][:4].repeat(4,1,1,1),rgb=data['rgb'][:4].repeat(4,1,1,1),
                rewards=data['rewards'][:4].repeat(4),dino=data['dino'][:4].repeat(4,1),clip=data['clip'][:4].repeat(4,1))
            active=[d['parent'] for d in logs if d['active']]
            if active:
                assert len(active)==1;p=active[0];slots=[k*4+p for k in range(4)];status('eta16_covariance_random4',unit=u['unit_id'],parent=p)
                h=historical_tape(adapter.sampler,int(data['noise_seed']),4,start=0)
                tape=NoiseTape(h.variance_noise[50:,[p]*4].clone(),50,tuple(range(4)),adapter.schedule_sha)
                end=adapter.advance(states[slots],50,100,tuple(range(4)),tape);r,rgb=adapter.score(end,'terminal')
                f=encoder.encode(rgb,u['unit_id']);clip=torch.tensor(adapter.clip.score(adapter.prompt,[Image.fromarray(im.permute(1,2,0).numpy()) for im in rgb])['clip_embeddings'])
                for key,val in [('final',end),('rgb',rgb),('rewards',r),('dino',f),('clip',clip)]:full[key][slots]=val
                torch.save(dict(parent=p,correction_slots=slots,states=states[slots],final=end,rewards=r,rgb=rgb,dino=f,clip=clip,
                    parent_unit=u,noise_seed=data['noise_seed'],schedule_sha=expected,future_tape_sha=tensor_sha(tape.variance_noise)),case/'delta.pt')
                for k,name in enumerate(cfg['methods'][1:]):
                    base=float(data['rewards'][p]);best=float(data['rewards'][4+8*p:12+8*p].max());value=float(r[k])
                    details.append(dict(unit_id=u['unit_id'],parent=p,method=name,eta=16,actual_rms=logs[p]['proposals'][name]['actual_rms'],
                        terminal_ir=value,base_ir=base,best_scout_ir=best,delta_ir_vs_base=value-base,delta_ir_vs_best_scout=value-best,
                        accepted=bool(max(0.,cfg['quality_threshold']-base)-max(0.,cfg['quality_threshold']-value)>=cfg['delta_badness'])))
            merged={key:torch.cat([data[key][:36],full[key]]) for key in full}
            rr=measure(cfg,u,merged['rewards'],merged['dino'],merged['clip'],logs);json.dumps(rr,allow_nan=False);rows+=rr
            atomic_json(case/'metrics.json',dict(unit=u,rows=rr));atomic_json(case/'DIRECTION.json',logs)
            atomic_json(case/'CACHE_RECEIPT.json',dict(parent_raw_sha256=cfg['source_cases'][u['unit_id']]['raw_sha256'],physical_new_suffix_nfe=200 if active else 0,
                cold_nfe_each_corrected_method=2360,all_scout_outputs_retained=True,identity_noops_reused=12 if active else 16))
            if active:render(merged,logs,case/'comparison.jpg')
        status('aggregate');analysis(cfg,root,rows,details,begin,adapter,encoder)
    except BaseException as exc:
        atomic_json(root/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),counters={} if adapter is None else dict(adapter.counters)));raise
    finally:
        if encoder:encoder.close()
        if adapter:(root/'xla_metrics.txt').write_text(adapter.sampler.rt.metrics())
if __name__=='__main__':main()
