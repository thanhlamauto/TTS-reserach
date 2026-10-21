"""Conditional mechanism metrics; all scouts stay eligible for every method."""
import csv,itertools,json
from pathlib import Path
import numpy as np
from otlab.io import atomic_json,file_sha
from ot_next.output_selection import cosine_distance

def measure(cfg,u,rewards,dino,clip,logs):
    r=rewards.double().numpy();dc=cosine_distance(clip.numpy());dd=cosine_distance(dino.numpy());rows=[]
    shared=np.arange(36);quality=cfg['quality_threshold'];novelty=cfg['novelty_threshold']
    common_best=[max(r[p],r[4+p*8:12+p*8].max()) for p in range(4)]
    # A fixed append-only proxy archive: quality-sorted common pool, then accepted
    # candidates by parent index. This is not maximum semantic mode coverage.
    common_archive=[]
    for i in np.argsort(-r[:36],kind='stable'):
        if r[i]>=quality and (not common_archive or dc[i,common_archive].min()>=novelty):common_archive.append(int(i))
    for k,name in enumerate(cfg['methods']):
        accepted=[];gain=[0.]*4;bad_gain=[0.]*4;novel=0;archive=common_archive.copy();active=sum(l['active'] for l in logs)
        for p in range(4):
            if k==0:continue
            i=36+(k-1)*4+p;base_bad=max(0.,quality-r[p]);new_bad=max(0.,quality-r[i])
            take=logs[p]['active'] and base_bad-new_bad>=cfg['delta_badness']
            bad_gain[p]=base_bad-new_bad if take else 0.
            if take:
                accepted.append(i);gain[p]=max(0.,r[i]-common_best[p])
                if r[i]>=quality and (not archive or dc[i,archive].min()>=novelty):archive.append(i);novel+=1
        pool=list(shared)+accepted;top=sorted(pool,key=lambda i:(-r[i],i))[:8]
        rows.append(dict(method=name,unit_id=u['unit_id'],prompt_id=u['prompt_id'],seed=u['seed'],
            primary_incremental_best_ir=float(np.mean(gain)),incremental_best_ir_by_parent=gain,
            pool_best_ir=float(r[pool].max()),common_pool_best_ir=float(r[:36].max()),
            common_pool_parent_best_ir=common_best,top8_ir_mean=float(r[top].mean()),
            top8_clip_diversity=float(dc[np.ix_(top,top)][np.triu_indices(8,1)].mean()),
            top8_dino_diversity=float(dd[np.ix_(top,top)][np.triu_indices(8,1)].mean()),
            accepted=len(accepted),active_directions=active,corrected_in_top8=sum(i>=36 for i in top),
            contribution_parents=sum(g>1e-12 for g in gain),quality_novel_proxy_added=novel,
            common_quality_novel_proxy_archive_size=len(common_archive),
            mean_accepted_badness_reduction=float(np.mean(bad_gain)),top8_indices=top,
            pool_indices=pool,all_scout_outputs_retained=True,
            cold_sampling_nfe=2160 if k==0 else 2360,
            semantic_good_distinct='NOT_EVALUATED; quality-novel is IR+CLIP proxy only'))
    return rows

def contrast(rows,left,right):
    per=[]
    for pid in sorted({r['prompt_id'] for r in rows}):
        a=[r['primary_incremental_best_ir'] for r in rows if r['prompt_id']==pid and r['method']==left]
        b=[r['primary_incremental_best_ir'] for r in rows if r['prompt_id']==pid and r['method']==right]
        assert len(a)==len(b)==2;per.append(float(np.mean(a)-np.mean(b)))
    d=np.array(per);rng=np.random.default_rng(2026102109)
    boot=d[rng.integers(0,len(d),(20000,len(d)))].mean(1)
    perm=[float((d*np.array(s)).mean()) for s in itertools.product([-1,1],repeat=len(d))]
    p=sum(v>=d.mean()-1e-15 for v in perm)/len(perm)
    return dict(left=left,right=right,delta=float(d.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
        exact_one_sided_p=p,prompt_deltas=per,wins=int((d>1e-12).sum()),ties=int((abs(d)<=1e-12).sum()),losses=int((d< -1e-12).sum()))

def aggregate(root,cfg):
    root=Path(root);rows=[];directions=[];hashes={}
    for u in cfg['units']:
        case=next(root.glob('chip*/'+u['unit_id']));meta=json.loads((case/'COMPLETE.json').read_text())
        assert meta['passed'] and meta['unit']==u and meta['raw_sha256']==file_sha(case/'raw.pt')
        rows.extend(json.loads((case/'metrics.json').read_text())['rows'])
        directions.extend(json.loads((case/'DIRECTION.json').read_text()))
        hashes[str(case.relative_to(root))]=meta['raw_sha256']
    assert len(rows)==60 and len(directions)==48
    summaries=[]
    keys=['primary_incremental_best_ir','pool_best_ir','top8_ir_mean','top8_clip_diversity','top8_dino_diversity','accepted','contribution_parents','corrected_in_top8','quality_novel_proxy_added']
    for n in cfg['methods']:
        selected=[r for r in rows if r['method']==n]
        summaries.append(dict(method=n,units=12,cold_nfe=selected[0]['cold_sampling_nfe'],**{k:float(np.mean([r[k] for r in selected])) for k in keys}))
    contrasts=[contrast(rows,'cov025','random025'),contrast(rows,'cov050','random050')]
    ordered=sorted(contrasts,key=lambda x:x['exact_one_sided_p']);prev=0.
    for i,c in enumerate(ordered):prev=max(prev,(2-i)*c['exact_one_sided_p']);c['holm_p']=min(1.,prev)
    ready=[c['left'] for c in contrasts if c['ci95'][0]>0 and c['holm_p']<=.05 and any(r['method']==c['left'] and r['contribution_parents']>0 for r in rows)]
    decision='MECHANISM_SIGNAL_REVIEW_REQUIRED' if ready else 'NO_MECHANISM_EVIDENCE_DO_NOT_OPEN_ALLOCATION_SWEEP'
    result=dict(summaries=summaries,contrasts=contrasts,decision=decision,eligible_caps=ready,
        parents=48,active_directions=sum(d['active'] for d in directions),
        below_quality_parents=sum(d['parent_below_quality'] for d in directions),flat_badness_parents=sum(d['flat_badness'] for d in directions),
        split_cosine=[d['split_cosine'] for d in directions],r_Q=[d['r_Q'] for d in directions],
        actual_covariance_rms={n:[d['proposals'][n]['actual_rms'] for d in directions] for n in ['cov025','cov050']},
        raw_bundle_hashes=hashes,semantic_success='NOT_EVALUATED',reserved_final_used=False,
        inference='Six development prompts; bootstrap descriptive; paired sign flips exact under sign symmetry. Threshold inherited from one old prompt; not universal.')
    atomic_json(root/'RESULTS.json',result)
    for name,data in [('summary.csv',summaries),('per_unit.csv',rows),('contrasts.csv',contrasts)]:
        with (root/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    lines=['# Diffusion-prior covariance repair — vòng 1','',f'Quyết định: **{decision}**. Hoàn tất 6 TRAIN development prompt × 2 seed × 4 parent = 48 parent; không dùng final/VALIDATION/TEST.',
        '',f"Có {result['below_quality_parents']}/48 parent dưới quality threshold; {result['active_directions']}/48 direction vượt signal gate; {result['flat_badness_parents']}/48 ensemble có badness phẳng.",
        '', '| Method | Cold NFE/unit | Incremental IR ngoài scout | Best IR pool | Mean IR top8 | Accepted/unit | Parent thêm best/unit |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:lines.append(f"| {r['method']} | {r['cold_nfe']} | {r['primary_incremental_best_ir']:.6f} | {r['pool_best_ir']:.6f} | {r['top8_ir_mean']:.6f} | {r['accepted']:.3f} | {r['contribution_parents']:.3f} |")
    lines+=['','## Covariance trừ random cùng actual BF16 RMS','']
    for c in contrasts:lines.append(f"- {c['left']}: Δprimary {c['delta']:+.6f}; CI95% {c['ci95']}; exact one-sided p={c['exact_one_sided_p']:.6f}; Holm p={c['holm_p']:.6f}; W/T/L prompt {c['wins']}/{c['ties']}/{c['losses']}.")
    lines+=['','## Phương pháp và giới hạn','',
        'SD1.5/DDIM100 eta1 CFG7.5 BF16 trên TPU hiện có. Tạo parent sau50 updates, forward-noise về đúng level index45 rồi denoise45..49 bằng sampler gốc để có8 biến thể tại level50. Các baseline/scout/correction của cùng parent dùng cùng tape50..99; các root dùng noise độc lập. K=8, eta1, hai cap0.25/0.5×median RMS(scouter-parent). Covariance không normalize lên cap; BF16 actual cap và random matching≤2% đã kiểm tra.',
        '', 'Signal gate: parent dưới quality threshold; badness std>1e−6; cosine giữa covariance của hai nửa ensemble>0. Parent đã tốt hoặc tín hiệu không ổn định giữ nguyên latent. Acceptance: badness giảm≥0.05, không backtracking hoặc đổi threshold. Toàn bộ baseline và8scout/parent luôn nằm trong pool; chỉ thêm correction được acceptance. Tất cả correction kể cả no-op vẫn rollout thật và trả đủ NFE.',
        '', 'Primary là phần tăng best IR ngoài toàn bộ shared pool của từng parent, average4parent rồi average2seed/prompt. Hai cap contrasts được Holm. Các CI chỉ có6 prompt và mang tính exploratory. Không có independent semantic guard nên không tuyên bố bảo toàn mọi yêu cầu prompt; IR+CLIP novelty chỉ proxy, không gọi là nhiều mode tốt.',
        '', 'Cold mỗi recipe: no-correction4×(50+8×5+9×50)=2160NFE; corrected4×(50+8×5+10×50)=2360NFE. Tổng actual bằng share prefix/scouts:12×2960+400replay=35920NFE; logical cold tất cả recipe139200. Zero padding, zero auxiliary UNet; CFG branches=71840. Tính cả160NFE tạo ensemble/unit, baseline, scout và correction. Đây không phải head-to-head BFS ở đúng2400NFE; không dùng số lịch sử để thay baseline mới.',
        '', 'Tỷ lệ50/25/25 và width sweep chưa chạy. Chỉ xem xét round2/round3 nếu có bằng chứng correction đóng góp ngoài scout và hơn matched random theo gate đã khóa. Không nominate finalist từ pilot này.',
        '', 'Raw ở chip*/p*/raw.pt; direction/actual update ở DIRECTION.json; noise hashes ở PAIRING.json; counters ở COMPLETE.json. comparison.jpg hiển thị cả candidate thô, bao gồm candidate bị reject, không mặc định là output được chấp nhận.']
    (root/'REPORT_VI.md').write_text('\n'.join(lines)+'\n')
    return result
