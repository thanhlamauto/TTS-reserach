import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

from sd15_sinkhorn_tpu.common import atomic_json
from sd15_sinkhorn_tpu.evaluate import summary_metrics
from sd15_sinkhorn_bfs_compare.analyze import read,csv_write
from .core import METHODS


def pool_metrics(bank,suffix):
    scores={k:bank[k]+suffix[k] for k in ['rewards','clip_alignment','clip_embeddings']}
    # Stable descending order: bank entries occur first and win exact ties.
    ids=np.argsort(-np.array(scores['rewards']),kind='stable')[:8]
    s=summary_metrics(scores,indices=ids)
    return dict(pool_best_reward=s['best_reward'],pool_top8_mean_reward=s['mean_reward'],
        pool_top8_diversity=s['clip_diversity'],candidate_fraction_in_top8=float((ids>=16).mean()))


def collect(root,reps):
    root=Path(root); plan=read(root/'frozen_plan.json'); prior=Path(plan['prior'])
    per_run=[]; sample_rows=[]
    for rep in reps:
        for i in plan['stage1_indices']:
            bank=read(prior/'banks/test'/f'p{i:03d}'/'bank.json')['sets']['reference']
            bank_best=max(bank['rewards']); records={}
            for name in METHODS:
                path=root/'full'/f'rep{rep}'/f'p{i:03d}'/name/'result.json'
                records[name]=read(path)
            none=records['none']['scores']; none_pool=pool_metrics(bank,none)
            for j,r in enumerate(bank['rewards']):
                sample_rows.append(dict(prompt_id=i,replicate=rep,method='bank',sample_id=j,
                    reward=r,embedding_json=json.dumps(bank['clip_embeddings'][j])))
            for name,r in records.items():
                scores=r['scores']; delta=np.array(scores['rewards'])-np.array(none['rewards'])
                pool=pool_metrics(bank,scores)
                row=dict(prompt_index=i,replicate=rep,method=name,
                    source_mean_reward=r['mean_reward'],source_best_reward=r['best_reward'],
                    source_diversity=r['clip_diversity'],source_mean_gain=float(delta.mean()),
                    source_gain_fraction=float((delta>0).mean()),source_loss_fraction=float((delta<0).mean()),
                    source_gain_p10=float(np.quantile(delta,.1)),**pool,
                    pool_best_gain=pool['pool_best_reward']-none_pool['pool_best_reward'],
                    pool_top8_mean_gain=pool['pool_top8_mean_reward']-none_pool['pool_top8_mean_reward'],
                    candidate_beats_bank_best_fraction=float((np.array(scores['rewards'])>bank_best).mean()),
                    candidate_beats_none_best_fraction=float((np.array(scores['rewards'])>max(none['rewards'])).mean()),
                    candidate_beats_control_pool_best_fraction=float((np.array(scores['rewards'])>none_pool['pool_best_reward']).mean()),
                    creates_new_pool_winner=float(max(scores['rewards'])>none_pool['pool_best_reward']),
                    denoise_seconds=r['denoise_seconds'],wall_seconds=r['wall_seconds'])
                per_run.append(row)
                for j,reward in enumerate(scores['rewards']):
                    sample_rows.append(dict(prompt_id=i,replicate=rep,method=name,sample_id=j,
                        reward=reward,embedding_json=json.dumps(scores['clip_embeddings'][j])))
    return per_run,sample_rows


def ci(values):
    a=np.array(values,dtype='float64'); rng=np.random.default_rng(81031)
    draw=a[rng.integers(len(a),size=(20000,len(a)))].mean(1)
    return [float(np.quantile(draw,.025)),float(np.quantile(draw,.975))]


def aggregate(rows):
    groups=defaultdict(list)
    for r in rows: groups[(r['method'],r['prompt_index'])].append(r)
    prompt_rows=[]
    keys=[k for k in rows[0] if k not in ['method','prompt_index','replicate']]
    for (method,i),rr in groups.items():
        prompt_rows.append(dict(method=method,prompt_index=i,
            **{k:float(np.mean([r[k] for r in rr])) for k in keys}))
    summary=[]
    for name in METHODS:
        rr=[r for r in prompt_rows if r['method']==name]
        summary.append(dict(method=name,n_prompts=len(rr),
            **{k:float(np.mean([r[k] for r in rr])) for k in keys},
            source_mean_gain_ci=ci([r['source_mean_gain'] for r in rr]),
            pool_best_gain_ci=ci([r['pool_best_gain'] for r in rr]),
            pool_top8_mean_gain_ci=ci([r['pool_top8_mean_gain'] for r in rr])))
    pairs=[]
    aa={r['prompt_index']:r for r in prompt_rows if r['method']=='A'}
    for name in METHODS:
        if name=='A': continue
        bb={r['prompt_index']:r for r in prompt_rows if r['method']==name}
        pair=dict(A='A',B=name,n_prompts=len(aa))
        for metric in ['source_mean_reward','pool_best_reward','pool_top8_mean_reward','source_diversity']:
            delta=[aa[i][metric]-bb[i][metric] for i in sorted(aa)]
            pair[metric+'_delta']=float(np.mean(delta)); pair[metric+'_ci']=ci(delta)
        pairs.append(pair)
    return prompt_rows,summary,pairs


def analyze(root,reps):
    root=Path(root); rows,samples=collect(root,reps)
    assert len(rows)==12*6*len(reps)
    prompt_rows,summary,pairs=aggregate(rows)
    out=root/('analysis' if len(reps)==2 else 'analysis_rep0'); out.mkdir(parents=True,exist_ok=True)
    csv_write(out/'per_run.csv',rows); csv_write(out/'per_prompt.csv',prompt_rows)
    csv_write(out/'summary.csv',summary); csv_write(out/'paired_comparisons.csv',pairs)
    with (out/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['prompt_id','replicate','method','sample_id','reward','embedding_json'])
        w.writeheader(); w.writerows(samples)
    by_rep={}
    for rep in reps:
        _,ss,pp=aggregate([r for r in rows if r['replicate']==rep])
        by_rep[str(rep)]=dict(summary=ss,paired=pp)
    atomic_json(out/'summary.json',dict(replicates=reps,n_prompts=12,summary=summary,
                                      paired=pairs,per_replicate=by_rep))
    stage0=read(root/'stage0/summary.json')
    lines=['# Chẩn đoán cơ chế Sinkhorn A trên TPU','',
        'SD1.5 BF16 / DDIM100 eta1 / CFG7.5 /512px; checkpoint sau50 update. Cấu hình frozen epsilon0.1, OT step0.3, trust0.1, raw cost, top25% (mode contrast với gamma_neg=0, tương đương top). 12 prompt đầu manifest, không chọn theo reward.','',
        f'Suffix replicates: {reps}. Bank và prefix giữ chung giữa replicate; chỉ suffix noise và shuffle/random-control seed thay đổi. Trong mỗi prompt/replicate, sáu nhánh dùng cùng tensor noise từng particle/từng bước. Các replicate được trung bình bên trong prompt trước bootstrap20.000 lần qua12 prompt.','',
        '## Stage0 — 48 dump từ checkpoint thật','',
        '| epsilon | Common energy | Target row KL | Interaction std/epsilon | Update RMS thực | Cast lost fraction | Cast relative L2 error |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in stage0:
        lines.append(f"| {r['epsilon']} | {r['common_energy_mean']:.8f} | {r['target_row_kl_mean']:.8f} | {r['interaction_std_over_epsilon_mean']:.6f} | {r['actual_update_rms_mean']:.6f} | {r['cast_lost_fraction_mean']:.6f} | {r['cast_relative_l2_error_mean']:.6f} |")
    lines+=['','## Stage1 — paired source và pool metrics','',
        '| Method | Source mean | Δsource vs none | CI95% | Δpool best | Δpool top8 mean | Candidate fraction top8 | New pool winner fraction | Source diversity |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lo,hi=r['source_mean_gain_ci']
        lines.append(f"| {r['method']} | {r['source_mean_reward']:.6f} | {r['source_mean_gain']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {r['pool_best_gain']:+.6f} | {r['pool_top8_mean_gain']:+.6f} | {r['candidate_fraction_in_top8']:.4f} | {r['creates_new_pool_winner']:.4f} | {r['source_diversity']:.6f} |")
    lines+=['','## A so với controls','',
        '| A − control | Δsource mean | CI95% | Δpool best | CI95% |',
        '|---|---:|---|---:|---|']
    for r in pairs:
        lo,hi=r['source_mean_reward_ci']; pl,ph=r['pool_best_reward_ci']
        lines.append(f"| A − {r['B']} | {r['source_mean_reward_delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {r['pool_best_reward_delta']:+.6f} | [{pl:+.6f}, {ph:+.6f}] |")
    lines+=['','## Protocol và giới hạn','',
        '- Stage0 tái tạo solver float32 gốc từ latent checkpoint thật. Run cũ không lưu mass plans; đây là plans trả về từ phép tái tạo đã kiểm chứng original steered tensor, không phải plans historical đã lưu. Không refine plan để giảm marginal error. Chỉ shuffled control dùng solver float64 riêng (tol1e-12).',
        '- X_after trong dump là BF16 sau đúng entry conversion của sampler và là tensor Stage1 truyền vào model evaluation tiếp theo. Không gọi FP32 proposal là update thực. Target/source final rewards không được dùng để tạo field; chỉ reference rewards tham gia.',
        '- Mean direction=sum(w-u)Z. Bốn controls match RMS từng source theo delta A sau cast; actual ratios/cosines lưu trong controls/*.pt. Threshold post-cast RMS relative error2% được khóa trước chạy. Mean matched không bắt buộc cùng biên độ mọi source.',
        '- Pool mỗi method/replicate chỉ gồm16 reference +8 suffix của chính method đó. Control pool=16 reference+8none. Ties ưu tiên bank; không gộp methods/replicates và không tính semantic duplicate detection. Candidate fractions là identity-level accounting.',
        '- CLIP diversity dùng chính evaluator cũ: mean(1−e_i·e_j) qua unordered pairs của normalized ViT-B/32 embeddings. Source8 và pool top8 được tính riêng. Source gain/loss fractions,p10 và các winner metrics bổ sung có trong per_run.csv.',
        '- New particle/step keyed variance noise khác stream run cũ, nên none/A đều được chạy lại. Không dùng suffix cũ. Custom DDIM arithmetic giữ nguyên, chỉ thay nguồn variance-noise tensor; scheduler full100-step không đổi thành50-step.',
        '- Đây là development diagnostics trên12 prompt cũ và một bank/prefix mỗi prompt. Replicate1 chỉ đo ổn định theo suffix noise; không nhân cỡ mẫu thành24 hay192 particle. Nhiều contrast chưa điều chỉnh multiplicity; CI chứa0 không chứng minh tương đương.',
        '- Full diagnostic suffix budget=12×6×8×50×số replicate. Reference cache không tính generation/reward mới; CSV copy16 bank rows mỗi prompt/replicate là accounting. Warmup và mandatory replay overhead báo riêng trong validation audit. Cold recipe mỗi A vẫn2400NFE.','']
    (out/'REPORT_VI.md').write_text('\n'.join(lines))
    print('ANALYSIS_COMPLETE',reps,len(rows),len(samples),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True)
    p.add_argument('--replicates',default='0,1'); a=p.parse_args()
    analyze(a.output,[int(v) for v in a.replicates.split(',')])
