import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np

from sd15_sinkhorn_tpu.common import atomic_json, digest
from sd15_sinkhorn_tpu.evaluate import summary_metrics
from sd15_sinkhorn_bfs_compare.analyze import read, recycled, bootstrap, csv_write


def prefix_top8(scores, count=24):
    subset={key:scores[key][:count] for key in ['rewards','clip_alignment','clip_embeddings']}
    assert all(len(v)==count for v in subset.values())
    order=np.argsort(subset['rewards'],kind='stable')[-8:]
    return summary_metrics(subset,indices=order)


def build_rows(root):
    root=Path(root); plan=read(root/'frozen_plan.json')
    prior=Path(plan['prior'])/'full/100eta1'; rows=[]; inherited=[]
    configs=plan['selection']['configs']
    for i,prompt in enumerate(plan['prompts']):
        bank_dir=prior/'banks/test'/f'p{i:03d}'; bank_path=bank_dir/'bank.json'
        bank=read(bank_path); inherited.append(bank_path)
        assert bank['prompt']==prompt and bank['seed']==plan['seed']
        assert bank['n_reference']==16 and bank['n_source']==8
        assert bank['sampler']['steps']==100 and bank['sampler']['ddim_eta']==1
        common=dict(prompt_index=i,prompt=prompt,diffusion_sample_steps=2400)
        ref=bank['sets']['reference']
        for name,n in plan['bfs']:
            path=root/'full/bfs'/f'p{i:03d}'/f'{name}_n{n}/result.json'
            r=read(path)
            assert r['prompt']==prompt and r['seed']==plan['seed']
            assert r['n']==24 and r['steps']==100 and r['ddim_eta']==1
            assert r['finite'] and r['diffusion_sample_steps']==2400
            assert len(r['events'])==3
            rows.append(dict(**common,method=r['method'],family='BFS',
                **{k:r[k] for k in ['mean_reward','best_reward','clip_alignment','clip_diversity']},
                output_policy='top8_terminal24',reward_evaluations=96,source_file=str(path)))
        bon_path=bank_dir/'fresh_bon_28/result.json'; bon=read(bon_path); inherited.append(bon_path)
        assert bon['prompt']==prompt and bon['seed']==plan['seed'] and bon['fresh_count']==28
        rows.append(dict(**common,method='fresh_bon24',family='BoN',
                         **prefix_top8(bon['scores']),output_policy='top8_first24',
                         reward_evaluations=24,source_file=str(bon_path)))
        # A separate independent-draw control. Source finals are legal ONLY in this row.
        parts=[('reference',ref,bank_dir/'reference/images'),
               ('source',bank['sets']['source'],bank_dir/'source/images')]
        rows.append(dict(**common,method='independent_bank24',family='control',**recycled(parts),
                         output_policy='top8_reference16_unguided8',reward_evaluations=24,
                         source_file=str(bank_path)))
        for cfg in configs:
            cid=digest(cfg)[:12]
            path=prior/'test/trials'/f'p{i:03d}'/cid/'result.json'
            r=read(path); inherited.append(path)
            assert r['config']==cfg and r['prompt']==prompt and r['seed']==plan['seed']
            assert r['bank_identity']==bank['identity'] and r['bf16_finite']
            assert cfg['checkpoint']==.5 and r['incremental_sample_steps']==400
            label=f"A_eps{cfg['epsilon']}"
            rows.append(dict(**common,method=label+'_guided',family='Sinkhorn',
                **summary_metrics(r['scores']),output_policy='guided8',reward_evaluations=24,
                epsilon=cfg['epsilon'],source_file=str(path)))
            # Deliberately never include the unguided source terminal images here.
            parts=[('reference',ref,bank_dir/'reference/images'),
                   ('guided',r['scores'],path.parent/'images')]
            rows.append(dict(**common,method=label+'_ref_guided',family='Sinkhorn',
                **recycled(parts),output_policy='top8_reference16_guided8',
                reward_evaluations=24,epsilon=cfg['epsilon'],source_file=str(path)))
    counts=defaultdict(int)
    for r in rows:
        counts[r['method']]+=1
        assert np.isfinite([r[k] for k in ['mean_reward','best_reward','clip_alignment','clip_diversity']]).all()
    assert len(counts)==8 and all(n==24 for n in counts.values()),dict(counts)
    return rows,sorted(set(inherited))


def analyze(root):
    root=Path(root); out=root/'analysis'; out.mkdir(parents=True,exist_ok=True)
    rows,inherited=build_rows(root)
    groups=defaultdict(list)
    for r in rows: groups[r['method']].append(r)
    summary=[]
    metrics=['best_reward','mean_reward','clip_diversity','clip_alignment']
    for name,rr in groups.items():
        summary.append(dict(method=name,n_prompts=len(rr),diffusion_sample_steps=2400,
            reward_evaluations=rr[0]['reward_evaluations'],output_policy=rr[0]['output_policy'],
            **{k:float(np.mean([r[k] for r in rr])) for k in metrics}))
    summary.sort(key=lambda r:-r['best_reward'])
    pairs=[]
    for a in ['bfs_manual_n24','bfs_graph_n24','fresh_bon24','independent_bank24']:
        for b in [m for m in groups if m.startswith('A_')]:
            aa={r['prompt_index']:r for r in groups[a]}; bb={r['prompt_index']:r for r in groups[b]}
            assert set(aa)==set(bb)==set(range(24))
            rec=dict(A=a,B=b,n_prompts=24,budget_matched=True)
            for metric in metrics:
                delta=np.array([aa[i][metric]-bb[i][metric] for i in range(24)])
                lo,hi=bootstrap(delta)
                rec.update({metric+'_delta':float(delta.mean()),metric+'_ci_low':lo,metric+'_ci_high':hi})
            pairs.append(rec)
    csv_write(out/'results.csv',rows); csv_write(out/'summary.csv',summary)
    csv_write(out/'paired_comparisons.csv',pairs)
    from .run import sha
    atomic_json(out/'inherited_manifest.json',{str(p):sha(p) for p in inherited})
    atomic_json(out/'analysis.json',dict(complete=True,summary=summary,paired=pairs))
    labels={'bfs_graph_n24':'BFS graph, N24','bfs_manual_n24':'BFS manual, N24',
        'fresh_bon24':'Best-of-24','independent_bank24':'Independent bank 16+8 (control)'}
    for eps in [.1,.3]:
        labels[f'A_eps{eps}_guided']=f'Sinkhorn A eps={eps}, guided8'
        labels[f'A_eps{eps}_ref_guided']=f'Sinkhorn A eps={eps}, top8 reference+guided'
    lines=['# So sánh tại 2.400 NFE/prompt','',
        'SD1.5 BF16, DDIM100 eta=1, CFG7.5, 512px; cùng 24 prompt follow-up và seed20361015. Hai cấu hình A và hai policy BFS giữ nguyên, không tuning lại. Mỗi phương pháp trả 8 ảnh. Best/prompt = lấy reward lớn nhất của mỗi prompt rồi trung bình qua 24 prompt.','',
        '| Method | NFE/prompt | Reward eval/prompt | Best/prompt | Mean reward 8 ảnh | CLIP diversity |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {labels[r['method']]} | 2400 | {r['reward_evaluations']} | {r['best_reward']:.6f} | {r['mean_reward']:.6f} | {r['clip_diversity']:.6f} |")
    lines+=['','## Paired bootstrap','',
        'Hiệu A−B; CI95% từ 20.000 paired prompt bootstrap, seed81031. Nhiều contrast mang tính thăm dò, chưa điều chỉnh multiplicity.','',
        '| A−B | Best/prompt Δ | CI95% |','|---|---:|---:|']
    for r in pairs:
        if r['A']=='independent_bank24': continue
        lines.append(f"| {labels[r['A']]} − {labels[r['B']]} | {r['best_reward_delta']:+.6f} | [{r['best_reward_ci_low']:+.6f}, {r['best_reward_ci_high']:+.6f}] |")
    lines+=['','## Cách tính ngân sách và tái sử dụng dữ liệu','',
        '- Sinkhorn: 16 reference ×100 + 8 source ×50 + 8 guided ×50 = 2400. Không dùng ảnh cuối hay reward của source không steering để tạo guided hoặc chọn ảnh Sinkhorn. Output guided8 chỉ trả 8 guided; output reference+guided chọn top8 từ 16 reference và 8 guided đã có trong đúng ngân sách này.',
        '- Các guided output đã được sinh trong benchmark trước và được dùng lại vì source prefix/checkpoint và reference không thay đổi. Chi phí ở đây là cold execution tối thiểu của recipe, không phải tổng compute thực thi lại hôm nay. Nhánh source đối chứng 400 NFE của run cũ bị loại khỏi budget và khỏi mọi tập ứng viên Sinkhorn.',
        '- Best-of-24 dùng đúng 24 draw đầu của fresh BoN28 cũ, rồi chấm lại thống kê từ 24 draw đó. Sampler microbatch4 và RNG theo offset đảm bảo 24 draw này trùng một run N24; không chọn subset dựa trên reward.',
        '- BFS được chạy mới với N24 thật, dùng resampling toàn bộ population và đúng 24 offspring. Không cắt kết quả N28. Manual indices [20,40,80], gamma0.024; graph indices [40,60,90], tau [8,8,32].',
        '- Independent bank16+8 là control riêng: chọn top8 từ 24 ảnh không steering, cùng reference và initial source streams với A. Control này được phép dùng source cuối vì không chạy thêm guided suffix.',
        '- Tất cả cùng diffusion NFE nhưng chưa cùng chi phí verifier: BFS có 3 event ×24 và final24 =96 ImageReward evaluations; A và BoN có24. Một NFE là một UNet sample-step có CFG; số sample branches bằng2×NFE. Chưa kết luận về wall time hay tổng chi phí CPU.',
        '- Đây là follow-up trên các prompt đã đánh giá, không phải bộ test mới chưa từng xem. Không chọn cấu hình tốt nhất từ kết quả này.',
        '- Raw run mới: full/bfs/; nguồn cũ và SHA256: inherited_manifest.json. frozen_plan.json ghi protocol trước khi chạy; validation_audit.json ghi kiểm tra đầu ra.','']
    (out/'REPORT_VI.md').write_text('\n'.join(lines))
    print('ANALYSIS_COMPLETE',len(rows),len(summary),len(pairs),flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True)
    analyze(p.parse_args().output)


if __name__=='__main__': main()
