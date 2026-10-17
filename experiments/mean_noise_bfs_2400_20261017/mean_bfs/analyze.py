"""Paired summaries: average seeds within prompt before descriptive bootstrap."""
from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image, ImageDraw
from otlab.io import atomic_json, load_tensor, file_sha


def contrast(rows, left, right, key):
    differences = []
    by_prompt = []
    for pid in sorted({r['prompt_id'] for r in rows}):
        r = [x for x in rows if x['prompt_id'] == pid]
        seeds = sorted({x['seed'] for x in r})
        ds = []
        for seed in seeds:
            a = next(x for x in r if x['seed']==seed and x['method']==left)
            b = next(x for x in r if x['seed']==seed and x['method']==right)
            ds.append(a[key]-b[key])
        assert len(ds) == 2
        mean = float(np.mean(ds))
        differences.append(mean)
        by_prompt.append(dict(prompt_id=pid, delta=mean, per_seed_deltas=ds))
    d = np.asarray(differences)
    rng = np.random.default_rng(2026101723)
    bootstrap = d[rng.integers(0,len(d),(20000,len(d)))].mean(axis=1)
    return dict(left=left,right=right,metric=key,delta=float(d.mean()),
        ci95=np.quantile(bootstrap,[.025,.975]).tolist(),prompt_rows=by_prompt,
        wins=int((d>1e-12).sum()),ties=int((np.abs(d)<=1e-12).sum()),losses=int((d < -1e-12).sum()),
        inference='descriptive six-TRAIN-prompt bootstrap after averaging two paired seeds; not confirmation')


def analyze(root, cfg):
    root = Path(root)
    rows, allocations, receipts = [], [], []
    inputs = {}
    for u in cfg['units']:
        cases = list(root.glob('chip*/'+u['unit_id']))
        assert len(cases)==1
        case = cases[0]
        receipt = json.loads((case/'COMPLETE.json').read_text())
        assert receipt['passed'] and receipt['unit']==u
        receipts.append(receipt)
        allocation = json.loads((case/'ALLOCATION.json').read_text())
        allocations.append(dict(unit_id=u['unit_id'],**allocation))
        assert allocation['shared_suffix9_bitwise']
        for name in cfg['methods']:
            full = json.loads((case/(name+'.json')).read_text())
            assert full['physical_cold_unet_sample_nfe']==2400 and full['auxiliary_unet_nfe']==0
            row = {k:full[k] for k in ['method','unit_id','prompt_id','seed','terminal_pool',
                'best_ir','pool_mean_ir','unique_terminal_root_ancestors']}
            for selector in ['top8_ir','qd8']:
                for metric in ['mean_ir','clip_diversity','dino_diversity','guided_count','scout_count']:
                    row[selector+'_'+metric] = full['views'][selector][metric]
            row['guided_incremental_best_ir'] = full.get('guided_incremental_best_ir')
            rows.append(row)
            inputs[str((case/(name+'.json')).relative_to(root))] = file_sha(case/(name+'.json'))
            assert file_sha(case/(name+'.pt'))==full['bundle_sha256']
    assert len(rows)==36
    summaries = []
    metrics=['best_ir','pool_mean_ir','top8_ir_mean_ir','top8_ir_clip_diversity',
        'top8_ir_dino_diversity','qd8_mean_ir','qd8_clip_diversity','top8_ir_guided_count',
        'top8_ir_scout_count','unique_terminal_root_ancestors']
    for name in cfg['methods']:
        subset=[r for r in rows if r['method']==name]
        assert len(subset)==12
        item=dict(method=name,units=12,prompts=6,seeds=2,cold_nfe=2400,
            terminal_pool=subset[0]['terminal_pool'],
            **{key:float(np.mean([r[key] for r in subset])) for key in metrics})
        if name != 'bfs_graph':
            item['mean_guided_incremental_best_ir'] = float(np.mean(
                [r['guided_incremental_best_ir'] for r in subset]))
        summaries.append(item)
    contrasts=[]
    for right in ['bfs_graph','allocation_uniform']:
        for key in ['best_ir','top8_ir_mean_ir','top8_ir_clip_diversity','qd8_mean_ir','qd8_clip_diversity']:
            contrasts.append(contrast(rows,'mean_gradient',right,key))
    active = sum(not a['replayed_same_proposals'] for a in allocations)
    rejected = sum(q['rejected'] for a in allocations for q in a['proposal_audit'])
    result=dict(methods=summaries,contrasts=contrasts,per_unit=rows,
        guided_different_from_uniform_units=active,total_units=12,cpu_rejected_proposals=rejected,
        scout_terminal_outcomes_always_available=True,final_pool_used=False,
        semantic_success='NOT_EVALUATED; IR and diversity are not semantic correctness',
        scope=cfg['scope'],analysis_input_hashes=inputs)
    atomic_json(root/'RESULTS.json',result)
    with (root/'per_unit.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (root/'summary.csv').open('w') as f:
        keys=list(dict.fromkeys(k for r in summaries for k in r))
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(summaries)
    with (root/'contrasts.csv').open('w') as f:
        keys=['left','right','metric','delta','ci95','wins','ties','losses']
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(contrasts)
    # Fixed prompt1/first seed, fixed top8 selector; never choose an illustrative winner.
    first=cfg['units'][0];case=next(root.glob('chip*/'+first['unit_id']))
    comparison=Image.new('RGB',(8*192,3*220),'white');draw=ImageDraw.Draw(comparison)
    for line,name in enumerate(cfg['methods']):
        data=load_tensor(case/(name+'.pt'))
        meta=json.loads((case/(name+'.json')).read_text())
        for col,i in enumerate(meta['views']['top8_ir']['indices']):
            image=Image.fromarray(data['rgb'][i].permute(1,2,0).numpy())
            x,y=col*192,line*220
            comparison.paste(image.resize((192,192)),(x,y))
            draw.text((x+3,y+194),f'{name} {float(data["rewards"][i]):.3f}',fill='black')
    comparison.save(root/'fixed_prompt1_seed2026101714_top8.jpg',quality=92)
    lines=['# Mean-gradient noise avoidance vs BFS — 2.400 NFE/prompt/seed',
        '', '**Đã hoàn tất pilot TRAIN ghép cặp: 6 prompt × 2 seed mới, mỗi recipe đúng 2.400 NFE kể cả scout, không padding, không auxiliary UNet.**',
        '', 'Đây là phép thử mới theo yêu cầu người dùng sau các diagnostic trước; radius 0.05 được chọn từ dữ liệu development đã xem. Không phải confirmation trên tập chưa đụng tới.',
        '', '| Method | Best IR/prompt | Mean IR top8 | CLIP diversity cùng top8 | Pool cuối | NFE |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['method']} | {r['best_ir']:.6f} | {r['top8_ir_mean_ir']:.6f} | {r['top8_ir_clip_diversity']:.6f} | {r['terminal_pool']} | 2400 |")
    lines+=['','## Paired contrasts', '', 'Lấy trung bình hai seed trong mỗi prompt rồi bootstrap 20.000 lần trên sáu prompt. Cỡ mẫu nhỏ, CI chỉ mô tả; các endpoint ngoài best IR là secondary, không chọn metric thuận lợi để tuyên bố thắng.',
        '', '| Mean-gradient trừ | Metric | Δ | CI95% | W/T/L prompt |','|---|---|---:|---|---|']
    for r in contrasts:
        lines.append(f"| {r['right']} | {r['metric']} | {r['delta']:+.6f} | [{r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}] | {r['wins']}/{r['ties']}/{r['losses']} |")
    lines+=['','## Ngân sách, công bằng và cách chạy',
        '', '- SD1.5/DDIM100 eta1 CFG7.5 BF16, sampler/UNet/VAE/ImageReward và schedule hash cũ giữ nguyên. Bốn chip TPU hiện có; không Modal/GPU thuê.',
        '- BFS dùng đúng graph/transfer N24: indices [40,60,90], tau [8,8,32], Max history khởi tạo zero, upstream SSP. Resample sau41/61/91 updates, x0 từ chính transition; không phát sinh preview UNet thêm. Đây không phải manual/paper schedule.',
        '- Mean: 12 root prefix40=480; root0 có18 scout suffix60=1080; ba guided suffix60=180; 11 root khác hoàn tất suffix60=660. Tổng2400, pool32. Tất cả scout và các no-op đều có quyền vào output; không giấu phí scout hoặc phí tạo root.',
        '- Ghép18 scout với2 suffix độc lập thành batch20; ba guided với9 suffix độc lập thành batch12. Prefix12 và BFS24 cũng chia hết microbatch4. Mỗi recipe khớp2400 cả logical lẫn physical sample NFE; mỗi sample NFE có hai CFG branches.',
        '- Hai bên cùng prompt, seed, initial_noise stream0 và historical noise trên root dùng chung. Mean chủ động chi phí vào học root0 và dùng12 roots; BFS dùng24 roots. Không gọi đây là cùng toàn bộ root population hoặc cùng trajectory sau resampling.',
        '- Uniform control dùng cùng prefix/scout/basis/radius và fresh suffix tapes, chỉ bỏ filter. Các phần dùng chung được thực thi một lần nhưng vẫn tính đầy đủ2400 cold NFE cho mỗi recipe. Chín suffix độc lập ở batch hai phải khớp bitwise.',
        '- Mean gradient fit riêng tại từng checkpoint, chỉ dùng reward scout A/B. q0 là sphere4D r=.05; dùng rejection sampling CPU đúng soft penalty và mixture25%, epsilon0, delta.05, scale.1, gamma1, floor.2. Không dùng reward guided/uniform hoặc BFS để đổi proposal; không chuyển gradient giữa roots/seeds.',
        '- Primary best IR trên toàn pool; secondary mean IR và CLIP/DINO diversity trên cùng bộ top8-IR. QD8 là selector phụ dùng chung đã khóa. Không lấy IR từ selector này ghép diversity của selector khác.',
        '- Chưa đánh giá semantic success, nên không đồng nhất IR cao hoặc CLIP diversity cao với good-distinct semantic coverage.',
        '', '## Filter có hoạt động thực tế không?',
        '', f'- Guided khác uniform ở {active}/12 prompt-seed units; tổng số CPU proposal bị từ chối là {rejected}. Không gọi đây là neural compute hoặc reward calls.',
        '- Mỗi anchor chỉ còn ba guided mẫu mới sau phí scout. Đây là giới hạn amortization thật của recipe đã khóa tại2400NFE, không phải free offline learning.',
        '', '## Integrity và thời gian',
        '', '- Single-chip smoke trước mở rộng: full100 vs checkpoint-resume và BFS continuation pass4/4 bitwise. Replay880NFE được tách riêng khỏi budget các method.',
        '- Actual neural work toàn campaign = 12×5520 +880 = 67120 sample NFE; CFG branch evaluations134240; padding0, auxiliary0. So cả ba recipe cold=86400; tiết kiệm thực thi do share prefix/scout giữa mean/uniform20160, rồi cộng replay880. Không miễn phí scout trong bảng method.',
        '- DINO/CLIP encode816 terminal records; ImageReward1680 scores gồm816 terminal +864 proxy. Các ảnh/candidate có thể lặp hoặc trùng, không xem là816 ảnh độc lập.',
        '- Thời gian và occupied-worker chip-hours được ghi trong COMPLETE.json. Bao gồm startup/compile/CPU scoring; không dùng campaign wall để tuyên bố method inference nhanh hơn. Actual billing chưa truy vấn.',
        '', '## Artifacts',
        '', '- [Config khóa](preregistered.json), [kết quả](RESULTS.json), [từng unit](per_unit.csv), [summary](summary.csv), [contrasts](contrasts.csv), [runtime/cost](COMPLETE.json).',
        '- [Ảnh top8 của prompt1/seed đầu đã khóa](fixed_prompt1_seed2026101714_top8.jpg). Chỉ là minh họa cố định, không chọn prompt mà một method thắng.',
        '- Mọi unit có ALLOCATION.json (gA/gB, proposals, RNG keys, scout roles, BFS events), tensor bundles, hashes và counter receipts. Chỉ TRAIN1–6, seeds2026101714/15; không dùng final/VALIDATION/TEST.',
        '', '**Không tự mở rộng campaign, retune hoặc nominate finalist từ pilot nhỏ này.**']
    (root/'REPORT_VI.md').write_text('\n'.join(lines)+'\n')
    return result
