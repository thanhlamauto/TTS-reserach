"""Complete exactly the ten missing P1 edits, preserving original evidence."""
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

from otlab.io import atomic_json, load_tensor, tensor_sha
from otlab.noise import NoiseTape
from otlab.target import cluster


def rank(values):
    x = np.asarray(values)
    result = np.zeros(len(x), dtype=float)
    order = np.argsort(x, kind='stable')
    for value in np.unique(x):
        positions = np.where(x[order] == value)[0]
        result[order[positions]] = positions.mean()
    return result


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def mean_diversity(features):
    f = torch.nn.functional.normalize(features.double(), dim=-1)
    d = 1 - f @ f.T
    return float(d[~torch.eye(len(f), dtype=torch.bool)].mean())


def contact_sheet(rgb, labels, path, columns=8):
    w, h = 160, 186
    sheet = Image.new('RGB', (w*columns, h*((len(rgb)+columns-1)//columns)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (x, label) in enumerate(zip(rgb, labels)):
        im = Image.fromarray(x.permute(1, 2, 0).numpy()).resize((160,160))
        left, top = (i%columns)*w, (i//columns)*h
        sheet.paste(im, (left,top))
        draw.text((left+3, top+161), label, fill='black')
    sheet.save(path)


def execute(adapter, features, config, output, status):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    old = Path(config['old_root'])
    attempt = old / 'attempts/8074f94c847d'
    proposed = load_tensor(old / 'cache/27ccdb0565a3b62ee011f2176dca25cc791acc1cf9c27d5109a4eacb0bfc2788.pt')
    preview = load_tensor(old / 'cache/3da294839caa5bf5898857046d97ca310e9cdeb45808ee71fb7281bd34a3ee7f.pt')
    tape = NoiseTape.from_bundle(load_tensor(attempt / 'export/noise.pt'))
    cp = load_tensor(attempt / 'export/checkpoint.pt')
    rows = [json.loads(line) for line in (attempt / 'results/rows.jsonl').read_text().splitlines()]
    old_rows = {r['method']: r for r in rows}
    none = load_tensor(attempt / 'results' / old_rows['none']['job_id'] / 'terminal.pt')
    chosen = load_tensor(attempt / 'results' / old_rows['p1_ot']['job_id'] / 'terminal.pt')
    choices = old_rows['p1_ot']['logs']['selection']['choices']
    assert preview['next_index'] == 54 and tape.start_index == 50
    assert torch.equal(preview['active'], proposed['active']) and bool(preview['active'].all())
    assert len(choices) == 8 and sum(c != 0 for c in choices) == 6
    assert cp['meta']['schedule_sha'] == tape.schedule_sha == adapter.schedule_sha
    if adapter.identity != json.loads((attempt/'replay.json').read_text())['records'][0]['adapter_identity']:
        raise RuntimeError('old/current sampler or reward prompt identity mismatch')
    before = dict(adapter.counters)
    status('E0_exact_replay_gates')
    # Replay each original8-image population with the original row order. The
    # earlier14-image parity gate changed the CPU verifier batch geometry;
    # exact reward equality must use its original8-image call contract.
    ids = list(range(8))*2
    slots = [0]*8 + choices
    entry = preview['states'][ids, slots]
    replay = adapter.advance(entry, 54, 100, ids, tape)
    expected = torch.cat([none['latent'], chosen['latent']])
    if not torch.equal(replay, expected):
        raise RuntimeError('index54 gate failed; missing edit generation forbidden')
    scored=[adapter.score(replay[start:start+8],'replay') for start in (0,8)]
    replay_reward=torch.cat([r for r,im in scored])
    replay_rgb=torch.cat([im for r,im in scored])
    expected_rewards = torch.cat([none['rewards'], chosen['rewards']])
    expected_rgb = torch.cat([none['rgb'], chosen['rgb']])
    parity=dict(latent_bitwise=True,pixels_bitwise=torch.equal(replay_rgb,expected_rgb),
                rewards_bitwise=torch.equal(replay_reward,expected_rewards),
                reward_max_abs=float((replay_reward-expected_rewards).abs().max()),
                pixel_max_abs=int((replay_rgb.int()-expected_rgb.int()).abs().max()),
                gate_suffix_nfe=16*46,verifier_batch_shape=8)
    atomic_json(output/'replay_partial.json',parity)
    if not parity['rewards_bitwise'] or not parity['pixels_bitwise']:
        raise RuntimeError('old reward/pixel parity failed')
    atomic_json(output/'replay.json', dict(passed=True, noops=8, selected_edits=6, exact_pixels_rewards=True,
                                         gate_suffix_nfe=16*46, replay_populations=2,verifier_batch_shape=8,
                                         endpoint_sha=tensor_sha(replay)))
    terminal_latent = torch.zeros_like(preview['states'])
    terminal_rgb = torch.zeros_like(preview['rgb'])
    terminal_reward = torch.zeros_like(preview['proxy_reward'])
    terminal_latent[:,0], terminal_rgb[:,0], terminal_reward[:,0] = none['latent'], none['rgb'], none['rewards']
    for i,c in enumerate(choices):
        if c:
            terminal_latent[i,c], terminal_rgb[i,c], terminal_reward[i,c] = chosen['latent'][i], chosen['rgb'][i], chosen['rewards'][i]
    missing = [(i,c) for i in range(8) for c in (1,2) if choices[i] != c]
    assert len(missing) == 10
    status('E0_missing_edits', count=10, suffix_nfe=460)
    states = torch.stack([preview['states'][i,c] for i,c in missing])
    final = adapter.advance(states, 54, 100, [i for i,c in missing], tape)
    rewards, rgb = adapter.score(final, 'terminal')
    for j,(i,c) in enumerate(missing):
        terminal_latent[i,c], terminal_rgb[i,c], terminal_reward[i,c] = final[j], rgb[j], rewards[j]
    torch.save(dict(latent=terminal_latent, rgb=terminal_rgb, rewards=terminal_reward, missing=missing), output/'all_proposals.pt')
    if not torch.isfinite(terminal_latent).all() or not torch.isfinite(terminal_reward).all():
        raise RuntimeError('nonfinite proposal outcome')
    status('E0_terminal_feature_evaluation')
    preview_f = features.encode(preview['rgb'].flatten(0,1), 'old_preview')
    terminal_f = features.encode(terminal_rgb.flatten(0,1), 'all_terminal')
    labels = cluster(preview_f).reshape(8,3)
    old_labels = torch.tensor(old_rows['p1_ot']['logs']['selection']['labels'])
    if not torch.equal(labels, old_labels):
        raise RuntimeError('frozen DINO preview labels no longer match old selector')
    centers = torch.stack([torch.nn.functional.normalize(preview_f[labels.flatten()==g].double().mean(0), dim=0)
                           for g in sorted(labels.unique().tolist())])
    terminal_labels = (1 - terminal_f.double() @ centers.T).argmin(1).reshape(8,3)
    # Independent supplementary CLIP metric uses the pinned legacy evaluator.
    from sd15_sinkhorn_tpu.reward import CLIPCPU
    clip = CLIPCPU(adapter.sampler.cfg, config['dcs_root'])
    ims = [Image.fromarray(x.permute(1,2,0).numpy()) for x in terminal_rgb.flatten(0,1)]
    clip_result = clip.score(adapter.prompt, ims)
    clip_f = torch.tensor(clip_result['clip_embeddings'])
    preview_gain = preview['proxy_reward'][:,1:] - preview['proxy_reward'][:,:1]
    terminal_gain = terminal_reward[:,1:] - terminal_reward[:,:1]
    source_oracle = terminal_reward.argmax(1)
    free_ids = torch.argsort(terminal_reward.flatten(), descending=True, stable=True)[:8]
    chosen_ids = torch.arange(8)*3 + torch.tensor(choices)
    source_ids = torch.arange(8)*3 + source_oracle
    banks = old/'bank_train_seed20261016/p000/reference/images'
    bank_rgb = torch.stack([torch.from_numpy(np.asarray(Image.open(f).convert('RGB')).copy()).permute(2,0,1)
                            for f in sorted(banks.glob('*.png'))])
    if len(bank_rgb) != 16:
        raise RuntimeError('actual terminal bank missing')
    bank_features = features.encode(bank_rgb, 'old_bank_terminal')
    bank_reward = cp['bank_reward']
    def view(indices):
        r = terminal_reward.flatten()[indices]
        joined = torch.cat([bank_reward, r])
        top = torch.argsort(joined, descending=True, stable=True)[:8]
        return dict(mean_ir=float(r.mean()), best_ir=float(r.max()),
                    dino_diversity=mean_diversity(terminal_f[indices]),
                    clip_diversity=mean_diversity(clip_f[indices]),
                    pool_top8_ir=float(joined[top].mean()),
                    unique_sources=int(torch.unique(indices//3).numel()), indices=indices.tolist())
    # All3^8 policies: reward/diversity frontier is diagnostic, not deployable.
    combos = np.array(list(itertools.product(range(3), repeat=8)))
    flat = np.arange(8)[None,:]*3 + combos
    rr = terminal_reward.flatten().double().numpy()[flat].mean(1)
    dino_dist = 1 - terminal_f.double().numpy() @ terminal_f.double().numpy().T
    clip_dist = 1 - clip_f.double().numpy() @ clip_f.double().numpy().T
    dino_div = dino_dist[flat[:,:,None], flat[:,None,:]].sum((1,2))/56
    clip_div = clip_dist[flat[:,:,None], flat[:,None,:]].sum((1,2))/56
    order = np.argsort(-rr, kind='stable')
    best_d, frontier = -1., []
    for j in order:
        if clip_div[j] > best_d + 1e-12:
            best_d = float(clip_div[j])
            frontier.append(int(j))
    with (output/'source_oracle_frontier.csv').open('w') as stream:
        stream.write('combination,mean_ir,dino_diversity,clip_diversity,clip_pareto\n')
        frontier_set = set(frontier)
        for j in range(len(combos)):
            stream.write(f'{"".join(map(str,combos[j]))},{rr[j]},{dino_div[j]},{clip_div[j]},{int(j in frontier_set)}\n')
    delta = terminal_gain.flatten().double().numpy()
    proxy = preview_gain.flatten().double().numpy()
    feasible = torch.tensor(old_rows['p1_ot']['logs']['selection']['feasible'])
    summary = dict(prompts=1, seed=20261016, inference='single TRAIN engineering diagnostic; oracle not inference',
                   actual_next_index=54, new_missing_suffix_nfe=460, auxiliary_unet_nfe=0,
                   new_counters={k:v-before.get(k,0) for k,v in adapter.counters.items()},
                   valid_mask=preview['active'].tolist(), feasible_mask=feasible.tolist(),
                   no_op_reason=[['explicit_noop',None,None] for _ in range(8)],
                   preview_labels=labels.tolist(), terminal_labels_from_frozen_preview_centers=terminal_labels.tolist(),
                   chi_active=float(np.mean([len(set(row))>1 for row in labels.tolist()])),
                   chi_feasible=float(np.mean([len(set(labels[i][feasible[i]].tolist()))>1 for i in range(8)])),
                   group_label_retention=float((terminal_labels==labels).float().mean()),
                   preview_terminal_pearson=correlation(proxy,delta),
                   preview_terminal_spearman=correlation(rank(proxy),rank(delta)),
                   positive_preview_but_negative_terminal=int(((proxy>0)&(delta<0)).sum()),
                   edited_proposals=16, terminal_gain_per_edit=terminal_gain.tolist(),
                   preview_gain_per_edit=preview_gain.tolist(),
                   views=dict(none=view(torch.arange(8)*3), selected=view(chosen_ids),
                              source_oracle=view(source_ids), unconstrained_oracle=view(free_ids)),
                   source_oracle_combinations=6561, semantic_success='PENDING_BLINDED_VISUAL_REVIEW',
                   primary_good_distinct='NOT_EVALUATED_THRESHOLDS_UNSET')
    torch.save(dict(preview_features=preview_f, terminal_features=terminal_f, bank_features=bank_features,
                    clip_terminal_features=clip_f, preview_centers=centers), output/'features.pt')
    atomic_json(output/'SUMMARY.json',summary)
    contact_sheet(terminal_rgb.permute(1,0,2,3,4).flatten(0,1),
                  [f's{i} p{c} IR {float(terminal_reward[i,c]):.2f}' for c in range(3) for i in range(8)],
                  output/'all_proposals.png')
    contact_sheet(bank_rgb, [f'bank{i} IR {float(bank_reward[i]):.2f}' for i in range(16)], output/'bank.png')
    return summary
