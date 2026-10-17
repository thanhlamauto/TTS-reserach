"""Export every R2 direct terminal outcome for blinded semantic audit; no neural work."""
import argparse,json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image,ImageDraw
from otlab.io import load_tensor,tensor_sha,file_sha,atomic_json,digest
from ot_next.output_selection import cosine_distance

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.protocol).read_text());source=Path(a.protocol).resolve().parent
    root=Path(cfg['root']);root.mkdir(parents=True,exist_ok=True)
    if (root/'EXPORT_COMPLETE.json').exists():return
    start=time.time();old=Path(cfg['old_r2_root']);images={};observations=[];pool_metrics={};inputs={}
    prior=json.loads((source/'prior_r2_annotations.json').read_text())
    oldindex=json.loads((source/'prior_r2_index.json').read_text())
    lookup={(r['prompt_id'],r['sha']):prior['images'][r['id']] for r in oldindex}
    for u in cfg['units']:
        case=next(old.glob(f'chip*/p{u["prompt_id"]:03d}'))
        for name in cfg['methods']:
            path=case/(name+'.pt');inputs[str(path)]=file_sha(path);d=load_tensor(path)
            n=len(d['rgb']);assert n==(24 if name=='bon24' else 32)
            distances=cosine_distance(d['clip'].numpy());dino=cosine_distance(d['dino'].numpy())
            key=f'p{u["prompt_id"]:03d}/{name}'
            pool_metrics[key]=dict(clip_distance=distances.tolist(),dino_distance=dino.tolist())
            for i in range(n):
                h=tensor_sha(d['rgb'][i]);k=(u['prompt_id'],h);images.setdefault(k,d['rgb'][i])
                observations.append(dict(prompt_id=u['prompt_id'],method=name,slot=i,sha=h,
                    root=d['root_ids'][i],level=d['levels'][i],reward=float(d['rewards'][i])))
    ordered=sorted(images,key=lambda k:digest(dict(seed=2026101711,prompt=k[0],pixels=k[1])))
    ids={k:f'L{i:04d}' for i,k in enumerate(ordered)}
    reused={ids[k]:lookup[k] for k in ordered if k in lookup}
    review=root/'blinded_review';review.mkdir();full=root/'full512';full.mkdir()
    for u in cfg['units']:
        keys=[k for k in ordered if k[0]==u['prompt_id'] and k not in lookup]
        for page in range((len(keys)+35)//36):
            subset=keys[36*page:36*(page+1)]
            sheet=Image.new('RGB',(1152,214*((len(subset)+5)//6)),'white');draw=ImageDraw.Draw(sheet)
            for j,k in enumerate(subset):
                im=Image.fromarray(images[k].permute(1,2,0).numpy());im.save(full/(ids[k]+'.png'))
                x,y=j%6*192,j//6*214;sheet.paste(im.resize((192,192)),(x,y));draw.text((x+4,y+195),ids[k],fill='black')
            sheet.save(review/f'p{u["prompt_id"]:03d}_{page:02d}.jpg',quality=92)
    for r in observations:r['image_id']=ids[(r['prompt_id'],r.pop('sha'))]
    atomic_json(root/'OBSERVATIONS_HIDDEN_UNTIL_LABEL_FREEZE.json',observations)
    atomic_json(root/'POOL_DISTANCES_HIDDEN_UNTIL_LABEL_FREEZE.json',pool_metrics)
    atomic_json(root/'BLINDED_IMAGE_INDEX.json',[dict(id=ids[k],prompt_id=k[0],sha=k[1],reused=k in lookup) for k in ordered])
    atomic_json(root/'REUSED_FROZEN_LABELS.json',reused)
    atomic_json(root/'INPUT_HASHES.json',inputs)
    atomic_json(root/'preregistered.json',cfg)
    result=dict(passed=True,status='EXPORTED_PENDING_FULL_POOL_BLINDED_ANNOTATION',observations=len(observations),
        unique_images=len(images),reused_frozen_labels=len(reused),new_review_images=len(images)-len(reused),
        neural_nfe=0,elapsed_seconds=time.time()-start,prior_annotation_sha256=file_sha(source/'prior_r2_annotations.json'),
        limitations=['Codex visual annotation is descriptive, not validated human semantic ground truth',
                    'Small adaptively allocated per-node denominators; raw conditional rates are not an unbiased policy-level population estimator',
                    'One root terminal is a direct base continuation for each of its20/40 original nodes; descendants are not backed up into other nodes'])
    atomic_json(root/'EXPORT_COMPLETE.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
