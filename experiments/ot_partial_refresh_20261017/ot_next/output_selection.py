"""Common output selection; exact good-distinct count on at most8 outputs."""
import numpy as np


def cosine_distance(features):
    f=np.asarray(features,dtype=np.float64)
    f=f/np.linalg.norm(f,axis=1,keepdims=True).clip(1e-12)
    return np.maximum(0.,1-f@f.T)


def quality_diversity_indices(rewards,features,quality_threshold,count=8):
    r=np.asarray(rewards,dtype=float)
    d=cosine_distance(features)
    ranked=np.argsort(-r,kind='stable').tolist()
    if len(r)<count or not np.isfinite(r).all():raise ValueError('invalid candidate pool')
    eligible=[i for i in ranked if r[i]>=quality_threshold]
    chosen=[]
    if eligible:
        chosen=[eligible.pop(0)]
    while eligible and len(chosen)<count:
        # Same DINO geometry and IR eligibility for EVERY method. No semantic
        # labels and no supplementary CLIP evaluator enter final selection.
        selected=min(eligible,key=lambda i:(-float(d[i,chosen].min()),-r[i],i))
        chosen.append(selected)
        eligible.remove(selected)
    chosen.extend([i for i in ranked if i not in chosen][:count-len(chosen)])
    assert len(chosen)==count and len(set(chosen))==count
    return chosen


def good_distinct(rewards,distance,semantic,quality_threshold,distance_threshold,unknown_as_success=False):
    r=np.asarray(rewards,dtype=float)
    d=np.asarray(distance,dtype=float)
    if len(r)>8 or d.shape!=(len(r),len(r)) or len(semantic)!=len(r):raise ValueError('at most8 actual outputs')
    eligible=[r[i]>=quality_threshold and (semantic[i] is True or (semantic[i] is None and unknown_as_success)) for i in range(len(r))]
    best=[]
    for mask in range(1<<len(r)):
        ids=[i for i in range(len(r)) if mask>>i&1]
        if len(ids)<=len(best) or not all(eligible[i] for i in ids):continue
        if all(d[i,j]>=distance_threshold for pos,i in enumerate(ids) for j in ids[pos+1:]):best=ids
    return dict(count=len(best),indices=best)
