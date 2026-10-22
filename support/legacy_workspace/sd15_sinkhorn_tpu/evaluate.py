import numpy as np


def summary_metrics(scored,paired=None,indices=None):
    r=np.asarray(scored['rewards'],dtype='float64')
    e=np.asarray(scored['clip_embeddings'],dtype='float64')
    a=np.asarray(scored['clip_alignment'],dtype='float64')
    if indices is not None: r,e,a=r[indices],e[indices],a[indices]
    distances=1-e@e.T
    diversity=float(distances[np.triu_indices(len(e),1)].mean()) if len(e)>1 else 0.
    result=dict(mean_reward=float(r.mean()),best_reward=float(r.max()),clip_alignment=float(a.mean()),
                clip_diversity=diversity,n_images=len(r))
    if paired is not None:
        gain=r-np.asarray(paired)
        result.update(reward_gain=float(gain.mean()),paired_improved_fraction=float((gain>0).mean()))
    return result


if __name__=='__main__':
    from .analyze import main
    main()
