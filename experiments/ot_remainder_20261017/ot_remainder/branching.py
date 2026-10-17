"""Finite original-checkpoint action set, with feedback only after a batch."""
import numpy as np
from ot_next.allocation import softmax, residual_resample, semi_relaxed, grouping


def anchors(features, rewards, threshold, merge_distance):
    eligible=np.flatnonzero(np.asarray(rewards)>=threshold)
    if not len(eligible):return None
    eligible=eligible[np.argsort(-np.asarray(rewards)[eligible],kind='stable')]
    g=grouping(features,eligible,[],merge_distance)
    return np.asarray(features)[g['medoids']].copy()


def allocation(method, level, batch_size, archive, centers, tau, uniforms):
    # A record may inform both original nodes only for an initial full root.
    n=np.zeros(8);utility=np.zeros(8)
    groups=0 if centers is None else len(centers)
    hits=np.zeros((8,groups));occupancy=np.zeros(groups)
    for record in archive:
        good=record['reward']>=tau
        group=None if centers is None else int(np.argmax(centers@record['feature']))
        if good and group is not None:occupancy[group]+=1
        if level not in record['direct_levels']:continue
        i=record['root'];n[i]+=1
        utility[i]+=1/(1+np.exp(-np.clip(record['reward']-tau,-30,30)))
        if good and group is not None:hits[i,group]+=1
    if np.any(n<1):raise ValueError('every original action requires its root direct rollout')
    mean=utility/n
    ucb=mean+np.sqrt(2*np.log(1+n.sum())/n)
    # A fixed softmax of bounded reward UCB, rather than an after-results fit.
    reward_weights=softmax(ucb)
    cost=None;alpha=None;coupling=None
    if method=='uniform' or centers is None:
        weights=np.full(8,1/8)
    elif method=='reward_ucb':weights=reward_weights
    else:
        p=(hits+.5)/(n[:,None]+.5*(groups+1))
        demand=np.maximum((len(archive)+batch_size)/groups-occupancy,0)
        alpha=demand/demand.sum()
        cost=-np.log(p.T+1e-12)
        if method=='action_ot':
            coupling,weights=semi_relaxed(alpha,cost,ucb,gamma=1.,epsilon=1.)
        elif method=='group':
            hard=p.argmax(1);weights=np.zeros(8)
            for g,mass in enumerate(alpha):
                members=np.flatnonzero(hard==g)
                if not len(members):members=np.arange(8)
                weights[members]+=mass*softmax(ucb[members])
        else:raise ValueError(method)
    # Exactly one of4 or two of8 slots explore, same rule in every method.
    explore=batch_size//4
    exploration=np.minimum((np.asarray(uniforms[:explore])*8).astype(int),7)
    chosen,counts=residual_resample(weights,batch_size-explore,uniforms[explore:])
    parents=np.concatenate([exploration,chosen]).tolist()
    return parents,dict(level=level,n_direct=n.tolist(),reward_utility_mean=mean.tolist(),
        ucb=ucb.tolist(),weights=weights.tolist(),offspring_counts=np.bincount(parents,minlength=8).tolist(),
        exploration_slots=explore,exploration_parents=exploration.tolist(),
        hit_counts=hits.tolist(),occupancy=occupancy.tolist(),alpha=None if alpha is None else alpha.tolist(),
        outcome_cost=None if cost is None else cost.tolist(),
        coupling=None if coupling is None else coupling.tolist(),
        archive_size_before_decision=len(archive),online_good='IR threshold only; semantic labels not available online')
