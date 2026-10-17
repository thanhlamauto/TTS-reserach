"""Frozen finite-difference response models and empirical agreement weights."""
import numpy as np


def probe_actions(radius,prompt_id,seed):
    axes=np.eye(4)*radius
    nine=np.vstack([np.zeros((1,4)),np.stack([axes,-axes],axis=1).reshape(8,4)])
    rng=np.random.default_rng(np.random.SeedSequence([seed,prompt_id,8172]))
    dense=rng.normal(size=(4,4));dense/=np.linalg.norm(dense,axis=1,keepdims=True)
    mixed=np.vstack([np.zeros((1,4)),np.stack([radius*dense,-radius*dense],axis=1).reshape(8,4)])
    return np.vstack([nine,nine,mixed]),['A']*9+['B']*9+['C']*9


def fit(rewards,radius):
    y=np.asarray(rewards,dtype=float).reshape(3,9)
    g=(y[:2,1::2]-y[:2,2::2])/(2*radius)
    h=(y[:2,1::2]+y[:2,2::2]-2*y[:2,:1])/(radius**2)
    return dict(g=g.tolist(),h=h.tolist())


def predict(model,z,kind):
    z=np.asarray(z,dtype=float);g=np.asarray(model['g']);h=np.asarray(model['h'])
    p=z@g.T
    if kind=='quadratic':p+=.5*(z*z)@h.T
    elif kind!='linear':raise ValueError(kind)
    return p


def weights(predictions,epsilon,cfg,positive=False):
    p=np.asarray(predictions,dtype=float)
    if positive:
        mu=p.mean(axis=1);exponent=cfg['positive_beta']*(mu-mu.max())/cfg['reward_scale']
        w=np.exp(np.maximum(-700,exponent));upper=None
    else:
        upper=p.max(axis=1)+epsilon
        harm=np.maximum(0,(-cfg['delta_reward']-upper)/cfg['reward_scale'])
        w=np.exp(np.maximum(-700,-cfg['gamma']*harm*harm))
    w=np.maximum(cfg['weight_floor'],w)
    ratio=cfg['rho']+(1-cfg['rho'])*w/w.mean()
    if not np.isfinite(ratio).all() or not np.isclose(ratio.mean(),1):raise RuntimeError('invalid density ratio')
    return w,ratio,upper


def calibrate(units,cfg):
    result={}
    for kind in ['linear','quadratic']:
        residual=[]
        for u in units:
            y=np.asarray(u['rewards']);delta=y[19:]-y[18]
            p=predict(u['model'],np.asarray(u['actions'])[19:],kind)
            residual.extend((delta-p.max(axis=1)).tolist())
        epsilon=max(0.,float(np.quantile(residual,.9,method='higher')))
        result[kind]=dict(epsilon=epsilon,calibration_residuals=residual,observations=len(residual),rule=cfg['epsilon_rule'])
    return dict(models=result,calibration_prompt_ids=[u['prompt_id'] for u in units],screen_outcomes_used=False)


def evaluate(units,calibration,cfg):
    result={}
    for kind in ['linear','quadratic']:
        rng=np.random.default_rng(2026101711);rows=[];null=np.zeros(cfg['shuffled_replicates']);strong=[];good_hidden=[]
        for u in units:
            y=np.asarray(u['rewards']);actual=y[19:]-y[18];z=np.asarray(u['actions'])[19:]
            p=predict(u['model'],z,kind);epsilon=calibration['models'][kind]['epsilon']
            w,ratio,upper=weights(p,epsilon,cfg)
            wp,rp,_=weights(p,epsilon,cfg,positive=True)
            bad=actual < -cfg['delta_reward'];useful=actual>cfg['delta_reward'];supp=w<=cfg['gate']['strong_weight_max']
            if supp.any():strong.extend(bad[supp].tolist());good_hidden.extend(useful[supp].tolist())
            uniform=float(bad.mean());weighted=float(ratio@bad/8)
            shuffled=np.stack([rng.permutation(ratio) for _ in range(cfg['shuffled_replicates'])])
            null+=uniform-shuffled@bad/8
            rows.append(dict(prompt_id=u['prompt_id'],actual_delta=actual.tolist(),predictions=p.tolist(),optimistic_score=upper.tolist(),
                weights=w.tolist(),density_ratio=ratio.tolist(),uniform_harm_rate=uniform,weighted_harm_rate=weighted,
                positive_harm_rate=float(rp@bad/8),harm_rate_reduction=uniform-weighted,
                uniform_mean_delta=float(actual.mean()),weighted_mean_delta=float(ratio@actual/8),positive_mean_delta=float(rp@actual/8),
                strongly_suppressed=int(supp.sum()),suppressed_harm_precision=None if not supp.any() else float(bad[supp].mean()),
                predictor_mean_mae=float(np.abs(p.mean(axis=1)-actual).mean()),
                empirical_optimistic_undercoverage=float(np.mean(actual>upper)),
                finite8_importance_ess=float(64/(ratio@ratio)),action_norms=np.linalg.norm(z,axis=1).tolist(),
                signs_agree_A_B=int(np.sum(np.sign(p[:,0])==np.sign(p[:,1])))))
        null/=len(rows);observed=float(np.mean([r['harm_rate_reduction'] for r in rows]));precision=None if not strong else float(np.mean(strong))
        tail=float((1+np.sum(null>=observed-1e-15))/(len(null)+1))
        gate=cfg['gate'];checks=dict(enough_suppression=len(strong)>=gate['min_strongly_suppressed'],
            harm_precision=precision is not None and precision>=gate['min_harm_precision'],
            practical_reduction=observed>=gate['min_mean_harm_rate_reduction'],shuffled_comparison=tail<=gate['max_shuffled_tail_fraction'],
            prompt_consistency=sum(r['harm_rate_reduction']>=0 for r in rows)>=gate['min_nonnegative_prompts'])
        result[kind]=dict(rows=rows,mean_harm_rate_reduction=observed,strongly_suppressed=len(strong),harm_precision=precision,
            suppressed_useful_fraction=None if not good_hidden else float(np.mean(good_hidden)),shuffled_tail_fraction=tail,
            shuffled_reduction_quantiles=np.quantile(null,[.025,.5,.975]).tolist(),gate_checks=checks,gate_passed=all(checks.values()),
            inference='three-anchor exploratory finite-action diagnostic; permutation tail is not population-level confirmation')
    passing=[k for k in result if result[k]['gate_passed']]
    winner=max(passing,key=lambda k:(result[k]['mean_harm_rate_reduction'],k=='linear')) if passing else None
    return dict(models=result,stage1_gate_passed=bool(passing),model_for_future_mechanism=winner,
        decision='READY_FOR_SEPARATE_MECHANISM_TEST' if passing else 'STOP_NO_SUPPORTED_LOCAL_FILTER',
        actual_post_scouting_guided_batch_run=False,equal_budget_competition_run=False,final_confirmation_run=False)
