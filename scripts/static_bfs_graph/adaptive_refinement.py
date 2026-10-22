"""TRAIN-only adaptive temporal graph experiment: frozen definitions and CPU analysis."""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import json, hashlib, itertools, math, csv
import numpy as np
from .graph_builder import all_edges, all_paths, policy_from_path
from .policy import StaticBFSPolicy
from .runner import static_record, write_atomic
from .tpu_reference_compare import checked_inputs, records

TAUS=(2.,8.,32.)
G0=(20,40,60,80,90)
FINE=(10,20,30,40,50,60,70,80,90)
SEEDS={'A':[2026100801],'B':[2026100802,2026100803],'C':[2026100804,2026100805,2026100806,2026100807]}
SOURCE_FILES=['adaptive_refinement.py','adaptive_refinement_tpu.py','adaptive_refinement_driver.py','adaptive_refinement_report.py','adapter.py','runner.py','graph_builder.py','policy.py']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):write_atomic(Path(p),x)
def csvout(p,rows,fields=None):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def freeze(root):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'config.json').exists():verify(root);return
    pool,cfg,old=checked_inputs();repo=Path(__file__).resolve().parents[2]
    ids=pool['train_indices'];assert len(ids)==60 and set(map(int,pool['prompts']))==set(ids)
    ref=records()['DENSE_REFERENCE'];assert ref['resampling_steps']==cfg['pilot']['candidate_steps']
    config={'protocol':'ADAPTIVE_GRAPH_REFINEMENT_V1','N':8,'K':3,'steps':100,'eta':1.,'dtype':'BF16','Max':True,'SSP':True,'verifier':'ImageReward','taus':list(TAUS),'G0':list(G0),'G_FINE':list(FINE),'snapshots':list(range(5,100,5)),'seeds':SEEDS,'rounds':1,'graph_bootstraps':20000,'confirmation_bootstraps':50000,'sign_flips':100000,'audit_bootstraps':50000,'stat_seed':2026100810,'proposal_mean_threshold':.01,'proposal_bootstrap_positive_threshold':.8,'confirmation_mean_threshold':.01,'holm_alpha':.05,'confirmation_test_direction':'one-sided-positive','audit_noninferiority_margin':-.02,'diagnostic_internal_only':True,'validation_test_external_loaded':False,'snapshot_convention':'zero-based sampling_idx, post-update/post-real-reference-event; no extra search','success_ratio_includes_discovery_confirmation':True,'tpu_chips':4,'resource_policy':'finite frozen task DAG; stop on integrity/gate/resource failure; no automatic scientific changes'}
    split={'A':ids[:20],'B':ids[20:],'C':ids[-20:],'C_overlaps_B':True,'C_unseen_prompt_claim':False,'TRAIN_prompts':pool['prompts']}
    dump(root/'config.json',config);dump(root/'data_split.json',split);dump(root/'reference_schedule.json',ref)
    sources={str(p.relative_to(repo)):sha(p) for p in [repo/'scripts/static_bfs_graph'/n for n in SOURCE_FILES]+[repo/'scripts/diffusion_classical_search_t2i/tpu_runner.py',repo/'configs/static_bfs_graph_t2i.json',repo/'configs/diffusion_classical_search_t2i_table12.json',repo/'modal_n8_gpu/train_pool.json']}
    dump(root/'manifest.json',{'source_sha256':sources,'config_sha256':sha(root/'config.json'),'split_sha256':sha(root/'data_split.json'),'reference_sha256':sha(root/'reference_schedule.json'),'model_revision':old['models']['sd15']['revision']})
def verify(root):
    m=read(root/'manifest.json');repo=Path(__file__).resolve().parents[2]
    assert all(sha(repo/p)==h for p,h in m['source_sha256'].items()),'scientific/source hash changed'
    for name,key in [('config.json','config_sha256'),('data_split.json','split_sha256'),('reference_schedule.json','reference_sha256')]:assert sha(root/name)==m[key]
def edge_tuple(e):return(e.src,e.dst,e.tau_dst)
def edge_key(e):return e.key
def intervention(root,src,dst,tau,mid=None,mtau=None):
    ref=read(root/'reference_schedule.json');r=dict(ref['temperature_by_step'])
    steps=[s for s in ref['resampling_steps'] if s<=src or s>dst]
    if mid is not None:steps.append(mid);r[mid]=float(mtau)
    if dst!=100:steps.append(dst);r[dst]=float(tau)
    steps=tuple(sorted(set(steps)))
    return StaticBFSPolicy(steps,tuple((s,r[s]) for s in steps),8)
def mid(src,dst):
    m=int((Decimal(max(src,0)+dst)/Decimal(10)).quantize(Decimal('1'),rounding=ROUND_HALF_UP))*5
    return m if 5<=m<=95 and src<m<dst else None
def job(src,dst,tau,midpoint=None,mtau=None):
    return {'src':src,'dst':dst,'tau':tau,'mid':midpoint,'mtau':mtau}
def jobid(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()[:20]
def unitpath(root,cohort,pid,seed,j):return root/'observations'/cohort/f'p{pid}-s{seed}'/(jobid(j)+'.json')
def jobs_for(nodes):return [job(*edge_tuple(e)) for e in all_edges(tuple(nodes),TAUS)]
def ci(x,idx):return np.quantile(np.asarray(x)[idx].mean(axis=1),[.025,.975]).tolist()
def graph(root,nodes,label):
    split=read(root/'data_split.json');cfg=read(root/'config.json');ids=split['A'];seed=cfg['seeds']['A'][0]
    edges=all_edges(tuple(nodes),TAUS);vals=np.array([[read(unitpath(root,'A',pid,seed,job(*edge_tuple(e))))['regret'] for e in edges] for pid in ids])
    emap={e.key:i for i,e in enumerate(edges)};paths=[]
    for es in all_paths(tuple(nodes),TAUS,3):
        p=policy_from_path(es,8);paths.append((p.id,[emap[e.key] for e in es],static_record(p),[e.key for e in es]))
    assert len(paths)==math.comb(len(nodes),3)*27
    paths.sort(key=lambda p:p[0]);positions=np.array([p[1] for p in paths]);cost=vals.mean(axis=0)[positions].sum(axis=1);order=np.argsort(cost,kind='stable')
    rng=np.random.default_rng(cfg['stat_seed']);winners=np.zeros(len(paths),dtype=np.int64)
    for _ in range(cfg['graph_bootstraps']//100):
        idx=rng.integers(0,len(ids),size=(100,len(ids)));means=vals[idx].mean(axis=1);s=means[:,positions].sum(axis=2);winners+=np.bincount(np.argmin(s,axis=1),minlength=len(paths))
    probs=winners/cfg['graph_bootstraps'];top=[]
    for k in order[:16]:top.append({'policy':paths[k][2],'edge_keys':paths[k][3],'predicted_regret':float(cost[k]),'bootstrap_probability':float(probs[k])})
    frequencies={e.key:float(sum(t['bootstrap_probability'] for t in top if e.key in t['edge_keys'])) for e in edges}
    data={'nodes':list(nodes),'edges':[e.record()|{'mean_regret':float(vals[:,i].mean())} for i,e in enumerate(edges)],'logical_edges':len(edges),'edge_observations':len(edges)*20,'path_count':len(paths),'top1':top[0],'top16':top,'top16_mass':float(sum(t['bootstrap_probability'] for t in top)),'bootstrap_exact_top1_frequency':top[0]['bootstrap_probability'],'candidate_entropy':float(-sum(p*np.log(p) for p in probs if p>0)),'edge_top16_bootstrap_appearance':frequencies}
    dump(root/f'{label}_graph.json',data)
    dump(root/f'{label}_bootstrap.json',{'replicates':cfg['graph_bootstraps'],'prompt_count':20,'top16_mass':data['top16_mass'],'candidate_entropy':data['candidate_entropy'],'all_path_probabilities':[{'policy_id':p[0],'probability':float(probs[i])} for i,p in enumerate(paths)]})
    csvout(root/f'{label}_edge_values.csv',[{'prompt':pid,'edge_key':e.key,'regret':float(vals[k,i])} for k,pid in enumerate(ids) for i,e in enumerate(edges)])
    return data
def split_jobs(root,g):
    union={key for p in g['top16'] for key in p['edge_keys']}
    return [job(e['src'],e['dst'],e['tau_dst'],mid(e['src'],e['dst']),t) for e in g['edges'] if e['key'] in union and mid(e['src'],e['dst']) is not None for t in TAUS]
def propose(root,jobs,g):
    cfg=read(root/'config.json');ids=read(root/'data_split.json')['A'];seed=cfg['seeds']['A'][0];rng=np.random.default_rng(cfg['stat_seed']+1);idx=rng.integers(0,20,size=(20000,20));groups={}
    for j in jobs:groups.setdefault((j['src'],j['dst'],j['tau']),[]).append(j)
    out=[]
    for group,versions in sorted(groups.items(),key=lambda kv:json.dumps(kv[0])):
        scores=[]
        for j in versions:
            base=job(j['src'],j['dst'],j['tau']);x=np.array([read(unitpath(root,'A',p,seed,j))['reward']-read(unitpath(root,'A',p,seed,base))['reward'] for p in ids]);scores.append((float(x.mean()),j['mtau'],j,x))
        scores.sort(key=lambda x:(-x[0],x[1]));mean,_,j,x=scores[0];prob=float(np.mean(x[idx].mean(axis=1)>0));proposed=mean>=.01 or prob>=.8
        out.append({**j,'proposal_id':jobid(j),'mean_gain_A':mean,'probability_positive_A':prob,'proposed':proposed,'new_node':j['mid'] not in G0,'edge_bootstrap_appearance':next(e for e in g['edges'] if(e['src'],e['dst'],e['tau_dst'])==group)['key']})
        out[-1]['edge_bootstrap_appearance']=g['edge_top16_bootstrap_appearance'][out[-1]['edge_bootstrap_appearance']]
    dump(root/'midpoint_proposals.json',out);csvout(root/'midpoint_proposals.csv',out,fields=['src','dst','tau','mid','mtau','proposal_id','mean_gain_A','probability_positive_A','proposed','new_node','edge_bootstrap_appearance'])
    return [x for x in out if x['proposed']]
def confirm(root,proposals):
    cfg=read(root/'config.json');ids=read(root/'data_split.json')['B'];rng=np.random.default_rng(cfg['stat_seed']+2);idx=rng.integers(0,40,size=(50000,40));out=[]
    for j in proposals:
        base=job(j['src'],j['dst'],j['tau']);x=np.array([np.mean([read(unitpath(root,'B',p,s,job(j['src'],j['dst'],j['tau'],j['mid'],j['mtau'])))['reward']-read(unitpath(root,'B',p,s,base))['reward'] for s in cfg['seeds']['B']]) for p in ids]);mean=float(x.mean());hits=0
        for _ in range(100):hits+=int(np.sum(((rng.integers(0,2,size=(1000,40))*2-1)*x).mean(axis=1)>=mean))
        lo,hi=ci(x,idx);out.append({**j,'mean_gain_B':mean,'SE':float(x.std(ddof=1)/np.sqrt(40)),'CI_low':lo,'CI_high':hi,'p_one_sided':(hits+1)/100001,'wins':int(sum(x>0)),'ties':int(sum(x==0)),'losses':int(sum(x<0))})
    order=sorted(range(len(out)),key=lambda i:out[i]['p_one_sided']);last=0.
    for rank,i in enumerate(order):last=max(last,min(1.,out[i]['p_one_sided']*(len(out)-rank)));out[i]['p_Holm']=last
    for row in out:row['verdict']='CONFIRMED' if row['mean_gain_B']>=.01 and row['p_Holm']<.05 else 'BORDERLINE' if row['mean_gain_B']>=.01 else 'NO'
    dump(root/'midpoint_confirmation.json',out)
    fields=list(out[0]) if out else ['proposal_id','mean_gain_B','SE','CI_low','CI_high','p_one_sided','p_Holm','wins','ties','losses','verdict']
    csvout(root/'midpoint_confirmation.csv',out,fields)
    return sorted(set(G0)|{j['mid'] for j in out if j['verdict']=='CONFIRMED' and j['new_node']})
