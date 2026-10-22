"""Uses unchanged, gate-tested live-XLA official-loop replay engine."""
from . import adaptive_refinement_tpu as backend
from . import sequential_refinement as sr
import itertools,time,os,argparse,traceback
from pathlib import Path
backend.verify=sr.verify

def worker(root,stage,chip):
    eng=backend.Engine(root,chip,stage)
    try:
        if stage=='gates':eng.gates()
        elif stage=='audit':
            frozen=sr.read(root/'frozen_policies.json');assert sr.sha(root/'frozen_policies.json')==sr.read(root/'policy_freeze_receipt.json')['sha256']
            unique={p['id']:p for p in frozen.values()};units=list(itertools.product(eng.split['C'],eng.cfg['seeds']['C']))
            for pid,seed in units[chip::4]:
                for policy in unique.values():
                    path=root/'audit'/f'p{pid}-s{seed}-{policy["id"]}.json'
                    if path.exists():assert sr.read(path)['policy']==policy;continue
                    start=time.time();row,_=eng.direct(pid,seed,policy);assert len(row['events'])==3
                    sr.dump(path,{'prompt':pid,'seed':seed,'policy':policy,'runtime_seconds':time.time()-start,**row});eng.persist()
        else:
            req=sr.read(root/'tasks'/f'{stage}.json');units=list(itertools.product(req['ids'],eng.cfg['seeds'][req['cohort']]))
            for pid,seed in units[chip::4]:eng.profile(req['cohort'],pid,seed,req['jobs'])
        eng.persist();sr.dump(root/'workers'/stage/f'chip{chip}.done.json',{'success':True,'account':eng.account})
    except BaseException as e:
        eng.persist();sr.dump(root/'failures'/f'{stage}-chip{chip}.json',{'exception':repr(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--stage',required=True);p.add_argument('--chip',type=int,required=True);a=p.parse_args();worker(a.run_root,a.stage,a.chip)
