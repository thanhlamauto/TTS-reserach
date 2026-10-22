from . import adaptive_refinement_tpu as backend
from . import seed_coverage_v2 as s
from pathlib import Path
import argparse,itertools,time,traceback
backend.verify=s.verify

def worker(root,stage,chip):
 eng=backend.Engine(root,chip,stage)
 try:
  if stage=='gates':eng.gates()
  elif stage=='audit':
   pol=s.read(root/'frozen_policies.json');assert s.sha(root/'frozen_policies.json')==s.read(root/'freeze_receipt.json')['sha256'];unique={p['id']:p for p in pol.values()};units=list(itertools.product(eng.split['C'],eng.cfg['seeds']['C']))
   for pid,seed in units[chip::4]:
    for p in unique.values():
     path=root/'audit'/f'p{pid}-s{seed}-{p["id"]}.json'
     if path.exists():assert s.read(path)['policy']==p;continue
     start=time.time();row,_=eng.direct(pid,seed,p);assert len(row['events'])==3;s.dump(path,{'prompt':pid,'seed':seed,'policy':p,'runtime_seconds':time.time()-start,**row});eng.persist()
  else:
   req=s.read(root/'tasks'/f'{stage}.json')
   for pid,seed in req['units'][chip::4]:eng.profile('A',pid,seed,req['jobs'])
  eng.persist();s.dump(root/'workers'/stage/f'chip{chip}.done.json',{'success':True,'account':eng.account})
 except BaseException as e:eng.persist();s.dump(root/'failures'/f'{stage}-chip{chip}.json',{'exception':repr(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--stage',required=True);p.add_argument('--chip',type=int,required=True);a=p.parse_args();worker(a.run_root.resolve(),a.stage,a.chip)
