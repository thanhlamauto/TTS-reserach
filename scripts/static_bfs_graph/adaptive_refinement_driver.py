"""Detached finite stage supervisor. No reward-dependent early stopping."""
from pathlib import Path
import argparse, fcntl, os, sys, time, subprocess, traceback, shutil
from .adaptive_refinement import *

def stage(root,name,cohort=None,jobs=None):
    verify(root)
    if (root/'stage_ledger'/f'{name}.json').exists():return
    if shutil.disk_usage(root).free < 5*1024**3:raise RuntimeError('resource stop: less than5GiB disk free')
    if cohort is not None:dump(root/'tasks'/f'{name}.json',{'cohort':cohort,'jobs':jobs})
    chips=[0] if name=='gates' else list(range(4));processes=[];logs=[];started=time.time()
    dump(root/'status.json',{'stage':name,'status':'RUNNING','start_unix':started})
    for chip in chips:
        p=root/'logs'/f'{name}-chip{chip}.log';p.parent.mkdir(parents=True,exist_ok=True);log=p.open('a');logs.append(log)
        env={**os.environ,'PJRT_DEVICE':'TPU','TPU_VISIBLE_CHIPS':str(chip),'PYTHONUNBUFFERED':'1','STATIC_BFS_CACHE_ROOT':f'/dev/shm/static-bfs-adaptive-refinement-v1/chip{chip}/cache'}
        processes.append(subprocess.Popen([sys.executable,'-m','scripts.static_bfs_graph.adaptive_refinement_tpu','--run-root',str(root),'--stage',name,'--chip',str(chip)],env=env,stdout=log,stderr=subprocess.STDOUT))
    dump(root/'active_workers.json',{'stage':name,'PIDs':[p.pid for p in processes]})
    try:
        while True:
            states=[p.poll() for p in processes]
            if any(s is not None and s!=0 for s in states):raise RuntimeError(f'{name} worker failed: {states}')
            if all(s==0 for s in states):break
            time.sleep(5)
        for chip in chips:assert read(root/'workers'/name/f'chip{chip}.done.json')['success']
        if name=='gates':assert read(root/'gates.json')['passed']
        dump(root/'stage_ledger'/f'{name}.json',{'complete':True,'wall_seconds':time.time()-started,'chips':chips})
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            if p.poll() is None:
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in logs:log.close()

def run(root):
    root.mkdir(parents=True,exist_ok=True);lock=(root/'driver.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dump(root/'driver.pid.json',{'pid':os.getpid(),'started_unix':time.time()})
    try:
        freeze(root);stage(root,'gates')
        stage(root,'coarse','A',jobs_for(G0));g0=graph(root,G0,'coarse')
        sj=split_jobs(root,g0);stage(root,'split_discovery','A',sj);proposals=propose(root,sj,g0)
        dump(root/'frozen_proposals.json',{'proposals':proposals,'proposal_sha256':sha(root/'midpoint_proposals.json'),'frozen_before_B':True})
        bj={jobid(j):j for p in proposals for j in [job(p['src'],p['dst'],p['tau']),job(p['src'],p['dst'],p['tau'],p['mid'],p['mtau'])]}
        if bj:stage(root,'confirmation','B',list(bj.values()))
        nodes=confirm(root,proposals);dump(root/'frozen_refinement_nodes.json',{'nodes':nodes,'confirmation_sha256':sha(root/'midpoint_confirmation.json')})
        new=[j for j in jobs_for(nodes) if jobid(j) not in {jobid(k) for k in jobs_for(G0)}]
        if new:stage(root,'refinement','A',new)
        g1=graph(root,nodes,'refined');stage(root,'fine','A',jobs_for(FINE));fine=graph(root,FINE,'fine')
        policies={'P_COARSE':g0['top1']['policy'],'P_ADAPTIVE':g1['top1']['policy'],'P_FINE':fine['top1']['policy'],'P_MANUAL':records()['MANUAL8'],'TRANSFER4_TO_8':records()['TRANSFER4_TO_8']}
        assert all(len(p['resampling_steps'])==3 for p in policies.values());dump(root/'frozen_policies.json',policies);dump(root/'policy_freeze_receipt.json',{'sha256':sha(root/'frozen_policies.json'),'before_audit':True,'timestamp':time.time()})
        stage(root,'audit')
        from .adaptive_refinement_report import finalize
        finalize(root)
        assert read(root/'INTEGRITY.json')['passed'];verify(root)
        sources=root/'sources';sources.mkdir(exist_ok=True)
        for n in SOURCE_FILES:shutil.copy2(Path(__file__).parent/n,sources/n)
        dump(root/'status.json',{'stage':'DONE','all_stages_complete':True,'archive_pending':True,'end_unix':time.time()})
        archive=root.with_suffix('.tar.gz');subprocess.run(['tar','-czf',str(archive),'-C',str(root.parent),root.name],check=True)
        receipt={'path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size};dump(root.parent/(root.name+'.archive_receipt.json'),receipt)
        dump(root/'status.json',{'stage':'DONE','all_stages_complete':True,'archive':receipt,'end_unix':time.time()})
    except BaseException as exc:
        dump(root/'status.json',{'stage':'ABORTED','exception':repr(exc),'traceback':traceback.format_exc(),'preserve_data':True,'end_unix':time.time()})
        (root/'REPORT.md').write_text('# Adaptive graph refinement: INCOMPLETE\n\n'+repr(exc)+'\n\nNo scientific verdict. Inspect status.json and logs; do not change frozen protocol to resume.\n')
        raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();run(a.run_root.resolve())
