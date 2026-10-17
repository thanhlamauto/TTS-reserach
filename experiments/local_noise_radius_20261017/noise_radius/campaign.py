"""Bounded effect-size calibration, then at most one fresh-anchor screen."""
import argparse,fcntl,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from otlab.io import atomic_json,file_sha
from noise_avoid.campaign import worker as verified_worker,read_rows,export_blind
from noise_avoid.model import calibrate,evaluate


def effective(cfg,phase):
    result=dict(cfg)
    if phase=='screen':
        chosen=json.loads((Path(cfg['root'])/'FROZEN_RADIUS.json').read_text())['selected_radius']
        if chosen is None:raise RuntimeError('no eligible radius, screen prohibited')
        result['radius']=chosen
        units=[u for u in cfg['units'] if u['split']=='screen']
        result['units']=[dict(u,prefix_seed=cfg['fresh_screen_seed']) for u in units]
    else:
        mapping={'r050':.05,'r100':.1}
        if phase not in mapping or mapping[phase] not in cfg['radii']:raise ValueError('unregistered radius')
        result['radius']=mapping[phase]
        units=[u for u in cfg['units'] if u['split']=='calibration']
        result['units']=[dict(u,split=phase,prefix_seed=cfg['development_seed']) for u in units]
    return result


def headroom(root,cfg):
    rows=[];rule=cfg['radius_selection']
    for phase,radius in [('r050',.05),('r100',.1)]:
        units=read_rows(root,phase)
        delta=np.concatenate([np.asarray(u['rewards'])[19:]-u['rewards'][18] for u in units])
        state=np.concatenate([np.asarray(u['state45_rms_from_tape_noop'])[19:] for u in units])
        harm=float(np.mean(delta < -cfg['delta_reward']));useful=float(np.mean(delta>cfg['delta_reward']))
        eligible=bool(rule['harm_rate_min']<=harm<=rule['harm_rate_max'] and useful>=rule['useful_rate_min'] and state.max()<=rule['max_state45_rms'])
        rows.append(dict(phase=phase,radius=radius,harm_rate=harm,useful_rate=useful,max_state45_rms=float(state.max()),eligible=eligible,
            actual_mixed_deltas=delta.tolist(),rule='actual development effects only; no fitted model, filter win or screen outcome used'))
    eligible=[r for r in rows if r['eligible']]
    choice=min(eligible,key=lambda r:r['radius']) if eligible else None
    return dict(rows=rows,selected_radius=None if choice is None else choice['radius'],selected_phase=None if choice is None else choice['phase'],
                radius_choices_complete=2,further_radii_allowed=False,heldout_screen_used=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--worker',type=int);p.add_argument('--phase');a=p.parse_args()
    source=Path(a.protocol).resolve().parent;repo=source.parents[1];cfg=json.loads(Path(a.protocol).read_text());root=Path(cfg['root'])
    parent_source=repo/'experiments/local_noise_avoidance_20261017'
    for rel,sha in cfg['parent_source_hashes'].items():
        if file_sha(repo/rel)!=sha:raise RuntimeError('parent implementation changed')
    if a.worker is not None:return verified_worker(effective(cfg,a.phase),a.worker,a.phase,parent_source)
    root.mkdir(parents=True,exist_ok=True);lock=(root/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'COMPLETE.json').exists():return
    if list(root.glob('r*/chip*')) or (root/'screen').exists():raise RuntimeError('partial prior work exists; classify before repair/resume')
    owners=subprocess.run(['fuser','/dev/vfio/0','/dev/vfio/1','/dev/vfio/2','/dev/vfio/3'],capture_output=True,text=True)
    if owners.stdout.strip():raise RuntimeError('TPU already occupied')
    if shutil.disk_usage(root).free<1024**3:raise RuntimeError('less than1GiB free')
    atomic_json(root/'preregistered.json',cfg)
    atomic_json(root/'SOURCE_HASHES.json',{str(p.relative_to(repo)):file_sha(p) for p in source.rglob('*') if p.is_file()})
    processes=[];begin=time.time()
    def interrupted(sig,frame):raise TimeoutError('persistent wall limit/cancellation')
    signal.signal(signal.SIGTERM,interrupted)
    def launch(phase):
        nonlocal processes
        atomic_json(root/'status.json',dict(stage=phase,elapsed_seconds=time.time()-begin));processes=[]
        active=effective(cfg,phase);atomic_json(root/(phase+'_effective_config.json'),active)
        for rank in range(3):
            log=(root/f'{phase}_chip{rank}.log').open('w')
            child=subprocess.Popen([sys.executable,'-B','-u','-m','noise_radius.campaign','--protocol',a.protocol,'--worker',str(rank),'--phase',phase],env=dict(os.environ,TPU_VISIBLE_CHIPS=str(rank)),stdout=log,stderr=subprocess.STDOUT)
            processes.append((child,log))
        atomic_json(root/'worker_pids.json',[p.pid for p,l in processes])
        exits=[p.wait() for p,l in processes]
        for p,l in processes:l.close()
        if any(exits):raise RuntimeError('worker failed; no automatic science retry')
    try:
        # Both radii run to completion. Neither filter model is consulted for radius selection.
        launch('r050');launch('r100')
        choice=headroom(root,cfg);atomic_json(root/'FROZEN_RADIUS.json',choice);print('FROZEN_RADIUS',json.dumps(choice),flush=True)
        screened=choice['selected_radius'] is not None
        if screened:
            active=effective(cfg,'screen');frozen=calibrate(read_rows(root,choice['selected_phase']),active)
            atomic_json(root/'FROZEN_ERROR_ALLOWANCE.json',frozen);print('ERROR_ALLOWANCE_FROZEN',json.dumps(frozen),flush=True)
            launch('screen');result=evaluate(read_rows(root,'screen'),frozen,active);export_blind(root,active)
        else:
            result=dict(stage1_gate_passed=False,model_for_future_mechanism=None,decision='STOP_NO_LOCAL_HEADROOM_AT_REGISTERED_RADII',fresh_screen_run=False)
        atomic_json(root/'RESULTS.json',result)
        receipts=[json.loads(p.read_text()) for p in sorted(root.glob('*/chip*/COMPLETE.json'))]
        count=len(receipts);expected=9 if screened else 6
        if count!=expected:raise RuntimeError('phase rectangle incomplete')
        totals={key:sum(r['counters'].get(key,0) for r in receipts) for key in set().union(*(r['counters'] for r in receipts))}
        if totals['denoise_sample_steps']!=2020*count or totals['original_full_replay_sample_steps']!=400*count or totals['denoise_padding_sample_steps']!=60*count:raise RuntimeError('NFE allocation mismatch')
        complete=dict(passed=True,anchors=count,terminal_rows=27*count,sampling_nfe=2420*count,diagnostic_nfe=1660*count,
            replay_extra_nfe=760*count,padding_sample_steps=60*count,physical_sample_steps=2480*count,replay_cases=4*count,
            cfg_branch_sample_evaluations=4960*count,aux_unet_nfe=0,worker_receipts=receipts,counters=totals,
            elapsed_seconds=time.time()-begin,occupied_worker_chip_hours=sum(r['elapsed_seconds'] for r in receipts)/3600,
            radius=choice['selected_radius'],fresh_screen_run=screened,stage1_gate_passed=result['stage1_gate_passed'],decision=result['decision'],
            actual_billing_usd=None,owner_attested_no_charge=True,physical_visible_chips=[0,1,2],final_pool_used=False,
            cumulative_with_parent_sampling_nfe=14520+2420*count)
        atomic_json(root/'COMPLETE.json',complete);atomic_json(root/'status.json',dict(stage='COMPLETE',decision=result['decision']))
        print('COMPLETE',json.dumps(complete),flush=True)
    except BaseException as exc:
        atomic_json(root/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()));raise
    finally:
        for p,l in processes:
            if p.poll() is None:p.terminate()
            if not l.closed:l.close()

if __name__=='__main__':main()
