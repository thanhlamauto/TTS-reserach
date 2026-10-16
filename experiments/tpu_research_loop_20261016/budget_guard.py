"""Fail-closed campaign accounting. Does not provision or stop any VM."""
from pathlib import Path
import argparse,fcntl,hashlib,json,math,os,signal,subprocess,time,uuid

def load(p):return json.loads(Path(p).read_text())
def history(root):
 p=root/'accelerator_cost_ledger.jsonl'
 return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
def used(rows):
 reserve={r['job_id']:r['reserved_chip_hours'] for r in rows if r['event']=='RESERVE'}
 done={r['job_id']:r['occupied_slice_chip_hours'] for r in rows if r['event']=='COMPLETE'}
 return sum(done.get(k,v) for k,v in reserve.items())
def authorize(root):
 p=root/'authorization.json'
 if not p.exists():raise PermissionError('BUDGET_UNVERIFIED: authorization.json missing; no accelerator initialization permitted')
 a=load(p)
 billing_basis=(a.get('billing_conditions_verified') is True or a.get('owner_attested_no_charge') is True)
 if not (a.get('human_authorized') is True and billing_basis and a.get('authorization_reference')):raise PermissionError('BUDGET_UNVERIFIED: human allocation and a verified or owner-attested billing basis required')
 if a.get('GPU_budget')!=0:raise PermissionError('GPU budget must remain0')
 amount=a['remaining_authorized_chip_hours']
 if isinstance(amount,bool) or not isinstance(amount,(int,float)) or not math.isfinite(amount):raise PermissionError('BUDGET_UNVERIFIED: allocation must be a finite number')
 cap=min(16.,float(amount))
 if cap<=0:raise PermissionError('BUDGET_EXHAUSTED')
 return a,cap

def append(root,row):
 with (root/'accelerator_cost_ledger.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def reserve(root,phase,seconds,slice_chips):
 if not math.isfinite(seconds) or seconds<=15:raise ValueError('finite deadline and cleanup headroom required')
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 with (root/'budget.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);a,cap=authorize(root);rows=history(root)
  now=time.time();clock_path=root/'campaign_clock.json'
  clock=load(clock_path) if clock_path.exists() else {'started_unix':now,'physical_slice_chips':4,'technical_cap_chip_hours':cap}
  if clock['physical_slice_chips']!=4 or clock['technical_cap_chip_hours']!=cap:raise PermissionError('frozen campaign clock/cap changed')
  elapsed_occupancy=max(0.,now-clock['started_unix'])*4/3600
  remaining=cap-max(used(rows),elapsed_occupancy)
  if (root/'KILL_SWITCH').exists():raise PermissionError('kill switch active')
  if slice_chips!=4:raise PermissionError('this campaign verified4 physical chips; new topology needs a new preflight')
  hours=seconds*slice_chips/3600
  if seconds<=15:raise ValueError('reserve cleanup headroom too')
  if hours>remaining+1e-12:raise PermissionError('BUDGET_EXHAUSTED: cumulative cap including unresolved reservations')
  exploration=sum(r['occupied_slice_chip_hours'] for r in rows if r['event']=='COMPLETE' and r['phase']!='confirmation')+sum(r['reserved_chip_hours'] for r in rows if r['event']=='RESERVE' and r['phase']!='confirmation' and not any(s['event']=='COMPLETE' and s['job_id']==r['job_id'] for s in rows))
  if phase!='confirmation' and exploration+hours>.6*cap+1e-12:raise PermissionError('confirmation reserve40percent would be consumed')
  deadline=clock['started_unix']+cap*3600/4
  phase_deadline=deadline if phase=='confirmation' else clock['started_unix']+.6*cap*3600/4
  if now+seconds>phase_deadline+1e-6:raise PermissionError('campaign wall deadline or confirmation wall reserve would be consumed; clock never resets')
  if phase=='pointmaze_port':
   port=sum(r.get('occupied_slice_chip_hours',0) for r in rows if r['event']=='COMPLETE' and r['phase']==phase)
   port+=sum(r['reserved_chip_hours'] for r in rows if r['event']=='RESERVE' and r['phase']==phase and not any(s['event']=='COMPLETE' and s['job_id']==r['job_id'] for s in rows))
   if port+hours>min(1.,.1*cap)+1e-12:raise PermissionError('PointMaze port subcap would be exceeded')
  if not clock_path.exists():clock_path.write_text(json.dumps(clock,sort_keys=True)+'\n')
  job=uuid.uuid4().hex;row={'event':'RESERVE','job_id':job,'phase':phase,'reserved_chip_hours':hours,'physical_slice_chips':slice_chips,'deadline_unix':now+seconds,'campaign_deadline_unix':deadline,'recorded_unix':now,'authorization_sha256':hashlib.sha256((root/'authorization.json').read_bytes()).hexdigest(),'actual_billing_USD':'UNKNOWN','occupancy_is_not_invoice':True}
  append(root,row);return row

def arm_child_deadline(lease):
 """Independent OS timer survives supervisor death; default SIGALRM needs no Python GIL."""
 remaining=float(lease['deadline_unix'])-time.time()-2
 if not math.isfinite(remaining) or remaining<=0:raise PermissionError('lease expired')
 signal.signal(signal.SIGALRM,signal.SIG_DFL)
 signal.setitimer(signal.ITIMER_REAL,remaining)
 return remaining

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--check',action='store_true');p.add_argument('--phase',choices=['SD15_smoke','pointmaze_port','exploration','confirmation']);p.add_argument('--max-seconds',type=float);p.add_argument('--entrypoint',type=Path);p.add_argument('child_args',nargs=argparse.REMAINDER);a=p.parse_args()
 if a.check:
  auth,cap=authorize(a.root);print(json.dumps({'budget_status':'VERIFIED','effective_chip_hours':cap,'used_or_reserved':used(history(a.root)),'confirmation_reserve':.4*cap}));return
 if not a.phase or not a.max_seconds or not a.entrypoint:raise ValueError('phase,deadline,and audited entrypoint required')
 if a.entrypoint.resolve()!=Path(__file__).with_name('reward_tpu_gate.py').resolve():raise PermissionError('only audited reward smoke entrypoint currently enabled; scientific pilot is not ready')
 record=reserve(a.root,a.phase,a.max_seconds,4);started=time.monotonic();child=None;reason='completed';returncode=None
 previous_sigterm=signal.getsignal(signal.SIGTERM)
 def interrupted(sig,frame):
  nonlocal reason
  reason='interrupted_by_signal';raise SystemExit(128+sig)
 signal.signal(signal.SIGTERM,interrupted)
 try:
  env={**os.environ,'PJRT_DEVICE':'TPU','TPU_VISIBLE_CHIPS':'0','CUDA_VISIBLE_DEVICES':'','PYTHONUNBUFFERED':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
  child_args=a.child_args[1:] if a.child_args[:1]==['--'] else a.child_args
  argv=[os.sys.executable,str(a.entrypoint.resolve()),*child_args];child=subprocess.Popen(argv,env=env,start_new_session=True)
  while True:
   returncode=child.poll()
   if returncode is not None:break
   if time.monotonic()-started>a.max_seconds-15 or (a.root/'KILL_SWITCH').exists():reason='deadline_or_kill_switch';break
   time.sleep(1)
 finally:
  if child is not None and child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  signal.signal(signal.SIGTERM,previous_sigterm)
  elapsed=time.monotonic()-started
  append(a.root,{'event':'COMPLETE','job_id':record['job_id'],'phase':a.phase,'elapsed_wall_seconds':elapsed,'occupied_slice_chip_hours':4*elapsed/3600,'active_device_hours':'UNKNOWN unless child instruments execution','returncode':None if child is None else child.returncode,'reason':reason,'actual_billing_USD':'UNKNOWN'})
 if reason!='completed' or returncode!=0:raise SystemExit(1)
if __name__=='__main__':main()
