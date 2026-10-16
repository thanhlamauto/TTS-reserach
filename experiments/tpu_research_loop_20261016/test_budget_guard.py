import importlib.util,json,tempfile,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('guard',Path(__file__).with_name('budget_guard.py'));g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
class GuardTests(unittest.TestCase):
 def auth(self,r):
  (r/'authorization.json').write_text(json.dumps({'human_authorized':True,'billing_conditions_verified':True,'authorization_reference':'SYNTHETIC_UNIT_TEST_ONLY','GPU_budget':0,'remaining_authorized_chip_hours':8}))
 def test_missing_authorization(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(PermissionError):g.authorize(Path(d))
 def test_reserve_does_not_reset_on_restart(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);self.auth(r);g.reserve(r,'exploration',3600,4)
   self.assertEqual(g.used(g.history(r)),4)
   with self.assertRaises(PermissionError):g.reserve(r,'exploration',1000,4)
 def test_confirmation_reserve(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);self.auth(r)
   with self.assertRaises(PermissionError):g.reserve(r,'exploration',5000,4)
 def test_port_cap(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);self.auth(r)
   with self.assertRaises(PermissionError):g.reserve(r,'pointmaze_port',800,4)
 def test_kill_switch(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);self.auth(r);(r/'KILL_SWITCH').touch()
   with self.assertRaises(PermissionError):g.reserve(r,'exploration',30,4)
 def test_owner_attestation_is_distinct_from_invoice_verification(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);self.auth(r);a=json.loads((r/'authorization.json').read_text());a['billing_conditions_verified']=False;a['owner_attested_no_charge']=True
   (r/'authorization.json').write_text(json.dumps(a));self.assertEqual(g.authorize(r)[1],8)
 def test_campaign_clock_blocks_idle_reset(self):
  import time
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);self.auth(r)
   (r/'campaign_clock.json').write_text(json.dumps({'started_unix':time.time()-7201,'physical_slice_chips':4,'technical_cap_chip_hours':8}))
   with self.assertRaises(PermissionError):g.reserve(r,'confirmation',30,4)
 def test_exploration_cannot_spend_confirmation_wall_reserve(self):
  import time
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);self.auth(r)
   (r/'campaign_clock.json').write_text(json.dumps({'started_unix':time.time()-4310,'physical_slice_chips':4,'technical_cap_chip_hours':8}))
   with self.assertRaises(PermissionError):g.reserve(r,'exploration',30,4)

class InterruptionTests(unittest.TestCase):
 def test_invalid_numbers_are_denied(self):
  for value in [float('nan'),float('inf'),True]:
   with tempfile.TemporaryDirectory() as d:
    r=Path(d);r.joinpath('authorization.json').write_text(json.dumps({'human_authorized':True,'billing_conditions_verified':True,'authorization_reference':'CPU_TEST_ONLY','GPU_budget':0,'remaining_authorized_chip_hours':value}))
    with self.assertRaises(PermissionError):g.authorize(r)
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):g.reserve(Path(d),'SD15_smoke',float('nan'),4)
 def test_worker_own_deadline_needs_no_supervisor(self):
  import subprocess,sys,time,signal
  program="import time; from budget_guard import arm_child_deadline; arm_child_deadline({'deadline_unix':time.time()+2.3}); time.sleep(20)"
  started=time.monotonic();run=subprocess.run([sys.executable,'-c',program],cwd=Path(__file__).parent,timeout=3)
  self.assertEqual(run.returncode,-signal.SIGALRM);self.assertLess(time.monotonic()-started,3)
 def test_sigterm_reaps_worker_and_records_ledger(self):
  import subprocess,sys,time,signal,os,shutil
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);GuardTests().auth(r);guard=r/'budget_guard.py';shutil.copyfile(Path(__file__).with_name('budget_guard.py'),guard)
   worker=r/'reward_tpu_gate.py';worker.write_text("import os,time\nfrom pathlib import Path\nPath(__file__).with_name('child.pid').write_text(str(os.getpid()))\ntime.sleep(60)\n")
   parent=subprocess.Popen([sys.executable,str(guard),'--root',str(r),'--phase','SD15_smoke','--max-seconds','20','--entrypoint',str(worker)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   try:
    end=time.monotonic()+4
    while not (r/'child.pid').exists() and time.monotonic()<end:time.sleep(.02)
    self.assertTrue((r/'child.pid').exists());child=int((r/'child.pid').read_text());parent.send_signal(signal.SIGTERM);parent.communicate(timeout=4)
    self.assertNotEqual(parent.returncode,0)
    with self.assertRaises(ProcessLookupError):os.kill(child,0)
    rows=g.history(r);self.assertEqual(rows[-1]['event'],'COMPLETE');self.assertEqual(rows[-1]['reason'],'interrupted_by_signal');self.assertGreater(g.used(rows),0)
   finally:
    if parent.poll() is None:parent.kill();parent.communicate()

if __name__=='__main__':unittest.main()
