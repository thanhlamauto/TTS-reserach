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
if __name__=='__main__':unittest.main()
