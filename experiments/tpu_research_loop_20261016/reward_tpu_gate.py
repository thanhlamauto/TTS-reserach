"""TPU-only FP32 ImageReward parity against saved CPU golden scores. Not yet run."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--golden-manifest',type=Path,required=True);p.add_argument('--dcs-root',type=Path,default=Path('/home/thanhlamtba31/static-bfs-dcs'));a=p.parse_args()
 from budget_guard import authorize,history,arm_child_deadline
 authorize(a.root)
 leases=history(a.root)
 active=[r for r in leases if r['event']=='RESERVE' and r['phase']=='SD15_smoke' and r['deadline_unix']>time.time() and not any(x['event']=='COMPLETE' and x['job_id']==r['job_id'] for x in leases)]
 if len(active)!=1:raise PermissionError('must execute under exactly one active budget_guard lease')
 arm_child_deadline(active[0])
 assert os.environ.get('PJRT_DEVICE')=='TPU' and os.environ.get('TPU_VISIBLE_CHIPS')=='0'
 import numpy as np,torch
 import torch_xla
 # FP32 parameters alone do not request full-precision TPU matrix products.
 # Binding verified in the installed2.4 release; do not relax the golden tolerance.
 torch_xla._XLAC._xla_set_use_full_mat_mul_precision(True)
 import torch_xla.core.xla_model as xm
 import torch_xla.runtime as xr
 import torch_xla.debug.metrics as met
 from PIL import Image
 assert xr.addressable_runtime_device_count()==1
 cache=Path('/dev/shm/static-bfs-research-loop-20261016/reward-gate-xla-full-fp32');cache.mkdir(parents=True,exist_ok=True);xr.initialize_cache(str(cache))
 attempt=a.root/'reward_gate_attempts'/active[0]['job_id'];attempt.mkdir(parents=True,exist_ok=False)
 device=xm.xla_device();manifest=json.loads(a.golden_manifest.read_text());rows=manifest['images'];assert len(rows)==8
 sys.path.insert(0,str(a.dcs_root/'official/text_to_image/fkd_diffusers'))
 from image_reward_utils import rm_load
 started=time.perf_counter();reward=rm_load('ImageReward-v1.0',device=device,download_root=str(a.dcs_root/'cache/ImageReward'));reward.eval();reward.requires_grad_(False)
 assert all(p.device.type=='xla' and p.dtype==torch.float32 for p in reward.parameters())
 images=[]
 for r in rows:
  image=Image.open(a.dcs_root/'experiments/early20_quick_compare_v1'/r['relative_image']).convert('RGB')
  assert hashlib.sha256(np.asarray(image,dtype=np.uint8).tobytes()).hexdigest()==r['selected_rgb_sha256']
  images.append(image)
 forwards={'visual':0,'text':0}
 def hook(name):
  def visit(m,args,kwargs):
   tensors=[]
   def collect(x):
    if isinstance(x,torch.Tensor):tensors.append(x)
    elif isinstance(x,(list,tuple)):
     for v in x:collect(v)
    elif isinstance(x,dict):
     for v in x.values():collect(v)
   collect(args);collect(kwargs);assert tensors and all(t.device.type=='xla' for t in tensors),name+' neural CPU input'
   forwards[name]+=1
  return visit
 reward.blip.visual_encoder.register_forward_pre_hook(hook('visual'),with_kwargs=True)
 reward.blip.text_encoder.register_forward_pre_hook(hook('text'),with_kwargs=True)
 batches=[]
 for phase in ['cold','warm1','warm2']:
  met.clear_all();xm.mark_step();xm.wait_device_ops();t=time.perf_counter()
  with torch.no_grad():scores=np.asarray(reward.score_batched([r['prompt'] for r in rows],images),dtype=np.float64)
  xm.mark_step();xm.wait_device_ops();wall=time.perf_counter()-t
  expected=np.asarray([r['saved_CPU_IR'] for r in rows]);tol=manifest['tolerance'];passed=bool(np.allclose(scores,expected,atol=tol['atol'],rtol=tol['rtol']))
  rank_checks=[]
  for prompt in sorted(set(r['prompt'] for r in rows)):
   ix=[i for i,r in enumerate(rows) if r['prompt']==prompt]
   if len(ix)<2:continue
   truth=expected[ix];got=scores[ix];gap=float(np.sort(truth)[-1]-np.sort(truth)[-2]);match=int(np.argmax(truth))==int(np.argmax(got))
   rank_checks.append({'prompt':prompt,'CPU_top_gap':gap,'argmax_match':match,'within_preregistered_tie_margin':gap<=.002})
   assert match or gap<=.002,'unexplained TPU reward rank flip outside preregistered tie margin'
  unsupported=[name for name in met.counter_names() if name.startswith('aten::') and name!='aten::_local_scalar_dense' and met.counter_value(name)]
  report={'phase':phase,'wall_seconds':wall,'scores':scores.tolist(),'max_abs_error':float(np.max(abs(scores-expected))),'numerical_pass':passed,'rank_checks':rank_checks,'unsupported_or_host_fallback':unsupported,'metrics_report':met.metrics_report()}
  batches.append(report);partial={'batches':batches,'full_mat_mul_precision':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(attempt/'partial.json').write_text(json.dumps(partial,indent=2));(a.root/'reward_gate_partial.json').write_text(json.dumps(partial,indent=2))
  assert passed,'TPU FP32 reward mismatch with saved CPU scores; stop without changing tolerance'
  assert not unsupported,'unexpected host fallback in neural verifier'
 assert np.array_equal(batches[1]['scores'],batches[2]['scores'])
 result={'passed':True,'elapsed_seconds_including_load':time.perf_counter()-started,'neural_forwards':forwards,'device':str(device),'precision':'float32','full_mat_mul_precision':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'CPU_reference':'saved historical scores only; no CPU neural forward','batches':batches}
 (attempt/'result.json').write_text(json.dumps(result,indent=2));(a.root/'reward_tpu_gate.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
