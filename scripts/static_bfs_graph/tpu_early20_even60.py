"""Frozen exploratory third arm: resample after completed even updates 2..20."""
from datetime import datetime,timezone
import json,os,statistics,time
from pathlib import Path
from .tpu_early20_train60 import CapturePipeline
from .tpu_reference_compare import checked_inputs,REPO,digest,read,write_once
from .policy import StaticBFSPolicy
from .adapter import event_seed

PROTOCOL='STATIC_BFS_EARLY20_EVEN60_TPU_V1'
SEED=42
STEPS=tuple(range(1,20,2))

def run(root,dcs,previous,chip):
    pool,cfg,baseline=checked_inputs()
    prompts=tuple(pool['train_indices'][chip::4])
    assert os.environ['PJRT_DEVICE']=='TPU' and os.environ['TPU_VISIBLE_CHIPS']==str(chip)
    old=previous/f'chip{chip}'
    prior=read(old/'config.json'); manifest=read(old/'manifest.json')
    assert prior['prompt_ids']==list(prompts) and prior['seed']==SEED
    protected=['scripts/static_bfs_graph/runner.py','scripts/static_bfs_graph/adapter.py','scripts/diffusion_classical_search_t2i/tpu_runner.py']
    assert all(digest(REPO/p)==manifest['source_sha256'][p] for p in protected)
    policy=StaticBFSPolicy(STEPS,tuple((i,8.) for i in STEPS),8)
    reference=StaticBFSPolicy((19,),((19,8.),),8)
    plan={'protocol':PROTOCOL,'method':'EVEN2_TO20','prompt_ids':list(prompts),'prompts':{str(p):pool['prompts'][str(p)] for p in prompts},'seed':SEED,'N':8,'sampler_steps':100,'eta':1.,'precision':'BF16','scoring':'Max','resampling':'SSP','tau':8.,'verifier':'ImageReward-v1.0','policy':policy.record(),'completed_update_numbers':list(range(2,21,2)),'logical_runs':len(prompts),'diffusion_NFE_per_run':800,'verifier_evaluations_per_run':88,'previous_run':str(previous),'previous_config_sha256':digest(old/'config.json'),'exploratory_TRAIN_only':True,'validation_or_test_used':False}
    write_once(root/'config.json',plan)
    write_once(root/'manifest.json',{'config_sha256':digest(root/'config.json'),'source_sha256':{p:digest(REPO/p) for p in protected+[str(Path(__file__).relative_to(REPO)),'scripts/static_bfs_graph/tpu_early20_train60.py']},'previous_manifest_sha256':digest(old/'manifest.json'),'model_revision':baseline['models']['sd15']['revision']})
    from .runner import Runtime,static_record
    started=time.time();write_once(root/'started.json',{'utc':datetime.fromtimestamp(started,timezone.utc).isoformat()})
    cfg={**cfg,'particles':8,'protocol_version':PROTOCOL}
    runtime=Runtime(root,dcs,SEED,chip,cfg,baseline,[{'prompt':pool['prompts'].get(str(i))} for i in range(100)])
    runtime.pipe=CapturePipeline(runtime.pipe)
    (root/'images').mkdir(exist_ok=True);rows=[]
    for prompt in prompts:
        row=runtime.run('early20_even',static_record(policy),prompt,reference)
        assert row['diffusion_NFE']==800 and row['verifier_calls']==88
        assert [e['sampling_index'] for e in row['resampling_events']]==list(STEPS)
        assert all(e['temperature']==8. and e['event_seed']==event_seed(SEED,prompt,e['sampling_index']) for e in row['resampling_events'])
        for name in ['ONLY20','EVERY1_TO20']:
            paired=read(old/'records'/f'p{prompt:03d}-{name}.json')
            assert paired['initial_noise_hash']==row['initial_generator_state_sha256']
        image=root/'images'/f'p{prompt:03d}-EVEN2_TO20.png'
        if runtime.pipe.images is None:raise RuntimeError('raw-only resume requires image recovery before completion')
        runtime.pipe.images[row['selected_particle_index']].save(image)
        compact={'prompt_id':prompt,'prompt':pool['prompts'][str(prompt)],'seed':SEED,'method':'EVEN2_TO20','ImageReward':row['selected_final_score'],'diffusion_NFE':row['diffusion_NFE'],'verifier_evaluations':row['verifier_calls'],'runtime_seconds':row['elapsed_seconds'],'event_update_numbers':[e['sampling_index']+1 for e in row['resampling_events']],'resampling_events':row['resampling_events'],'initial_noise_hash':row['initial_generator_state_sha256'],'selected_rgb_sha256':row['selected_image_rgb_sha256'],'final_particle_scores':row['final_particle_scores'],'image':str(image.relative_to(root)),'status':'success'}
        write_once(root/'records'/f'p{prompt:03d}-EVEN2_TO20.json',compact);rows.append(compact)
    (root/'raw_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result={'successful_runs':len(rows),'NFE':800*len(rows),'verifier_evaluations':88*len(rows),'mean_IR':statistics.mean(r['ImageReward'] for r in rows),'wall_seconds_including_load_and_compile':time.time()-started}
    write_once(root/'results.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--dcs-root',type=Path,required=True);p.add_argument('--previous-run',type=Path,required=True);p.add_argument('--chip',type=int,required=True)
    a=p.parse_args();run(a.run_root.resolve(),a.dcs_root.resolve(),a.previous_run.resolve(),a.chip)
