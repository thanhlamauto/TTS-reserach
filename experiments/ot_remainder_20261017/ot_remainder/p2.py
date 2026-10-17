"""Same KL OT, epsilon and1e-10 tolerance; accelerated reduced dual solve."""
import argparse,json,time,traceback,contextlib,sys
from pathlib import Path
import torch
import yaml
from otlab.io import atomic_json,load_tensor,file_sha,tensor_sha
from otlab.ot import masses,quadratic,epsilon_scale,solve as legacy_solve
from otlab.target import target_weights,cluster


@torch.no_grad()
def repaired_solve(cost,a=None,b=None,epsilon=.1,tol=1e-10,max_iter=10000):
    c=cost.detach().double();a=masses(a,len(c),c.device);b=masses(b,c.shape[1],c.device)
    ia,ib=a>0,b>0;aa,bb=a[ia],b[ib];cc=c[ia][:,ib]
    la,lb=aa.log(),bb.log();v=torch.zeros_like(bb);history=[]
    # Physical dual g=epsilon*v is preserved between scales. Warm starts do
    # not change the terminal regularizer or the target marginal tolerance.
    previous=epsilon*4
    for eps in [epsilon*4,epsilon*2,epsilon]:
        v=v*(previous/eps);previous=eps
        kernel=la[:,None]+lb[None,:]-cc/eps
        for it in range(1000):
            u=la-torch.logsumexp(kernel+v[None,:],1)
            v=lb-torch.logsumexp(kernel+u[:,None],0)
        v=v-v[-1]
        logits=lb[None,:]-cc/eps
        def eval_dual(value):
            z=logits+value[None,:]
            q=torch.softmax(z,1);p=aa[:,None]*q
            loss=(aa*torch.logsumexp(z,1)).sum()-(bb*value).sum()
            return loss,q,p
        for it in range(300):
            loss,q,p=eval_dual(v);col=p.sum(0);err=float((col-bb).abs().max())
            if it%10==0 or err<=tol:history.append(dict(epsilon=float(eps),newton_iteration=it,marginal_error=err))
            if err<=tol:break
            grad=(col-bb)[:-1]
            H=(torch.diag(col)-torch.einsum('i,ij,ik->jk',aa,q,q))[:-1,:-1]
            if not len(grad):break
            step=torch.linalg.solve(H+1e-15*torch.eye(len(H),dtype=torch.float64),-grad)
            step=step*min(1.,100./max(float(step.abs().max()),1e-30))
            slope=float(grad@step);rate=1.;accepted=False
            for ls in range(50):
                proposal=v.clone();proposal[:-1]+=rate*step
                newloss,_,newp=eval_dual(proposal)
                if float(newloss)<=float(loss)+1e-4*rate*slope+1e-15:
                    v=proposal;accepted=True;break
                rate*=.5
            if not accepted:raise RuntimeError('dual Newton line search failed')
        if eps==epsilon and err>tol:raise RuntimeError(f'exact-epsilon marginal gate failed:{err}')
    _,_,p=eval_dual(v)
    err=max(float((p.sum(1)-aa).abs().max()),float((p.sum(0)-bb).abs().max()))
    if err>tol:raise RuntimeError(f'final marginal gate failed:{err}')
    logp=p.clamp_min(1e-300).log()
    kl=(p*(logp-la[:,None]-lb[None,:])).sum()
    value=(p*cc).sum()+epsilon*kl
    full=torch.zeros_like(c);full[ia[:,None]&ib[None,:]]=p.flatten()
    return full,value,dict(marginal_error=err,epsilon=float(epsilon),tol=tol,
        method='log-Sinkhorn physical-dual epsilon continuation plus gauge-fixed Newton',history=history,
        source_positive_atoms=int(ia.sum()),target_positive_atoms=int(ib.sum()))


def envelope(x,y,a,b,epsilon,offset=None):
    c=quadratic(x,y)
    if offset is not None:c=c+offset
    p,value,_=repaired_solve(c,a,b,epsilon)
    correction=(p*c).sum()
    return value+correction-correction.detach()


def gradient_check(x,y,a,b,epsilon,term,base_cost=None):
    rng=torch.Generator().manual_seed(2026101708)
    dx=torch.randn(x.shape,generator=rng,dtype=torch.float64);dx/=dx.norm()
    dy=torch.randn(y.shape,generator=rng,dtype=torch.float64);dy/=dy.norm()
    nominal=quadratic(x,y) if term=='cross' else quadratic(x,x) if term=='source_self' else quadratic(y,y)
    offset=None if base_cost is None else base_cost.double()-nominal
    def objective(xx,yy):
        if term=='cross':return envelope(xx,yy,a,b,epsilon,offset)
        if term=='source_self':return envelope(xx,xx,a,a,epsilon,offset)
        return envelope(yy,yy,b,b,epsilon,offset)
    xx=x.detach().clone().requires_grad_();yy=y.detach().clone().requires_grad_()
    gx,gy=torch.autograd.grad(objective(xx,yy),(xx,yy),allow_unused=True)
    analytic=float((0 if gx is None else (gx*dx).sum())+(0 if gy is None else (gy*dy).sum()))
    differences=[]
    for h in [1e-3,3e-4,1e-4]:
        fd=float((objective(x+h*dx,y+h*dy)-objective(x-h*dx,y-h*dy))/(2*h))
        differences.append(dict(h=h,fd=fd,absolute_error=abs(fd-analytic),relative_error=abs(fd-analytic)/max(abs(analytic),1e-8)))
    passed=any(v['relative_error']<1e-3 or v['absolute_error']<1e-7 for v in differences)
    return dict(term=term,passed=passed,analytic=analytic,finite_differences=differences,
                self_both_arguments_differentiated=term!='cross',
                precision_note='float64 local polynomial extension anchored at exact original float32 cost; constant rounding offset')


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);args=p.parse_args()
    cfg=json.loads(Path(args.protocol).read_text());root=Path(cfg['root']);root.mkdir(parents=True,exist_ok=True)
    if (root/'COMPLETE.json').exists():return
    start=time.time();torch.set_num_threads(4)
    atomic_json(root/'preregistered.json',cfg)
    try:
        path=Path(cfg['original_request'])
        if file_sha(path)!=cfg['original_request_sha256']:raise RuntimeError('failed original request changed')
        cp=load_tensor(path)
        atomic_json(root/'status.json',dict(stage='encode_exact_old_checkpoint',started_unix=start))
        with contextlib.redirect_stdout(sys.stderr):
            from otlab.features import TensorFeatures
            encoder=TensorFeatures(yaml.safe_load((Path(args.protocol).parent.parents[1]/'ot_tts_lab_v2_reimpl/configs/worker_dinov2.yaml').read_text()))
        with torch.no_grad():
            sf=torch.cat([encoder.encode_clean(im[None]) for im in cp['source_clean']])
            tf=torch.cat([encoder.encode_clean(im[None]) for im in cp['bank_clean']])
        weights,good=target_weights(cp['bank_reward'],cluster(tf))
        eps=epsilon_scale(sf,tf,1.)
        source=torch.full((len(sf),),1/len(sf),dtype=torch.float64)
        x,y=sf.double(),tf.double()
        torch.save(dict(source_features=sf,target_features=tf,source_mass=source,target_mass=weights,
                        epsilon=eps,encoder_identity=encoder.identity),root/'exact_instance.pt')
        # Original objective constructs distances in float32, THEN casts in
        # solve. Preserve that order to reproduce its exact failing instance.
        terms=[('cross',quadratic(sf,tf).double(),source,weights),('source_self',quadratic(sf,sf).double(),source,source),
               ('target_self',quadratic(tf,tf).double(),weights,weights)]
        rows=[]
        for name,c,a,b in terms:
            atomic_json(root/'status.json',dict(stage='solve',term=name,epsilon=eps))
            torch.save(dict(cost=c,a=a,b=b,epsilon=eps,tolerance=1e-10),root/(name+'.pt'))
            try:
                plan,val,info=legacy_solve(c,a,b,eps)
                legacy=dict(passed=True,value=float(val),info=info)
            except RuntimeError as exc:legacy=dict(passed=False,error=str(exc))
            try:
                plan,val,info=repaired_solve(c,a,b,eps)
                torch.save(dict(plan=plan,value=val,info=info),root/(name+'_repaired.pt'))
                repaired=dict(passed=True,value=float(val),info=info)
            except RuntimeError as exc:repaired=dict(passed=False,error=str(exc))
            row=dict(term=name,cost_sha=tensor_sha(c),legacy=legacy,repaired=repaired)
            rows.append(row);atomic_json(root/'TERM_RESULTS.json',rows)
        all_pass=all(r['repaired']['passed'] for r in rows)
        gradient=[]
        if all_pass:
            for name,c,a,b in terms:gradient.append(gradient_check(x,y,source,weights,eps,name,c))
        result=dict(passed=all_pass and all(r['passed'] for r in gradient),terms=rows,gradient_checks=gradient,
            old_failure_reproduced=any(not r['legacy']['passed'] for r in rows),epsilon=eps,tolerance=1e-10,
            source_atoms=8,target_atoms=16,positive_target_atoms=int((weights>0).sum()),
            encoder_counters=dict(encoder.counters),encoder_identity=encoder.identity,
            original_request_sha256=file_sha(path),elapsed_seconds=time.time()-start,
            denoising_nfe=0,tpu_chip_hours=0,gpu_hours=0,terminal_quality_tested=False,
            next='CORRECTNESS_ONLY_NO_AUTOMATIC_DIFFUSION_RUN')
        atomic_json(root/'COMPLETE.json',result);print('COMPLETE',json.dumps(result),flush=True)
    except BaseException as exc:
        atomic_json(root/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),elapsed_seconds=time.time()-start));raise


if __name__=='__main__':main()
