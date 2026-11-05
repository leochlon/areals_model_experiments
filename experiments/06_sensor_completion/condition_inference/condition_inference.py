import argparse
from contextlib import nullcontext
import time
from common import *


def checkpoint(name,seed):
    if name=='nine_angle' and seed:
        return ROOT/'legacy_references/nine_angle/conditioned'/f'seed{seed}/field_best.pt'
    return legacy_checkpoint(name,'conditioned',seed)


def expand_geometry(geometry,count):
    values={}
    for part in dataclasses.fields(geometry):
        value=getattr(geometry,part.name)
        if isinstance(value,torch.Tensor):
            assert value.shape[0]==1
            value=value.expand((count,)+value.shape[1:])
        values[part.name]=value
    return dataclasses.replace(geometry,**values)


class Solver:
    def __init__(self,name,seed):
        self.name=name
        self.model,self.experiment,state=load_checkpoint(checkpoint(name,seed),name,'conditioned')
        self.meta=state['benchmark']
        for p in self.model.parameters():
            p.requires_grad_(False)
        stats=conditions(name)
        self.mean=np.asarray(stats['mean']);self.std=np.asarray(stats['std'])
        self.bounds=np.asarray(stats['training_bounds'])
        self.spec=read(ROOT/'inference_protocol.json')
        self.geometry=None

    @torch.no_grad()
    def prepare(self,p,points):
        self.p=p;self.points=points
        if self.name!='nine_angle':
            sample=fb.encoder_inputs(p,len(points),np.empty(0,np.int64),np.empty(0))
            batch=fb.to_device(collate([sample]),'cuda')
            self.geometry=FieldEncoder.encode(self.model,batch,batch['values'],batch['present'],
                                               batch['anchor_values'],batch['anchor_present'])

    def output(self,normalised_condition,points,amp=False):
        n=len(normalised_condition)
        queries=torch.as_tensor(points,dtype=torch.float32,device='cuda')[None].expand(n,-1,-1)
        context=torch.autocast('cuda',dtype=torch.bfloat16) if amp else nullcontext()
        with context:
            if self.name=='nine_angle':
                samples=[]
                for c in normalised_condition.detach().cpu().numpy():
                    samples.append(input_sample(self.name,'conditioned',self.p,len(self.points),
                                   np.empty(0,np.int64),np.empty(0),self.meta,c*self.std+self.mean))
                batch=fb.to_device(collate(samples),'cuda')
                g=self.model.encode(batch,batch['values'],batch['present'],batch['anchor_values'],batch['anchor_present'])
            else:
                g=expand_geometry(self.geometry,n)
                delta=self.model.condition_projection(normalised_condition)[:,None]
                g=dataclasses.replace(g,tokens=g.tokens+delta)
            out=self.model.predict(g,queries)
        return out*self.meta['std']+self.meta['mean']

    def domain(self,extended):
        lo,hi=self.bounds.copy()
        if extended:
            span=hi-lo;lo-=span;hi+=span
            if self.name=='blendednet':
                lo[0]=max(0.,lo[0]);hi[0]=min(.99,hi[0]);lo[2]=max(0.,lo[2])
        return np.stack([lo,hi])

    @torch.no_grad()
    def grid(self,extended):
        if self.name=='nine_angle':
            c=np.arange(self.bounds[0,0],self.bounds[1,0]+.5,1.)[:,None]
        else:
            lo,hi=self.domain(extended)[:,0]
            c=np.arange(lo,hi+self.spec['angle_grid_step']/2,self.spec['angle_grid_step'])[:,None]
        outputs=[]
        for start in range(0,len(c),8):
            z=torch.as_tensor((c[start:start+8]-self.mean)/self.std,dtype=torch.float32,device='cuda')
            outputs.extend(self.output(z,self.points).cpu().numpy())
        return c,np.asarray(outputs,np.float64)

    def optimise(self,index,values,extended):
        lo,hi=self.domain(extended)
        lo=torch.as_tensor((lo-self.mean)/self.std,dtype=torch.float32,device='cuda')
        hi=torch.as_tensor((hi-self.mean)/self.std,dtype=torch.float32,device='cuda')
        design=torch.quasirandom.SobolEngine(len(lo),scramble=True,seed=self.spec['solver_seed']).draw(
            self.spec['initial_candidates']).cuda()
        candidates=lo+(hi-lo)*design
        candidates=torch.cat([candidates,((lo+hi)/2)[None]],dim=0)
        target=torch.as_tensor(values,dtype=torch.float32,device='cuda')
        query=self.points[index]
        with torch.no_grad():
            scores=[]
            for chunk in candidates.split(32):
                scores.extend(((self.output(chunk,query)-target[None])**2).mean(1).cpu().tolist())
        best=np.argsort(scores)[:self.spec['starts']]
        z=candidates[best].detach().clone().requires_grad_()
        best_z=z.detach().clone()
        best_loss=torch.as_tensor(np.asarray(scores)[best],dtype=torch.float32,device='cuda')
        opt=torch.optim.Adam([z],lr=self.spec['adam_lr'])
        trace=[]
        for step in range(self.spec['adam_steps']):
            opt.zero_grad(set_to_none=True)
            losses=((self.output(z,query)-target[None])**2).mean(1)
            assert torch.isfinite(losses).all()
            with torch.no_grad():
                improved=losses<best_loss;best_z[improved]=z[improved];best_loss=torch.minimum(losses,best_loss)
            (losses.sum()/self.meta['std']**2).backward();opt.step()
            with torch.no_grad():
                z.copy_(torch.maximum(lo,torch.minimum(hi,z)))
            if step%25==0:
                trace.append(dict(step=step,sensor_mse=float(best_loss.min())))
        # Optimise all starts independently; their losses have no coupling across starts.
        z=best_z.detach().clone().requires_grad_()
        polish=torch.optim.LBFGS([z],lr=.5,max_iter=self.spec['lbfgs_steps'],line_search_fn='strong_wolfe',
                                 tolerance_grad=1e-8,tolerance_change=1e-10)
        def closure():
            polish.zero_grad(set_to_none=True)
            bounded=torch.maximum(lo,torch.minimum(hi,z))
            loss=((self.output(bounded,query)-target[None])**2).mean(1).sum()/self.meta['std']**2
            loss.backward();return loss
        polish.step(closure)
        with torch.no_grad():
            z=torch.maximum(lo,torch.minimum(hi,z))
            losses=((self.output(z,query)-target[None])**2).mean(1)
            improved=losses<best_loss;best_z[improved]=z[improved];best_loss=torch.minimum(losses,best_loss)
            choice=int(best_loss.argmin());chosen=best_z[choice:choice+1]
            prediction=self.output(chosen,self.points)[0].cpu().numpy().astype(np.float64)
            raw=chosen.cpu().numpy()[0]*self.std+self.mean
        return prediction,raw,dict(initial_sensor_mse=float(min(scores)),final_sensor_mse=float(best_loss.min()),
              start_residuals=best_loss.cpu().tolist(),trace=trace,
              boundary=bool(np.any(np.minimum(raw-self.domain(extended)[0],self.domain(extended)[1]-raw)<1e-4)))


def posterior(candidates,index,values,sigma):
    ss=np.sum((candidates[:,index]-values[None])**2,axis=1)
    best=int(ss.argmin())
    if sigma==0:
        weights=np.eye(len(candidates))[best]
    else:
        weights=np.exp(-(ss-ss.min())/(2*sigma*sigma));weights/=weights.sum()
    return weights@candidates,best,weights


def run(name,seed,split):
    if split=='test':
        selection=read(ROOT/'inference_selection'/name/f'seed{seed}.json')
        assert selection['status']=='PASS'
    solver=Solver(name,seed)
    model=str(checkpoint(name,seed))
    obs,masks=fb.read_observations(ROOT/'data'/name,split)
    groups=fb.by_car(obs['entries'])
    selected_rows=records(name,split)
    if split=='val' and name=='blendednet':
        ids=set(read(ROOT/'data'/name/'selection.json')['cases'])
        selected_rows=[r for r in selected_rows if r['id'] in ids]
    result=ROOT/'condition_inference'/name/f'seed{seed}'/split
    result.mkdir(parents=True,exist_ok=True)
    rows=[]
    for number,r in enumerate(selected_rows,1):
        f,p=field(name,r),prepared(name,r)
        solver.prepare(p,f.points)
        outputs={}
        settings=solver.spec['sensor_counts']
        for extended in ([False] if name=='nine_angle' else [False,True]):
            domain='extended' if extended else 'training_support'
            grid=solver.grid(extended) if name!='blendednet' else None
            for e in groups[r['id']]:
                if e['setting'] not in {f'sensors-{k}' for k in settings}:
                    continue
                ix=masks[e['key']];values=f.values[ix].copy();started=time.perf_counter()
                if grid is not None:
                    c,candidates=grid
                    sigmas=[float(x*solver.meta['std']) for x in solver.spec['sigma_factors']] if split=='val' else [
                        0.,selection['sigma_by_domain_and_setting'][domain][e['setting']]]
                    for sigma in sorted(set(sigmas)):
                        prediction,choice,weights=posterior(candidates,ix,values,sigma)
                        row=dict(case=r['id'],family=r['family'],setting=e['setting'],domain=domain,sigma=sigma,
                            estimate=c[choice].tolist(),posterior_mean_condition=(weights@c).tolist(),**metrics(f,ix,prediction))
                        rows.append(row)
                        if split=='test':
                            outputs[f'{domain}_{e["setting"]}_sigma{sigma:g}']=prediction.astype(np.float32)
                else:
                    prediction,c,diagnostic=solver.optimise(ix,values,extended)
                    rows.append(dict(case=r['id'],family=r['family'],setting=e['setting'],domain=domain,
                        estimate=c.tolist(),diagnostic=diagnostic,**metrics(f,ix,prediction)))
                    if split=='test':
                        outputs[f'{domain}_{e["setting"]}']=prediction.astype(np.float32)
                rows[-1]['seconds']=time.perf_counter()-started
        # True labels are accessed only after sensor-only inference has finished for this case.
        truth=conditions(name)['cases'][r['id']]
        with torch.no_grad():
            z=torch.as_tensor((np.asarray(truth)-solver.mean)/solver.std,dtype=torch.float32,device='cuda')[None]
            oracle=solver.output(z,f.points)[0].cpu().numpy().astype(np.float64)
        for e in groups[r['id']]:
            if e['setting'] in {f'sensors-{k}' for k in settings}:
                rows.append(dict(case=r['id'],family=r['family'],setting=e['setting'],domain='true_condition_fp32',
                    estimate=truth,**metrics(f,masks[e['key']],oracle)))
        if split=='test':
            outputs['true_condition_fp32']=oracle.astype(np.float32)
            np.savez_compressed(result/(r['id']+'.npz'),**outputs)
        write(result/'progress.json',dict(cases=number,total=len(selected_rows)))
        if number%8==0:
            write(result/'partial_rows.json',rows)
            print(name,seed,split,number,len(selected_rows),flush=True)
    write(result/'rows.json',rows)
    write(result/'complete.json',dict(status='PASS',cases=len(selected_rows),checkpoint=model))
    if split=='val':
        choices={}
        if name!='blendednet':
            for domain in sorted({r['domain'] for r in rows if 'sigma' in r}):
                choices[domain]={}
                for setting in sorted({r['setting'] for r in rows}):
                    temperatures=sorted({r['sigma'] for r in rows if r['domain']==domain and r['setting']==setting})
                    scores={sigma:np.mean([r['nmse'] for r in rows if r['domain']==domain and r['setting']==setting and r.get('sigma')==sigma]) for sigma in temperatures}
                    choices[domain][setting]=min(scores,key=scores.get)
        write(ROOT/'inference_selection'/name/f'seed{seed}.json',dict(status='PASS',sigma_by_domain_and_setting=choices,
              checkpoint=model,selection_split='val'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True)
    p.add_argument('--split',required=True,choices=['val','test']);p.add_argument('--index',type=int,default=0)
    args=p.parse_args();setup_torch();run(args.dataset,args.index,args.split)
