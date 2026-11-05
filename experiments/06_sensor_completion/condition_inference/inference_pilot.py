import time
from common import *
from condition_inference import Solver


def main():
    setup_torch()
    checks=[]
    for name in ['nine_angle','doubledelta','blendednet']:
        solver=Solver(name,0)
        r=records(name,'val')[0];f,p=field(name,r),prepared(name,r)
        solver.prepare(p,f.points)
        c=np.asarray(conditions(name)['cases'][r['id']])
        z=torch.as_tensor((c-solver.mean)/solver.std,dtype=torch.float32,device='cuda')[None]
        sample=input_sample(name,'conditioned',p,len(f.points),np.empty(0,np.int64),np.empty(0),solver.meta,c)
        batch=fb.to_device(collate([sample]),'cuda')
        queries=torch.as_tensor(f.points[:100],dtype=torch.float32,device='cuda')[None]
        with torch.no_grad():
            g=solver.model.encode(batch,batch['values'],batch['present'],batch['anchor_values'],batch['anchor_present'])
            direct=solver.model.predict(g,queries)*solver.meta['std']+solver.meta['mean']
            specialised=solver.output(z,f.points[:100])
        difference=float((direct-specialised).abs().max())
        assert difference<2e-5,(name,difference)
        check=dict(dataset=name,condition_interface_max_difference=difference)
        if name=='blendednet':
            obs,masks=fb.read_observations(ROOT/'data'/name,'val')
            e=next(e for e in obs['entries'] if e['car']==r['id'] and e['setting']=='sensors-20')
            index=masks[e['key']]
            lo,hi=solver.domain(True)
            synthetic=[]
            for fraction in [.35,.65]:
                condition=lo+(hi-lo)*fraction
                zz=torch.as_tensor((condition-solver.mean)/solver.std,dtype=torch.float32,device='cuda')[None]
                with torch.no_grad():
                    values=solver.output(zz,f.points[index])[0].cpu().numpy()
                started=time.perf_counter()
                output,inferred,diagnostic=solver.optimise(index,values,True)
                relative=diagnostic['final_sensor_mse']/solver.meta['std']**2
                assert relative<1e-5,(fraction,relative,diagnostic)
                synthetic.append(dict(fraction=fraction,normalised_sensor_mse=relative,
                    condition_error=(inferred-condition).tolist(),seconds=time.perf_counter()-started))
            check['synthetic_controls']=synthetic
            # Inference receives only geometry, observed indices and observed pressures.
            # Poison every case label after model construction and require identical results.
            class ForbiddenLabels(dict):
                def __getitem__(self,key):
                    raise AssertionError('Inference accessed the true case condition')
                get=__getitem__
            labels=conditions(name)['cases'];conditions(name)['cases']=ForbiddenLabels()
            again,_,_=solver.optimise(index,values,True)
            conditions(name)['cases']=labels
            baseline,_,_=solver.optimise(index,values,True)
            altered_difference=float(np.max(np.abs(output-again)))
            repeat_difference=float(np.max(np.abs(output-baseline)))
            assert max(altered_difference,repeat_difference)<1e-5
            assert altered_difference<=3*repeat_difference+1e-6
            check['label_poison_max_difference']=altered_difference
            check['unchanged_input_repeat_max_difference']=repeat_difference
        checks.append(check);print(json.dumps(check),flush=True)
        del solver;torch.cuda.empty_cache()
    # Two optimiser runs on unchanged inputs differ by about 2.6e-6 on the GPU, so the label-poison run is
    # compared against that repeat difference (with an absolute 1e-5 ceiling) rather than for exact equality.
    write(ROOT/'inference_pilot.json',dict(status='PASS',checks=checks))


if __name__=='__main__':
    main()
