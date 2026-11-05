import time
from common import *
from condition_inference import Solver,posterior


def sensor_forward(model,p,f,index,meta):
    sample=input_sample('', 'measurements',p,len(f.points),index,f.values[index].copy(),meta)
    batch=fb.to_device(collate([sample]),'cuda')
    queries=torch.as_tensor(f.points,dtype=torch.float32,device='cuda')[None]
    with torch.no_grad():
        packet=model.encode(batch,batch['values'],batch['present'],batch['anchor_values'],batch['anchor_present'])
        prediction=model.predict(packet,queries)[0].cpu().numpy()*meta['std']+meta['mean']
    return prediction


def main():
    setup_torch();rows=[]
    for name in ['nine_angle','doubledelta','blendednet']:
        solver=Solver(name,0)
        model,_,state=load_checkpoint(legacy_checkpoint(name,'measurements',0),name,'measurements')
        observations,masks=fb.read_observations(ROOT/'data'/name,'val')
        lookup={(e['car'],e['setting']):e for e in observations['entries']}
        used=set();chosen=[]
        for r in records(name,'val'):
            if r['family'] not in used:
                chosen.append(r);used.add(r['family'])
            if len(chosen)==4:
                break
        for r in chosen:
            f,p=field(name,r),prepared(name,r)
            index=masks[lookup[(r['id'],'sensors-20')]['key']]
            for repeat in range(4):
                torch.cuda.synchronize();start=time.perf_counter()
                sensor_forward(model,p,f,index,state['benchmark'])
                torch.cuda.synchronize();sensor_seconds=time.perf_counter()-start
                torch.cuda.synchronize();start=time.perf_counter()
                solver.prepare(p,f.points)
                if name=='blendednet':
                    solver.optimise(index,f.values[index].copy(),True)
                else:
                    _,candidates=solver.grid(name!='nine_angle')
                    posterior(candidates,index,f.values[index].copy(),0.)
                torch.cuda.synchronize();inference_seconds=time.perf_counter()-start
                if repeat:
                    rows.append(dict(dataset=name,case=r['id'],repeat=repeat,
                         sensor_seconds=sensor_seconds,inference_seconds=inference_seconds))
        del solver,model;torch.cuda.empty_cache()
    summary=[]
    for name in ['nine_angle','doubledelta','blendednet']:
        subset=[r for r in rows if r['dataset']==name]
        sensor=float(np.median([r['sensor_seconds'] for r in subset]));infer=float(np.median([r['inference_seconds'] for r in subset]))
        summary.append(dict(dataset=name,median_sensor_seconds=sensor,median_inference_seconds=infer,
                            median_ratio=infer/sensor,cases=4,repeats_per_case=3))
    # One case, 20 taps, full-field prediction; prepared geometry is cached for both methods, checkpoint loading
    # and preprocessing are excluded, and the first repeat of each case is a warm-up.
    write(ROOT/'results/latency.json',dict(status='PASS',gpu=torch.cuda.get_device_name(),precision='FP32',
       rows=rows,summaries=summary))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
