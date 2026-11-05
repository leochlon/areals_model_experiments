import csv
import json
import sys

import numpy as np

from setup_confirmation import ROOT,write
sys.path.insert(0,'/workspace/runs/field_benchmark_4gpu_20261023')
import field_benchmark as fb


def independent_metrics(target,weight,prediction):
    y,w,p=map(lambda x:np.asarray(x,np.float64),(target,weight,prediction))
    assert y.shape==w.shape==p.shape and np.isfinite(p).all() and (w>0).all()
    total=w.sum();mean=np.sum(w*y)/total
    variance=np.sum(w*(y-mean)**2)/total
    square=np.sum(w*(p-y)**2)
    assert variance>0
    return dict(nrmse=float(np.sqrt(square/total/variance)),rel_l2=float(np.sqrt(square/np.sum(w*y*y))))


def main():
    fixture=independent_metrics([0.,2.],[1.,3.],[0.,0.])
    assert fixture==dict(nrmse=2.,rel_l2=1.)
    protocol=json.loads((ROOT/'confirmation_protocol.json').read_text())
    tasks=json.loads((ROOT/'tasks.json').read_text())
    ids=protocol['splits']['test'];assert len(ids)==40 and len(tasks)==18
    bench=ROOT/'bench';config=fb.bench_config(bench)
    metadata,observations=fb.read_observations(bench,'test')
    assert len(metadata['entries'])==40 and all(not len(v) for v in observations.values())
    fields={identity:fb.load_field(config['fields'],identity) for identity in ids}
    rows,checks,maximum=[],[],0.
    for task in tasks:
        done=json.loads((ROOT/'results/tasks'/f"{task['task']:02d}.json").read_text())
        assert done['passed'] and done['true_downstream_seed']==task['seed'] and done['train_cars']==380
        path=bench/'predictions'/task['name']/'test.npz'
        with np.load(path) as predictions:
            assert set(predictions.files)=={e['key'] for e in metadata['entries']}
            for entry in metadata['entries']:
                identity=entry['car'];field=fields[identity]
                p=predictions[entry['key']]
                scores=independent_metrics(field.values,field.weights,p)
                native=fb.problem_metrics(field,observations[entry['key']],p)
                difference=max(abs(scores[k]-native[k]) for k in scores)
                maximum=max(maximum,difference);assert difference<1e-10
                rows.append(dict(group=task['group'],seed=task['seed'],car=identity,**scores))
        log=(ROOT/'logs'/f"task_{task['task']:02d}.log").read_text()  # the check output saved by run_task.py
        passes=sum(line.startswith('PASS ') for line in log.splitlines())
        failures=sum(line.startswith('FAIL ') for line in log.splitlines())
        assert passes==4 and failures==0
        checks.append(dict(task=task['task'],name=task['name'],passed=passes,failed=failures))
    lookup={(r['group'],r['seed'],r['car']):r['nrmse'] for r in rows}
    matrices={group:np.asarray([[lookup[group,seed,car] for car in ids] for seed in range(3)])
              for group in sorted({r['group'] for r in rows})}
    rng=np.random.default_rng(20261024);draws=rng.integers(0,40,(10000,40))
    summaries={group:dict(mean=float(a.mean()),per_seed=a.mean(axis=1).tolist(),
         car_bootstrap95=np.quantile(a.mean(axis=0)[draws].mean(axis=1),[.025,.975]).tolist()) for group,a in matrices.items()}
    contrasts=[];alpha=.05/4
    for first,second in protocol['primary_contrasts']:
        a,b=matrices[first],matrices[second]
        difference=(a-b).mean(axis=0);samples=difference[draws].mean(axis=1)
        low,high=np.quantile(samples,[alpha/2,1-alpha/2])
        seed_differences=(a-b).mean(axis=1)
        contrasts.append(dict(first=first,second=second,mean_difference=float(difference.mean()),
            bonferroni_9875_interval=[float(low),float(high)],
            per_seed_differences=seed_differences.tolist(),
            first_lower_each_seed=bool((seed_differences<0).all()),
            interval_upper_below_zero=bool(high<0),
            relative_mean_error_reduction=float(1-a.mean()/b.mean())))
    out=ROOT/'results/confirmation';out.mkdir(exist_ok=True)
    with (out/'per_car.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(passed=True,models=len(tasks),test_cars=40,train_cars=380,validation_cars=16,
        summaries=summaries,primary_contrasts=contrasts,
        fixture=fixture,hidden_pressure_checks=sum(c['passed'] for c in checks),
        native_metric_max_disagreement=maximum,artifacts=checks)
    write(out/'summary.json',result)
    print('CONFIRMATION_SCORED',json.dumps({k:v for k,v in result.items() if k!='artifacts'}),flush=True)


if __name__=='__main__':main()
