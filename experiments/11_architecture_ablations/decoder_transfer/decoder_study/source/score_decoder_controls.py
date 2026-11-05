import csv
import json

import numpy as np

from setup_confirmation import ROOT,write
from score_confirmation import independent_metrics,fb


def main():
    primary=json.loads((ROOT/'results/confirmation/summary.json').read_text())
    assert primary['passed'] and primary['models']==18
    p=json.loads((ROOT/'confirmation_protocol.json').read_text())
    extension=json.loads((ROOT/'decoder_controls_protocol.json').read_text())
    tasks=json.loads((ROOT/'decoder_controls_tasks.json').read_text())
    ids=p['splits']['test'];bench=ROOT/'bench';config=fb.bench_config(bench)
    metadata,observations=fb.read_observations(bench,'test')
    fields={c:fb.load_field(config['fields'],c) for c in ids}
    with (ROOT/'results/confirmation/per_car.csv').open() as stream:
        rows=[dict(group=r['group'],seed=int(r['seed']),car=r['car'],nrmse=float(r['nrmse']),rel_l2=float(r['rel_l2'])) for r in csv.DictReader(stream)]
    checks=[];maximum=primary['native_metric_max_disagreement']
    for task in tasks:
        done=json.loads((ROOT/'results/extra_tasks'/f"{task['task']:02d}.json").read_text())
        assert done['passed'] and done['true_downstream_seed']==task['seed']
        path=bench/'predictions'/task['name']/'test.npz'
        with np.load(path) as predictions:
            for entry in metadata['entries']:
                c=entry['car'];field=fields[c]
                prediction=predictions[entry['key']]
                scores=independent_metrics(field.values,field.weights,prediction)
                native=fb.problem_metrics(field,observations[entry['key']],prediction)
                discrepancy=max(abs(scores[k]-native[k]) for k in scores)
                maximum=max(maximum,discrepancy);assert discrepancy<1e-10
                rows.append(dict(group=task['group'],seed=task['seed'],car=c,**scores))
        text=(ROOT/'logs'/f"task_{task['task']+18:02d}.log").read_text()  # the check output saved by run_task.py
        passes=sum(line.startswith('PASS ') for line in text.splitlines())
        failures=sum(line.startswith('FAIL ') for line in text.splitlines())
        assert passes==4 and failures==0
        checks.append(dict(model=task['name'],passed=passes,failed=failures,encoder_gradient=done['encoder_gradient']))
    lookup={(r['group'],r['seed'],r['car']):r['nrmse'] for r in rows}
    matrices={g:np.array([[lookup[g,s,c] for c in ids] for s in range(3)]) for g in sorted({r['group'] for r in rows})}
    assert len(matrices)==8 and len(rows)==24*40
    draws=np.random.default_rng(20261024).integers(0,40,(10000,40));alpha=.05/6
    contrasts=[]
    for first,second in p['primary_contrasts']+extension['comparisons']:
        a,b=matrices[first],matrices[second];difference=(a-b).mean(0)
        low,high=np.quantile(difference[draws].mean(1),[alpha/2,1-alpha/2])
        contrasts.append(dict(first=first,second=second,means=[float(a.mean()),float(b.mean())],
            mean_difference=float(difference.mean()),bonferroni_all6_interval=[float(low),float(high)],
            per_seed_differences=(a-b).mean(1).tolist(),first_lower_each_seed=bool(((a-b).mean(1)<0).all()),
            interval_upper_below_zero=bool(high<0),relative_mean_error_reduction=float(1-a.mean()/b.mean())))
    out=ROOT/'results/complete';out.mkdir(exist_ok=True)
    with (out/'per_car.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(passed=True,models=24,test_cars=40,train_cars=380,validation_cars=16,
        means={g:dict(mean=float(a.mean()),per_seed=a.mean(1).tolist()) for g,a in matrices.items()},
        six_contrasts=contrasts,hidden_pressure_checks=primary['hidden_pressure_checks']+sum(c['passed'] for c in checks),
        native_metric_max_disagreement=maximum,extra_checks=checks)
    write(out/'summary.json',result)
    print('ALL_CONTROLS_SCORED',json.dumps({k:v for k,v in result.items() if k!='extra_checks'}),flush=True)


if __name__=='__main__':main()
