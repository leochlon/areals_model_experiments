import argparse
import csv
import json
import sys

import numpy as np

from registry import ROOT,write

sys.path.insert(0,'/workspace/runs/field_benchmark_4gpu_20261023')
import field_benchmark as fb


def interval(values):
    values=np.asarray(values,np.float64)
    assert values.ndim==1 and len(values)==48 and np.isfinite(values).all()
    draws=np.random.default_rng(20261024).integers(0,48,(4000,48))
    low,high=np.quantile(values[draws].mean(1),[.025,.975])
    return dict(mean=float(values.mean()),low=float(low),high=float(high),cars=48)


def contrast(a,b):
    assert a.shape==b.shape==(3,48)
    draws=np.random.default_rng(20261024).integers(0,48,(4000,48))
    relative=1-a.mean(0)[draws].mean(1)/b.mean(0)[draws].mean(1)
    return dict(first_minus_second=interval((a-b).mean(0)),relative_mean_reduction=float(1-a.mean()/b.mean()),
                relative_reduction_95=np.quantile(relative,[.025,.975]).tolist(),
                per_seed_means_first=a.mean(1).tolist(),per_seed_means_second=b.mean(1).tolist(),
                per_seed_differences=(a-b).mean(1).tolist())


def scores(model,field_cache):
    if model['family']=='frozen':
        path=ROOT/'predictions/frozen'/model['name']/'test.npz'
        info=json.loads(path.with_suffix('.json').read_text())
        assert info['passed'] and info['cases']==48
        rows=[];maximum=0.
        with np.load(path) as z:
            for row in info['rows']:
                identity=row['car'];p=z[identity+'_prediction'].astype(np.float64);y=z[identity+'_truth'].astype(np.float64)
                with np.load(ROOT/'frozen/data'/f'{identity}.npz') as target:
                    np.testing.assert_array_equal(y,target['fixed_target'].astype(np.float64).reshape(-1))
                score=float(np.sqrt(np.sum((p-y)**2)/np.sum(y**2)))
                maximum=max(maximum,abs(score-row['rel_l2']))
                rows.append(dict(car=identity,setting='fixed-2048-query',metric='rel_l2',value=score))
        assert maximum<1e-12
        return rows,dict(independent_maximum_difference=maximum,
                         hidden_value_checks=len(info['pressure_perturbation_checks']))
    bench=ROOT/'benches'/('completion' if model['family']=='replication' else 'geometry')
    config=fb.bench_config(bench);meta,observations=fb.read_observations(bench,'test')
    path=bench/'predictions'/model['name']/'test.npz'
    rows=[];maximum=0.
    with np.load(path) as z:
        assert set(z.files)=={e['key'] for e in meta['entries']}
        for entry in meta['entries']:
            identity=entry['car']
            if identity not in field_cache:field_cache[identity]=fb.load_field(config['fields'],identity)
            field=field_cache[identity]
            y,w,p=field.values.astype(np.float64),field.weights.astype(np.float64),z[entry['key']].astype(np.float64)
            assert p.shape==y.shape and np.isfinite(p).all()
            hidden=np.ones(len(y),bool);hidden[observations[entry['key']]]=False
            mean=np.sum(w*y)/w.sum();variance=np.sum(w*(y-mean)**2)/w.sum()
            square=np.sum(w[hidden]*(p[hidden]-y[hidden])**2)
            values=dict(nrmse=float(np.sqrt(square/w[hidden].sum()/variance)),
                        rel_l2=float(np.sqrt(square/np.sum(w[hidden]*y[hidden]**2))))
            native=fb.problem_metrics(field,observations[entry['key']],p)
            maximum=max(maximum,*[abs(v-native[k]) for k,v in values.items()])
            for metric,value in values.items():
                rows.append(dict(car=identity,setting=entry['setting'],metric=metric,value=value))
    assert maximum<1e-10
    log=(ROOT/'logs'/f'evaluate_{model["task"]:02d}.log').read_text()
    passes=sum(line.startswith('PASS ') for line in log.splitlines())
    assert passes>0 and not any(line.startswith('FAIL ') for line in log.splitlines())
    return rows,dict(independent_maximum_difference=maximum,hidden_value_checks=passes)


def main(primary):
    catalog=json.loads((ROOT/'models.json').read_text())
    models=[m for m in catalog if m['primary'] or not primary]
    assert len(models)==(12 if primary else 48)
    ids=json.loads((ROOT/'protocol.json').read_text())['test_ids']
    rows,artifacts,cache=[],[],{}
    for model in models:
        done=json.loads((ROOT/'results/tasks'/f'{model["task"]:02d}.json').read_text())
        assert done['passed']
        model_rows,artifact=scores(model,cache)
        group=(model['arm']+'_lr'+model['backbone_lr']) if model['family']=='online' else model['arm']
        for row in model_rows:rows.append(dict(family=model['family'],group=group,seed=model['seed'],name=model['name'],**row))
        artifacts.append(dict(model=model['name'],task=model['task'],**artifact))
    summaries,matrices,seeds={},{},[]
    for family in sorted({r['family'] for r in rows}):
        summaries[family]={}
        for group in sorted({r['group'] for r in rows if r['family']==family}):
            summaries[family][group]={}
            settings=sorted({r['setting'] for r in rows if r['family']==family and r['group']==group})
            if family=='replication':settings+=['all-25-original-settings']
            for setting in settings:
                summaries[family][group][setting]={}
                for metric in (['rel_l2'] if family=='frozen' else ['nrmse','rel_l2']):
                    a=np.asarray([[np.mean([r['value'] for r in rows if r['family']==family and r['group']==group
                             and r['seed']==seed and r['car']==identity and r['metric']==metric
                             and (r['setting']==setting if setting!='all-25-original-settings' else r['setting']!='unobserved')])
                             for identity in ids] for seed in range(3)])
                    assert a.shape==(3,48) and np.isfinite(a).all()
                    matrices[family,group,setting,metric]=a
                    summaries[family][group][setting][metric]=interval(a.mean(0))
                    seeds.append(dict(family=family,group=group,setting=setting,metric=metric,means=a.mean(1).tolist()))
    comparisons=[]
    pairs=[('replication','pretrained','scratch','all-25-original-settings','nrmse',True),
           ('online','areals_lr1e-4','pointmae_lr1e-3','unobserved','nrmse',True)]
    if not primary:
        pairs += [('online',f'areals_lr{lr}',f'pointmae_lr{lr}','unobserved','nrmse',False) for lr in ['1e-3','1e-4']]
        pairs += [('architecture','reference',other,'unobserved','nrmse',False) for other in ['ordinary_attention','unanchored_latents','oriented_normal_target','unoriented_vector_target']]
        pairs += [('architecture','anchored_no_rotary','unanchored_no_rotary','unobserved','nrmse',False)]
        pairs += [('frozen','areal',other,'fixed-2048-query','rel_l2',False) for other in ['native','native_normals']]
    for family,first,second,setting,metric,is_primary in pairs:
        comparisons.append(dict(family=family,first=first,second=second,setting=setting,metric=metric,primary=is_primary,
                           **contrast(matrices[family,first,setting,metric],matrices[family,second,setting,metric])))
    out=ROOT/'results'/('primary' if primary else 'complete');out.mkdir(exist_ok=True)
    with (out/'per_problem.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(passed=True,primary_only=primary,models=len(models),test_cars=48,summaries=summaries,
          comparisons=comparisons,seeds=seeds,artifacts=artifacts)
    write(out/'summary.json',result)
    print('scored',json.dumps(dict(primary_only=primary,models=len(models),comparisons=comparisons)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--primary',action='store_true');main(parser.parse_args().primary)
