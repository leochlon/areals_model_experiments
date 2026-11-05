import argparse
import csv
from common import *


def distribution(matrices,draws=5000,seed=1029):
    names=list(matrices)
    shapes={m.shape for m in matrices.values()}
    assert len(shapes)==1
    seeds,groups=next(iter(shapes))
    rng=np.random.default_rng(seed)
    seed_draw=rng.integers(0,seeds,size=(draws,seeds))
    group_draw=rng.integers(0,groups,size=(draws,groups))
    return {name:matrix[seed_draw[:,:,None],group_draw[:,None,:]].mean((1,2)) for name,matrix in matrices.items()}


def matrix(rows,setting,metric):
    selected=[r for r in rows if r['setting']==setting]
    seeds=sorted({r['seed'] for r in selected});groups=sorted({r['family'] for r in selected})
    values=np.empty((len(seeds),len(groups)))
    for i,s in enumerate(seeds):
        for j,g in enumerate(groups):
            cases=[r[metric] for r in selected if r['seed']==s and r['family']==g]
            assert cases
            values[i,j]=np.mean(cases)
    return values,groups,seeds


def summary(rows,setting):
    result=dict(setting=setting)
    for metric in ['nrmse','nmse','force_rel']:
        if any(r[metric] is None for r in rows if r['setting']==setting):
            result[metric]=None
            continue
        a,groups,seeds=matrix(rows,setting,metric)
        boot=distribution({metric:a})[metric]
        means={str(seed):float(a[i].mean()) for i,seed in enumerate(seeds)}
        result[metric]=dict(mean=float(a.mean()),sample_sd=float(np.std(list(means.values()),ddof=1)) if len(seeds)>1 else None,
            per_seed=means,bootstrap95=np.quantile(boot,[.025,.975]).tolist(),geometries=len(groups),seeds=len(seeds))
    return result


def inference_results(partial=False):
    summaries=[];all_rows=[];complete=[]
    for name in ['nine_angle','doubledelta','blendednet']:
        method_groups={}
        for seed in [0,1,2]:
            folder=ROOT/'condition_inference'/name/f'seed{seed}'/'test'
            if not (folder/'complete.json').exists():
                if partial:
                    continue
                raise RuntimeError(f'Incomplete inference result: {folder}')
            done=read(folder/'complete.json');assert done['status']=='PASS'
            assert read(folder.parent/'val/complete.json')['checkpoint']==done['checkpoint']
            rows=read(folder/'rows.json');selection=read(ROOT/'inference_selection'/name/f'seed{seed}.json')
            for row in rows:
                row=dict(row,dataset=name,seed=seed)
                all_rows.append(row)
                if row['domain']=='true_condition_fp32':
                    method_groups.setdefault(('true_condition_fp32','reference'),[]).append(row)
                elif 'sigma' not in row:
                    method_groups.setdefault((row['domain'],'plugin'),[]).append(row)
                else:
                    if row['sigma']==0:
                        method_groups.setdefault((row['domain'],'plugin'),[]).append(row)
                    if row['sigma']==selection['sigma_by_domain_and_setting'][row['domain']][row['setting']]:
                        method_groups.setdefault((row['domain'],'mixture'),[]).append(row)
            complete.append(dict(dataset=name,seed=seed,cases=done['cases']))
        for (domain,method),rows in method_groups.items():
            for setting in sorted({r['setting'] for r in rows}):
                relevant=[r for r in rows if r['setting']==setting]
                reference=conditions(name)['cases']
                estimates=np.asarray([r.get('posterior_mean_condition',r['estimate']) if method=='mixture' else r['estimate'] for r in relevant])
                truth=np.asarray([reference[r['case']] for r in relevant])
                error=np.abs(estimates-truth)
                details=dict(condition_components=conditions(name)['names'],
                             condition_mae=error.mean(0).tolist(),
                             condition_mae_in_training_sd=(error/conditions(name)['std']).mean(0).tolist())
                if name!='blendednet':
                    details['angle_within_half_degree']=float(np.mean(error[:,0]<=.5))
                    details['exact_angle_accuracy']=float(np.mean(error[:,0]<1e-6))
                if any('diagnostic' in r for r in relevant):
                    details['boundary_fraction']=float(np.mean([r['diagnostic']['boundary'] for r in relevant]))
                summaries.append(dict(dataset=name,domain=domain,method=method,**summary(rows,setting),**details))
    write(ROOT/'results/condition_inference_comparison.json',dict(status='PARTIAL' if len(complete)<9 else 'COMPLETE',
           completed_runs=complete,summaries=summaries,precision='FP32'))
    write(ROOT/'results/condition_inference_rows.json',all_rows)
    return dict(completed_runs=len(complete),predictions=len(all_rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial-inference',action='store_true')
    print(json.dumps(inference_results(p.parse_args().partial_inference)))
