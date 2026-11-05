import argparse
import math
from common import *


def independent(f,index,prediction):
    y=f.values.astype(np.float64);w=f.weights.astype(np.float64)
    hidden=np.ones(len(y),bool);hidden[index]=False
    centre=np.average(y,weights=w)
    scale2=np.average((y-centre)**2,weights=w)
    hidden_error=np.average((prediction[hidden]-y[hidden])**2,weights=w[hidden])
    nmse=hidden_error/scale2
    result=dict(nmse=float(nmse),nrmse=math.sqrt(nmse))
    if f.normals is not None:
        completed=prediction.copy();completed[index]=y[index]
        truth=np.einsum('i,i,ij->j',w,y,f.normals)
        predicted=np.einsum('i,i,ij->j',w,completed,f.normals)
        result['force_rel']=float(np.linalg.norm(predicted-truth)/np.linalg.norm(truth))
    return result


def verify_inference(name):
    maximum=dict(nrmse=0.,nmse=0.,force_rel=0.)
    checked=0
    records_by_id={r['id']:r for r in records(name,'test')}
    obs,masks=fb.read_observations(ROOT/'data'/name,'test')
    entry={(e['car'],e['setting']):e for e in obs['entries']}
    for seed in [0,1,2]:
        folder=ROOT/'condition_inference'/name/f'seed{seed}'/'test'
        done=read(folder/'complete.json');assert done['status']=='PASS'
        val=read(folder.parent/'val/complete.json')
        assert done['checkpoint']==val['checkpoint']
        groups={}
        for row in read(folder/'rows.json'):
            groups.setdefault(row['case'],[]).append(row)
        assert set(groups)==set(records_by_id)
        for case,rows in groups.items():
            f=field(name,records_by_id[case])
            with np.load(folder/(case+'.npz')) as a:
                for row in rows:
                    domain,setting=row['domain'],row['setting']
                    key='true_condition_fp32' if domain=='true_condition_fp32' else f'{domain}_{setting}'
                    if 'sigma' in row:
                        key+=f'_sigma{row["sigma"]:g}'
                    pred=a[key].astype(np.float64)
                    ix=masks[entry[(case,setting)]['key']]
                    actual=independent(f,ix,pred)
                    for metric,value in actual.items():
                        maximum[metric]=max(maximum[metric],abs(value-row[metric]))
                    checked+=1
    assert maximum['nrmse']<1e-5 and maximum['nmse']<1e-5 and maximum['force_rel']<1e-4,maximum
    write(ROOT/'results'/f'{name}_inference_verification.json',dict(status='PASS',predictions=checked,
          maximum_saved_float32_score_difference=maximum))
    print(name,checked,maximum,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DATASETS,required=True)
    verify_inference(p.parse_args().dataset)
