import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(os.environ['STUDY_ROOT'])


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu-index',type=int)
    args=parser.parse_args()
    matrix=read(ROOT/'matrix.json')
    norms,by_case={},defaultdict(list)
    for job in matrix:
        folder=ROOT/'runs'/job['name']
        complete=read(folder/'complete.json')
        key=f'{job["seed"]}/{job["budget"]}'
        if key in norms:
            assert norms[key]==complete['normalisation']
        norms[key]=complete['normalisation']
        test=read(folder/'test.json')
        for row in test['rows']:
            by_case[row['case']].append((job['name'],row))
    maximum,count=0.,0
    for i,(case,rows) in enumerate(sorted(by_case.items())):
        assert len(rows)==60
        with np.load(ROOT/'data/fields'/(case+'.npz')) as f:
            truth,weights=f['values'].astype(np.float64),f['weights'].astype(np.float64)
        spread=np.sqrt(np.average((truth-np.average(truth,weights=weights))**2,weights=weights))
        for name,row in rows:
            k=row['sensors']
            with np.load(ROOT/'runs'/name/f'predictions_s{k}'/(case+'.npz')) as pred:
                assert int(pred['sensors'])==k
                prediction=pred['prediction'].astype(np.float64)
                observed=pred['observed20'] if k else []
            hidden=np.ones(len(truth),bool);hidden[observed]=False
            score=np.sqrt(np.average((prediction[hidden]-truth[hidden])**2,weights=weights[hidden]))/spread
            gap=abs(score-row['nrmse'])
            assert math.isfinite(gap) and gap<1e-6,(name,case,k,gap)
            maximum=max(maximum,float(gap));count+=1
        if (i+1)%100==0:
            print('REPLAYED_CASES',i+1,'/',len(by_case),flush=True)
    result=dict(passed=True,scored_rows=count,paired_training_normalisation_identical=True,
        max_metric_difference=maximum,metric_tolerance=1e-6)
    if args.gpu_index is not None:
        sys.path.insert(0,str(ROOT/'source'));sys.path.insert(0,str(ROOT/'model_src'))
        import torch
        import study
        torch.set_num_threads(2);torch.backends.cuda.enable_cudnn_sdp(False)
        job=matrix[args.gpu_index];folder=ROOT/'runs'/job['name']
        record=next(r for r in sorted(study.records(),key=lambda r:r['id']) if r['split']=='test')
        field,prepared=study.field(record),study.prepared(record)
        observed=[]
        for k in (0,20):
            rng=np.random.default_rng(study.stable(f'eval-v1|test|{record["id"]}|{k}'))
            ix=np.sort(rng.choice(len(field.points),k,replace=False,p=field.weights/field.weights.sum()))
            observed.append((k,ix,field.values[ix].copy()))
        old={(r['case'],r['sensors']):r for r in read(folder/'test.json')['rows']}
        gaps=[]
        for k in (0,20):
            model,_,state=study.fb.load_field_encoder(folder/f'best_s{k}.pt','cuda')
            rows=study.evaluate(model,[(record,field,prepared,observed)],state['benchmark'])
            row=next(r for r in rows if r['sensors']==k)
            gap=abs(row['nrmse']-old[(row['case'],k)]['nrmse'])
            assert math.isfinite(gap) and gap<1e-5,gap
            gaps.append(gap)
        result.update(gpu_checkpoint_replay=job['name'],gpu_score_difference=max(gaps),
                      gpu_replay_tolerance=1e-5,gpu=torch.cuda.get_device_name())
    (ROOT/'results/replay.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
