import argparse
from collections import defaultdict
import json
import sys
import time

import numpy as np
import torch
from torch_geometric.data import Data

from common import ROOT, ORIGINAL, DATA, UPSTREAM, SEEDS, VARIANTS, records, run_dir, write_json

sys.path.insert(0,str(ROOT/'source'))
import field_benchmark as fb
import gabi_adapter as ga
sys.path.insert(0,str(UPSTREAM))
from models import GCNGeomAutoencoder

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=VARIANTS,required=True)
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--split',choices=['val','test'],required=True)
    args=parser.parse_args()
    path=run_dir(args.variant,args.seed)
    complete=json.loads((path/'complete.json').read_text())
    assert complete['epochs']==455 and complete['updates']==10010
    checkpoint=str(path/'best.pt')
    output=path/args.split
    output.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    config=json.loads((ROOT/'config.json').read_text())
    model=GCNGeomAutoencoder(**config['data_dims'],**config['hyperparameters'],**config['model_settings'])
    state=torch.load(path/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    model.cuda().eval()
    stats=state['trainset_stats']
    xm,xs=float(stats['x']['mean'].reshape(-1)[0]),float(stats['x']['std'].reshape(-1)[0])
    pm,ps=stats['pos']['mean'].double().numpy(),stats['pos']['std'].double().numpy()
    meta,masks=fb.read_observations(ORIGINAL/'bench',args.split)
    by_car=fb.by_car(meta['entries'])
    groups=defaultdict(list)
    for row in records(args.split):
        groups[row['grid_id']].append(row)
    def generator(identity,what):
        seed=fb.stable_seed(args.seed,f'gabi|{identity}|{what}',0) % (2**62)
        return torch.Generator(device='cuda').manual_seed(seed)
    all_scores=[]
    for grid,rows in sorted(groups.items()):
        info=output/(grid+'.json')
        archive=output/(grid+'.npz')
        if info.exists() and archive.exists():
            saved=json.loads(info.read_text())
            assert saved['checkpoint']==checkpoint
            all_scores.extend(saved['scores'])
            continue
        with np.load(ROOT/'graphs'/(grid+'.npz')) as cache:
            points,edge_index=cache['points'].copy(),cache[args.variant].copy()
        data=Data(pos=torch.as_tensor((points-pm)/ps,dtype=torch.float32),
                  edge_index=torch.as_tensor(edge_index),x=torch.zeros(len(points),1),
                  boundary_encoding=torch.zeros(len(points),0))
        problems,noises,entries,fields=[],[],[],{}
        for row in sorted(rows,key=lambda r:r['id']):
            identity=row['id']
            field=fb.load_field(DATA/'fields',identity)
            assert np.array_equal(field.points,points)
            fields[identity]=field
            for entry in by_car[identity]:
                observed=masks[entry['key']]
                entries.append((row,entry,observed))
                if len(observed):
                    y=torch.as_tensor((field.values[observed]-xm)/xs,dtype=torch.float32,device='cuda')
                    problems.append((torch.as_tensor(observed,device='cuda'),y))
                    noises.append(generator(identity,entry['key']))
                else:
                    problems.append(None)
                    noises.append(None)
        assert torch.count_nonzero(data.x)==0
        started=time.monotonic()
        means=ga.abc_pool(model,data,problems,generator(grid,'prior'),noises,50000,100,.01,15,'cuda')
        predictions={entry['key']:(mean.double().cpu().numpy()*xs+xm).astype(np.float32)
                     for (_,entry,_),mean in zip(entries,means,strict=True)}
        scores=[]
        for row,entry,observed in entries:
            prediction=predictions[entry['key']].astype(np.float64)
            assert np.isfinite(prediction).all()
            metrics=fb.problem_metrics(fields[row['id']],observed,prediction)
            assert all(np.isfinite(v) for v in metrics.values()), (grid,entry,metrics)
            scores.append(dict(case=row['id'],grid_id=grid,physical_geometry=row['family'],
                               setting=entry['setting'],seed=args.seed,variant=args.variant,**metrics))
        with archive.with_suffix('.tmp').open('wb') as handle:
            np.savez_compressed(handle,**predictions)
        archive.with_suffix('.tmp').replace(archive)
        write_json(info,dict(grid_id=grid,checkpoint=checkpoint,seconds=time.monotonic()-started,scores=scores))
        all_scores.extend(scores)
        print(json.dumps({'event':'SCORED','variant':args.variant,'seed':args.seed,'split':args.split,
                          'grid':grid,'seconds':time.monotonic()-started}),flush=True)
    assert len(all_scores)==144*8
    summary={setting:{metric:float(np.mean([r[metric] for r in all_scores if r['setting']==setting]))
                     for metric in ['nrmse','force_rel']} for setting in sorted({r['setting'] for r in all_scores})}
    write_json(output/'scores.json',dict(status='PASS',variant=args.variant,seed=args.seed,split=args.split,
               checkpoint=checkpoint,summary=summary,rows=all_scores))
    print(json.dumps({'event':'EVALUATION_COMPLETE','split':args.split,'summary':summary}),flush=True)

if __name__=='__main__':
    main()
