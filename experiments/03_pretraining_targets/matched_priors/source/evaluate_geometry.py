"""Full held-out geometry diagnostic after the fixed pretraining budget."""
import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from prepare_corpus import ROOT,write
from pretrain import make_model,dataset,to_device,compute,deterministic


def run(arm):
    deterministic();torch.set_num_threads(1);device=torch.device('cuda')
    checkpoint=ROOT/f'pretraining/{arm}/final.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['step']==10626
    model,config=make_model(arm);model.load_state_dict(state['model'],strict=True);model.to(device).eval()
    index=ROOT/'prepared/index.json';doc=json.loads(index.read_text())
    result={}
    for split,seed in [('val',101),('test',303)]:
        count=sum(r['split']==split and (arm=='areals' or r['patch' if arm=='patch' else 'query']['status']=='ok') for r in doc['objects'])
        data,collate=dataset(arm,index,split,seed,count,config);rows=[]
        with torch.no_grad():
            for i,batch in enumerate(DataLoader(data,batch_size=1,collate_fn=collate,num_workers=4)):
                loss,metrics=compute(model,to_device(batch,device),arm,config,state['multiplier'],device,metrics=True)
                identity=data.rows[int(data.order[i])]['id'] if arm=='areals' else data.store.rows[int(data.order[i])]['id'] if arm=='patch' else data.corpus.store.rows[int(data.order[i])]['id']
                rows.append(dict(id=identity,loss=float(loss),**{k:float(v) for k,v in metrics.items()}))
        result[split]=dict(objects=count,mean={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='id'},per_object=rows)
        write(ROOT/f'results/geometry_{arm}.json',dict(arm=arm,checkpoint=str(checkpoint),splits=result,
            patch_measure='8 area-sampled patches per object; fresh local queries.'))


if __name__=='__main__':
    os.umask(0o077);p=argparse.ArgumentParser();p.add_argument('--arm',required=True);run(p.parse_args().arm)
