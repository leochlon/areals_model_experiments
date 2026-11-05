"""Held-out-car pressure comparison with independent validation LR selection."""
import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
from torch import nn

from prepare_corpus import ROOT,write
from prepare_windsor import BENCH
from pretrain import make_model,deterministic
from areal_tokens.field import FieldEncoder
from pressure_model import PressureModel
import train_query_geometry as query

BUDGETS=[18,35,69,137,274]
RATES=[2e-5,2e-4,1e-3]
ARMS=['patch','areals','hessian']


class Models(nn.Module):
    def __init__(self,arm,initialization,seed,checkpoint=None):
        super().__init__();self.arm=arm
        torch.manual_seed(seed)
        base,config=make_model(arm)
        if arm=='patch':
            self.net=PressureModel(base.config,condition_dim=0,encoder_chunk=1024,gradient_checkpointing=False)
            # Constructor and new head are identical in both initialization arms.
            if initialization=='pretrained':
                state=torch.load(checkpoint,map_location='cpu',weights_only=False)
                self.net.encoder.load_state_dict(state['model'],strict=True)
            self.backbone=[p for p in self.net.encoder.parameters()]
        elif arm=='areals':
            if initialization=='pretrained':
                state=torch.load(checkpoint,map_location='cpu',weights_only=False)
                base.load_state_dict(state['model'],strict=True)
            self.net=FieldEncoder(base)
            self.backbone=list(self.net.base.parameters())
        else:
            cfg=dict(base.config,extra_inputs=2,field_outputs=1)
            self.net=query.GeometryNet(**cfg)
            with torch.no_grad():self.net.embed[0].weight[:,-2:]=0
            if initialization=='pretrained':
                state=torch.load(checkpoint,map_location='cpu',weights_only=False)
                weights=dict(state['model']);first=weights['embed.0.weight']
                weights['embed.0.weight']=torch.cat((first,first.new_zeros(first.shape[0],2)),1)
                missing,unexpected=self.net.load_state_dict(weights,strict=False)
                assert not unexpected and set(missing)=={'out.field.weight','out.field.bias'}
            self.backbone=[p for name,p in self.net.named_parameters() if not name.startswith('out.field.')]
        self.old_ids={id(p) for p in self.backbone}
        self.new_parameters=[p for p in self.parameters() if id(p) not in self.old_ids]

    def encode(self,car,device):
        def t(v):return torch.as_tensor(v,device=device,dtype=torch.float32)
        if self.arm=='patch':
            scene={k:t(car['patch'][k]) for k in ['inputs','origin','frame','radius']}
            scene['scale']=t(car['meta']['scale'])
            return self.net.encode_scene(scene,t([]))
        a={k:t(v)[None] for k,v in car['geometry'].items()}
        if self.arm=='areals':
            a['complete']=torch.ones(1,1,device=device)
            zeros=t(np.zeros((1,len(car['geometry']['points']),1),np.float32))
            anchors=zeros.new_zeros(1,len(car['geometry']['anchors']),1)
            return self.net.encode(a,zeros,zeros,anchors,anchors)
        return self.net.encode(a['points'],a['normals'],a['point_weights'],
            torch.as_tensor(car['anchor_index'],device=device,dtype=torch.long)[None],a['anchor_weights'])

    def decode(self,encoded,world,car,device):
        xyz=torch.as_tensor(world,device=device,dtype=torch.float32)
        if self.arm=='patch':
            xyz=(xyz-xyz.new_tensor(car['meta']['center']))/car['meta']['scale']
            return self.net.decode(encoded,xyz,standardized=True)
        if self.arm=='areals':return self.net.predict(encoded,xyz[None])[0]
        xyz=(xyz-xyz.new_tensor(car['geometry']['center']))/float(car['geometry']['scale'][0])
        tokens,pos,bias=encoded
        return self.net.decode(xyz[None],tokens,pos,bias,xyz.new_ones(1))['field'][0,:,0].float()


class Cars:
    def __init__(self,index=None):
        doc=json.loads(Path(index or ROOT/'windsor/index.json').read_text())
        self.rows={r['id']:r for r in doc['records']};self.cache=OrderedDict()

    def get(self,identity,arm):
        key=identity,arm
        if key not in self.cache:
            row=self.rows[identity]
            with np.load(BENCH/'data/fields'/f'{identity}.npz') as z:
                field={k:z[k].copy() for k in ['points','values','weights']}
            field['values']=field['values'].reshape(-1).astype(np.float64)
            field['weights']=field['weights'].reshape(-1).astype(np.float64)
            assert np.isfinite(field['values']).all() and np.all(field['weights']>0)
            field['weights']/=field['weights'].sum()
            result=dict(id=identity,split=row['split'],field=field)
            if arm=='patch':
                with h5py.File(row['pack'],'r') as f:
                    group=f[row['group']]
                    result['patch']={k:group[k][:].astype(np.float32) for k in ['inputs','origin','frame','radius']}
                result['meta']=row['metadata']
            else:
                with np.load(BENCH/'cache/geometry'/f'{identity}.npz') as z:
                    result['geometry']={k[2:]:z[k].copy() for k in z.files if k.startswith('a_')}
                    result['anchor_index']=z['anchor_index'].copy()
            self.cache[key]=result
            if len(self.cache)>48:self.cache.popitem(last=False)
        self.cache.move_to_end(key);return self.cache[key]


def stats(ids):
    first=[];second=[]
    for identity in ids:
        with np.load(BENCH/'data/fields'/f'{identity}.npz') as z:
            v=z['values'].reshape(-1).astype(np.float64);w=z['weights'].reshape(-1).astype(np.float64);w/=w.sum()
            first.append(float(w@v));second.append(float(w@(v*v)))
    mean=float(np.mean(first));std=float(np.sqrt(np.mean(second)-mean**2))
    assert std>0
    return dict(mean=mean,std=std,training_ids=ids)


@torch.no_grad()
def evaluate(model,cars,ids,normalization,device):
    model.eval();results=[]
    for identity in ids:
        car=cars.get(identity,model.arm);field=car['field']
        torch.cuda.synchronize() if device.type=='cuda' else None
        begun=time.monotonic()
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=False):
            encoded=model.encode(car,device);pred=[]
            for i in range(0,len(field['values']),2048):
                pred.append(model.decode(encoded,field['points'][i:i+2048],car,device).float().cpu().numpy())
        torch.cuda.synchronize() if device.type=='cuda' else None
        elapsed=time.monotonic()-begun
        pred=np.concatenate(pred).astype(np.float64)*normalization['std']+normalization['mean']
        weights,truth=field['weights'],field['values']
        rmse=float(np.sqrt(weights@((pred-truth)**2)))
        spread=float(np.sqrt(weights@((truth-weights@truth)**2)))
        assert spread>0 and np.isfinite(pred).all()
        results.append(dict(id=identity,nrmse=rmse/spread,rmse=rmse,reference_std=spread,
                            encode_and_full_car_prediction_seconds=elapsed,queries=len(truth)))
    return dict(mean_nrmse=float(np.mean([r['nrmse'] for r in results])),cars=results)


def matrix():
    rows=[]
    for arm in ARMS:
        for initialization in ['pretrained','scratch']:
            for budget in BUDGETS:
                for i,lr in enumerate(RATES):
                    rows.append(dict(arm=arm,initialization=initialization,budget=budget,seed=0,lr=lr,lr_index=i,
                                     name=f'{arm}/{initialization}/n{budget}/seed0-lr{i}'))
    return rows


def train(row,args):
    deterministic();torch.set_num_threads(1)
    device=torch.device(args.device)
    subsets=json.loads((ROOT/'windsor_protocol/subsets.json').read_text())
    train_ids=['WindsorML__run_150'] if args.software_check else subsets[f'{row["seed"]}/{row["budget"]}']
    cars=Cars(args.index)
    assert all(cars.rows[k]['split']=='train' for k in train_ids)
    val_ids=sorted(k for k,r in cars.rows.items() if r['split']=='val')
    if args.software_check:val_ids=val_ids[:1]
    assert val_ids
    checkpoint=ROOT/f'pretraining/{row["arm"]}/final.pt'
    if args.checkpoint:checkpoint=Path(args.checkpoint)
    model=Models(row['arm'],row['initialization'],row['seed'],checkpoint).to(device)
    normalization=stats(train_ids)
    config=dict(row,updates=args.steps,batch=8,queries=2048,new_lr=2e-4,validation_every=args.validate_every,
        normalization=normalization,complete_geometry=True,sensors=0,deterministic_attention='math',
        precision='fp32',optimizer='AdamW',weight_decay=0.,schedule='constant',early_stopping=False,
        geometry_checkpoint=str(checkpoint) if row['initialization']=='pretrained' else None)
    out=ROOT/(args.out or 'field/runs/'+row['name']);out.mkdir(parents=True,exist_ok=True)
    optimizer=torch.optim.AdamW([dict(params=model.backbone,lr=row['lr']),dict(params=model.new_parameters,lr=2e-4)],weight_decay=0.)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda step:1.)
    start,best=0,float('inf')
    if (out/'last.pt').exists():
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        assert state['config']==config
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start,best=state['step'],state['best_validation']
        torch.set_rng_state(state['rng_cpu'])
        if device.type=='cuda':torch.cuda.set_rng_state(state['rng_cuda'])
    write(out/'config.json',config)
    log=(out/'training.jsonl').open('a' if start else 'w',buffering=1)
    end=min(args.steps,args.stop_after) if args.stop_after else args.steps
    for step in range(start+1,end+1):
        model.train();optimizer.zero_grad(set_to_none=True);losses=[]
        rng=np.random.default_rng([row['seed'],step,811])
        chosen=rng.choice(train_ids,8,replace=True)
        for slot,identity in enumerate(chosen):
            car=cars.get(identity,row['arm']);field=car['field']
            q=np.random.default_rng([row['seed'],step,slot,912]).choice(len(field['values']),2048,replace=True,p=field['weights'])
            target=torch.as_tensor((field['values'][q]-normalization['mean'])/normalization['std'],device=device,dtype=torch.float32)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=False):
                encoded=model.encode(car,device);pred=model.decode(encoded,field['points'][q],car,device)
                loss=(pred.float()-target).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError((row['name'],step,identity))
            (loss/8).backward();losses.append(float(loss.detach()))
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
        optimizer.step();scheduler.step()
        log.write(json.dumps(dict(step=step,loss=float(np.mean(losses)),gradient_norm=float(grad),lr=[g['lr'] for g in optimizer.param_groups]))+'\n')
        if step%args.validate_every==0 or step==end:
            report=evaluate(model,cars,val_ids,normalization,device)
            report.update(step=step)
            with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
            improved=report['mean_nrmse']<best
            if improved:best=report['mean_nrmse']
            state=dict(config=config,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                step=step,best_validation=best,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state() if device.type=='cuda' else None)
            temp=out/'last.tmp';torch.save(state,temp);temp.replace(out/'last.pt')
            if improved:
                temp=out/'best.tmp';torch.save(state,temp);temp.replace(out/'best.pt')
                write(out/'best_validation.json',report)
            print(row['name'],step,report['mean_nrmse'],flush=True)
    log.close()
    if end==args.steps:write(out/'complete.json',dict(config=config,step=end,best_validation=best))


def select():
    groups={}
    for row in matrix():
        path=ROOT/'field/runs'/row['name'];complete=json.loads((path/'complete.json').read_text())
        assert complete['step']==8000
        key=f'{row["arm"]}/{row["initialization"]}/{row["budget"]}'
        groups.setdefault(key,[]).append(dict(row,validation=complete['best_validation'],checkpoint=str(path/'best.pt')))
    winners={key:min(rows,key=lambda r:(r['validation'],r['lr'])) for key,rows in groups.items()}
    follow=[]
    for row in winners.values():
        for seed in [1,2]:
            follow.append(dict({k:row[k] for k in ['arm','initialization','budget','lr','lr_index']},
                seed=seed,name=f'{row["arm"]}/{row["initialization"]}/n{row["budget"]}/seed{seed}-selected'))
    write(ROOT/'field/selections.json',dict(selection='minimum full-validation mean per-car area-weighted nRMSE; tie: lower encoder LR',winners=winners,followup=follow))


def selected_runs():
    """The 90 selected fits: 30 validation winners at seed 0, then their seeds 1 and 2."""
    selected=json.loads((ROOT/'field/selections.json').read_text())
    rows=list(selected['winners'].values())+selected['followup']
    assert len(rows)==90
    return [dict(row,checkpoint=str(ROOT/'field/runs'/row['name']/'best.pt')) for row in rows]


def score(index,args):
    row=selected_runs()[index]
    assert json.loads((Path(row['checkpoint']).parent/'complete.json').read_text())['step']==8000
    deterministic();torch.set_num_threads(1);device=torch.device(args.device)
    state=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
    model=Models(row['arm'],'scratch',row['seed']).to(device);model.load_state_dict(state['model'],strict=True)
    cars=Cars();ids=sorted(k for k,r in cars.rows.items() if r['split']=='test')
    assert len(ids)==35
    result=evaluate(model,cars,ids,state['config']['normalization'],device)
    write(ROOT/f'field/test/{index:03d}.json',dict(row=row,metrics=result))


def report():
    rows=[json.loads((ROOT/f'field/test/{i:03d}.json').read_text()) for i in range(90)]
    values={}
    expected_ids=[c['id'] for c in rows[0]['metrics']['cars']]
    for r in rows:
        assert [c['id'] for c in r['metrics']['cars']]==expected_ids
        row=r['row'];values[row['arm'],row['initialization'],row['budget'],row['seed']]=r['metrics']
    comparison=[]
    rng=np.random.default_rng(2091)
    for arm in ARMS:
        for budget in BUDGETS:
            by_arm={k:np.array([[c['nrmse'] for c in values[arm,k,budget,s]['cars']] for s in range(3)]) for k in ['pretrained','scratch']}
            delta=by_arm['scratch']-by_arm['pretrained'];draws=[]
            for _ in range(5000):draws.append(delta[rng.integers(0,3,3)][:,rng.integers(0,35,35)].mean())
            comparison.append(dict(arm=arm,budget=budget,pretrained_mean=float(by_arm['pretrained'].mean()),scratch_mean=float(by_arm['scratch'].mean()),
                pretrained_seeds=by_arm['pretrained'].mean(1).tolist(),scratch_seeds=by_arm['scratch'].mean(1).tolist(),
                gain_nrmse=float(delta.mean()),gain_paired_bootstrap_95=np.quantile(draws,[.025,.975]).tolist(),
                prediction_seconds_per_car={k:float(np.mean([c['encode_and_full_car_prediction_seconds'] for seed in range(3) for c in values[arm,k,budget,seed]['cars']])) for k in ['pretrained','scratch']}))
    thresholds={f'{a}/{k}':next((b for b in BUDGETS if np.mean([values[a,k,b,s]['mean_nrmse'] for s in range(3)])<=.25),None) for a in ARMS for k in ['pretrained','scratch']}
    write(ROOT/'results/comparison.json',dict(comparisons=comparison,threshold_nrmse=.25,smallest_measured_budget=thresholds,
        null_threshold='not reached within 274 cars; no extrapolation',
        timing_scope='Encode complete geometry and predict all 32000 native pressure queries per car, including device transfers; no file loading; first car retained.'))


if __name__=='__main__':
    os.umask(0o077)
    p=argparse.ArgumentParser();p.add_argument('command',choices=['tune','followup','select','score','report'])
    p.add_argument('--task',type=int,default=0);p.add_argument('--steps',type=int,default=8000);p.add_argument('--validate-every',type=int,default=250)
    p.add_argument('--device',default='cuda');p.add_argument('--out');p.add_argument('--index');p.add_argument('--checkpoint');p.add_argument('--stop-after',type=int)
    p.add_argument('--software-check',action='store_true')
    a=p.parse_args()
    if a.command in ['tune','followup']:
        rows=matrix() if a.command=='tune' else json.loads((ROOT/'field/selections.json').read_text())['followup'];train(rows[a.task],a)
    elif a.command=='select':select()
    elif a.command=='score':score(a.task,a)
    else:report()
