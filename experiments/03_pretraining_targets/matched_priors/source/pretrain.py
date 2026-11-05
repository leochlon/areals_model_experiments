#!/usr/bin/env python3
"""Matched object-exposure pretraining, with native AREALS and supplied new arms."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source/native'))
from areal_tokens.config import ExperimentConfig
from areal_tokens.data import GeometryDataset,collate_clouds
from areal_tokens.model import GeometryAutoencoder
from areal_tokens.losses import reconstruction_loss
from geometry_data import PatchObjects,QueryObjects,collate_patches,object_order
import train_mesh_geometry as patch
import train_query_geometry as query
from prepare_corpus import write


class NativeObjects(Dataset):
    def __init__(self,split,seed,length,config):
        self.rows=[r for r in json.loads((ROOT/'protocol/objects.json').read_text()) if r['split']==split]
        self.order=object_order(self.rows,split,seed,length)
        self.count=sum(r['split']==split for r in self.rows)
        self.data=GeometryDataset(self.rows,config.model,config.data,seed,training=split=='train')
    def __len__(self):return len(self.order)
    def __getitem__(self,i):
        self.data.set_epoch(int(i)//self.count)
        return self.data[int(self.order[i])]


class RankExamples(Dataset):
    def __init__(self,base,start,steps,batch,rank,world):
        self.base,self.start,self.steps=base,start,steps
        self.batch,self.rank,self.world=batch,rank,world
        self.local=batch//world
    def __len__(self):return (self.steps-self.start)*self.local
    def __getitem__(self,i):
        step,slot=divmod(int(i),self.local)
        return self.base[(self.start+step)*self.batch+self.rank*self.local+slot]


def make_model(arm):
    config=ExperimentConfig.from_dict(json.loads((ROOT/'protocol/native_config.json').read_text()))
    if arm=='areals':return GeometryAutoencoder(config.model),config
    if arm=='patch':return patch.PatchEncoder(),config
    return query.GeometryNet(),config


def dataset(arm,index,split,seed,length,config):
    if arm=='areals':return NativeObjects(split,seed,length,config),collate_clouds
    if arm=='patch':return PatchObjects(index,split,seed,length),collate_patches
    return QueryObjects(index,split,seed,length,evaluation=split!='train'),None


def to_device(batch,device):
    return {k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}


def compute(model,batch,arm,config,multiplier,device,distributed=False,metrics=False):
    if arm=='patch':
        predicted=model(batch['inputs'].float()).double()
        uv,tau=batch['uv'].double(),batch['tau'].double()
        reference=patch.geometry(batch['coeff'].double(),uv,tau,3)
        learned=patch.geometry(predicted,uv,tau,3)
        weights=reference['area'].detach()
        la=patch.weighted_mean((learned['A']-reference['A']).square().sum((-1,-2)),weights)
        lp=patch.weighted_mean((learned['p']-reference['p']).square().sum(-1)*batch['radius'].double()[:,None].square(),weights)
        cp=lp/.01**2-1
        if distributed:
            mean=cp.detach().clone();dist.all_reduce(mean);mean/=dist.get_world_size()
            cp=cp+(mean-cp.detach())
        penalty=(torch.relu(multiplier+cp).square()-multiplier**2)/2
        terms=dict(A=la,p=lp,constraint=cp)
        if metrics:
            with torch.no_grad():
                ev,U=torch.linalg.eigh(reference['A'])
                inv=(U*ev.rsqrt()[...,None,:])@U.transpose(-1,-2)
                relative=inv@learned['A']@inv-torch.eye(3,device=device,dtype=torch.float64)
                error=torch.linalg.eigvalsh(relative).abs().amax(-1)
                terms.update(position_nrmse=lp.sqrt(),relative_A_rms=patch.weighted_mean(error.square(),weights).sqrt(),
                    normal_degrees=patch.weighted_mean(torch.rad2deg(torch.acos((learned['n']*reference['n']).sum(-1).abs().clamp(0,1))),weights))
        return la+penalty,terms
    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
        prediction=model(batch)
    if arm=='areals':
        terms=reconstruction_loss(prediction,batch,config.train)
        if metrics:
            with torch.no_grad():
                error=(prediction['udf'].float()-batch['udf'].float())/batch['scale'].float()
                cosine=(prediction['normals'].float()*batch['query_normals'].float()).sum(-1).abs().clamp(0,1)
                valid=batch['normal_valid'].float()
                terms.update(distance_nrmse=error.square().mean().sqrt(),
                    normal_degrees=(torch.rad2deg(torch.acos(cosine))*valid).sum()/valid.sum().clamp_min(1))
        return terms.pop('loss'),terms
    terms=query.loss_terms(query.as_float(prediction),batch,.05)
    terms['d']=torch.nn.functional.smooth_l1_loss(prediction['d'].float(),batch['d'].float(),beta=.05)
    penalty=((prediction['P'].float()-batch['P'].float()).square()*prediction['P'].new_tensor(query.FRO)).sum(-1)
    valid=batch['near'].float()
    terms['P']=((penalty*valid).sum(1)/valid.sum(1).clamp_min(1)).mean()
    if metrics:
        with torch.no_grad():
            def matrix(a):
                xx,yy,zz,xy,xz,yz=a.unbind(-1)
                return torch.stack([xx,xy,xz,xy,yy,yz,xz,yz,zz],-1).reshape(*a.shape[:-1],3,3)
            pn=torch.linalg.eigh(matrix(prediction['P'].float()))[1][...,-1]
            tn=torch.linalg.eigh(matrix(batch['P'].float()))[1][...,-1]
            angle=torch.rad2deg(torch.acos((pn*tn).sum(-1).abs().clamp(0,1)))
            terms.update(distance_nrmse=(prediction['d'].float()-batch['d'].float()).square().mean().sqrt(),
                         normal_degrees=(angle*valid).sum()/valid.sum().clamp_min(1))
            if 'H2' in batch:
                terms['H_label_floor']=((batch['H']-batch['H2']).square()*batch['H'].new_tensor(query.FRO)).sum(-1).mean()/2
    return terms['d']+terms['P']+terms['H'],terms


class LossModule(torch.nn.Module):
    def __init__(self,model,arm,config,device):
        super().__init__();self.model=model;self.arm=arm;self.config=config;self.device=device
    def forward(self,batch,multiplier):
        loss,terms=compute(self.model,batch,self.arm,self.config,multiplier,self.device,distributed=dist.is_initialized())
        return loss,{k:v.detach() for k,v in terms.items()}


def deterministic():
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def rate(step,steps,peak):
    warmup=min(200,max(1,steps//10))
    if step<=warmup:return peak*step/warmup
    fraction=(step-warmup)/max(1,steps-warmup)
    return peak*(.05+.95*.5*(1+math.cos(math.pi*fraction)))


def run(args):
    rank=int(os.getenv('RANK','0'));world=int(os.getenv('WORLD_SIZE','1'))
    local=int(os.getenv('LOCAL_RANK','0'))
    assert args.batch%world==0
    device=torch.device('cuda',local) if args.device=='cuda' else torch.device('cpu')
    torch.set_num_threads(1)
    deterministic()
    if device.type=='cuda':torch.cuda.set_device(device)
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',timeout=timedelta(hours=2))
    torch.manual_seed(17)
    model,config=make_model(args.arm);model.to(device)
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    peak=1e-4 if args.arm=='patch' else 3e-4
    opt=torch.optim.AdamW(model.parameters(),lr=peak,weight_decay=0 if args.arm=='patch' else .05)
    cfg=dict(arm=args.arm,steps=args.steps,global_object_batch=args.batch,world_size=world,
             seed=17,complete_inputs=True,peak_lr=peak,final_lr_ratio=.05,
             deterministic_attention='math',targets='d,P,H' if args.arm=='hessian' else 'd,P' if args.arm=='areals' else 'A subject to position',
             patches_per_object=8 if args.arm=='patch' else None,
             queries_per_patch=64 if args.arm=='patch' else None)
    start,multiplier=0,0.
    if args.resume:
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        assert state['training_config']==cfg,'Resume configuration differs'
        model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
        start,multiplier=state['step'],state['multiplier']
        torch.set_rng_state(state['rng'][rank]['cpu'])
        if device.type=='cuda':torch.cuda.set_rng_state(state['rng'][rank]['cuda'],device)
    elif (out/'last.pt').exists():raise ValueError('Existing run requires --resume')
    data,collate=dataset(args.arm,args.index,'train',17,args.steps*args.batch,config)
    if rank==0:
        write(out/'config.json',dict(cfg,parameters=sum(p.numel() for p in model.parameters())))
    loss_model=LossModule(model,args.arm,config,device)
    wrapped=DDP(loss_model,device_ids=[local] if device.type=='cuda' else None,find_unused_parameters=True) if world>1 else loss_model
    end=min(args.steps,args.stop_after) if args.stop_after else args.steps
    loader=DataLoader(RankExamples(data,start,end,args.batch,rank,world),batch_size=args.batch//world,
        num_workers=args.workers,collate_fn=collate,pin_memory=device.type=='cuda',
        generator=torch.Generator().manual_seed(17000+rank),persistent_workers=args.workers>0)
    if rank==0:
        vd,vc=dataset(args.arm,args.index,'val',101,args.validation_objects,config)
        validation=DataLoader(vd,batch_size=1,collate_fn=vc,num_workers=0)
        log=(out/'train.jsonl').open('a' if args.resume else 'w',buffering=1)
        print('START',json.dumps(cfg),flush=True)
    begun=time.monotonic()
    for step,batch in enumerate(loader,start+1):
        wrapped.train();batch=to_device(batch,device)
        lr=rate(step,args.steps,peak)
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True)
        loss,terms=wrapped(batch,multiplier)
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at {step}')
        loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        keys=sorted(terms)
        vector=torch.stack([loss.detach().double()]+[terms[k].detach().double() for k in keys])
        if world>1:dist.all_reduce(vector);vector/=world
        if args.arm=='patch':multiplier=max(0.,multiplier+float(vector[1+keys.index('constraint')]))
        if rank==0:log.write(json.dumps(dict(step=step,loss=float(vector[0]),lr=lr,gradient_norm=float(gradient),
                                multiplier=multiplier,**dict(zip(keys,map(float,vector[1:])))))+'\n')
        if step%args.validate_every==0 or step==end:
            if world>1:dist.barrier()
            if rank==0:
                model.eval();values=[];metric_rows=[]
                with torch.no_grad():
                    for vb in validation:
                        vl,mt=compute(model,to_device(vb,device),args.arm,config,multiplier,device,metrics=True)
                        values.append(float(vl))
                        metric_rows.append({k:float(v) for k,v in mt.items()})
                report=dict(step=step,validation_loss=float(np.mean(values)),objects=len(values),
                            elapsed_seconds=time.monotonic()-begun,
                            metrics={k:float(np.mean([m[k] for m in metric_rows])) for k in metric_rows[0]})
                write(out/'validation_latest.json',report);print(json.dumps(report),flush=True)
                with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
            rng=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
            states=[None]*world
            if world>1:dist.all_gather_object(states,rng)
            else:states=[rng]
            if rank==0:
                payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,multiplier=multiplier,
                    training_config=cfg,rng=states,model_config=model.config if args.arm!='areals' else config.model.to_dict() if hasattr(config.model,'to_dict') else vars(config.model))
                if args.arm=='areals':payload.update(version=1,config=config.to_dict())
                elif args.arm=='patch':payload.update(version=patch.VERSION)
                else:payload.update(version=query.VERSION,config={})
                temp=out/'last.tmp';torch.save(payload,temp);temp.replace(out/'last.pt')
                if step==args.steps:
                    temp=out/'final.tmp';torch.save(payload,temp);temp.replace(out/'final.pt')
                    write(out/'complete.json',dict(step=step,object_examples=step*args.batch,seed=17,arm=args.arm))
            if world>1:dist.barrier()
    if rank==0:log.close()
    if world>1:dist.destroy_process_group()


if __name__=='__main__':
    os.umask(0o077)
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['patch','areals','hessian'],required=True)
    p.add_argument('--index',default=str(ROOT/'prepared/index.json'));p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=10626);p.add_argument('--batch',type=int,default=64)
    p.add_argument('--device',default='cuda',choices=['cuda','cpu']);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--validate-every',type=int,default=500);p.add_argument('--validation-objects',type=int,default=128)
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int)
    run(p.parse_args())
