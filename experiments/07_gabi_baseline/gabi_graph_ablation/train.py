import argparse
import json
import random
import sys
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from common import ROOT, UPSTREAM, VARIANTS, SEEDS, run_dir, write_json
from dataset import GraphDataset, EpochSampler

sys.path.insert(0,str(UPSTREAM))
from datatools import norm_data
from models import GCNGeomAutoencoder
from training_scripts.train_single_gpu import compute_loss

def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temporary)
    temporary.replace(path)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=VARIANTS,required=True)
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    path=(ROOT/'pilot'/f'{args.variant}_seed{args.seed}') if args.pilot else run_dir(args.variant,args.seed)
    path.mkdir(parents=True,exist_ok=True)
    if (path/'complete.json').exists():
        return
    config=json.loads((ROOT/'config.json').read_text())
    epochs=2 if args.pilot else int(config['hyperparameters']['epochs'])
    torch.set_num_threads(4)
    torch.multiprocessing.set_sharing_strategy('file_system')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model=GCNGeomAutoencoder(**config['data_dims'],**config['hyperparameters'],**config['model_settings'])
    stats=torch.load(ROOT/'normalization.pt',map_location='cpu',weights_only=False)
    model.trainset_stats=stats
    model.cuda()
    train,val=GraphDataset('train',args.variant),GraphDataset('val',args.variant)
    assert train.get_data_dims_dict()==config['data_dims']
    sampler=EpochSampler(len(train),args.seed)
    train_loader=DataLoader(train,batch_size=100,sampler=sampler,num_workers=4,persistent_workers=True,pin_memory=False)
    val_loader=DataLoader(val,batch_size=100,shuffle=False,num_workers=4,persistent_workers=True,pin_memory=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=config['hyperparameters']['lr_decay'])
    start_epoch,best_loss,best_epoch,history=0,float('inf'),None,[]
    if (path/'last_full.pt').exists():
        previous=torch.load(path/'last_full.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(previous['model_state_dict'])
        optimizer.load_state_dict(previous['optimizer_state_dict'])
        scheduler.load_state_dict(previous['scheduler_state_dict'])
        start_epoch,best_loss,best_epoch=previous['epoch'],previous['best_loss'],previous['best_epoch']
        history=previous['history']
    write_json(path/'initialization.json',dict(seed=args.seed,variant=args.variant,device=torch.cuda.get_device_name()))
    print(json.dumps({'event':'START','variant':args.variant,'seed':args.seed,'epochs':epochs}),flush=True)
    for epoch in range(start_epoch+1,epochs+1):
        started=time.monotonic()
        sampler.epoch=epoch
        torch.manual_seed(20261028+args.seed*10000+epoch)
        torch.cuda.manual_seed_all(20261028+args.seed*10000+epoch)
        torch.cuda.reset_peak_memory_stats()
        train_metrics=np.zeros(3)
        model.train()
        for batch in train_loader:
            batch=norm_data(batch,stats).cuda()
            optimizer.zero_grad()
            losses=compute_loss(model(batch))
            assert all(torch.isfinite(v) for v in losses), (epoch,'train')
            losses[0].backward()
            optimizer.step()
            train_metrics += [v.item() for v in losses]
        train_metrics /= len(train_loader)
        scheduler.step()
        validation=np.zeros(3)
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                losses=compute_loss(model(norm_data(batch,stats).cuda()))
                assert all(torch.isfinite(v) for v in losses), (epoch,'val')
                validation += [v.item() for v in losses]
        validation /= len(val_loader)
        weights=dict(model_state_dict=model.state_dict(),trainset_stats=stats,epoch=epoch)
        if validation[0]<best_loss:
            best_loss,best_epoch=float(validation[0]),epoch
            save(path/'best.pt',weights)
        row=dict(epoch=epoch,updates=epoch*len(train_loader),train=train_metrics.tolist(),validation=validation.tolist(),
                 best_validation=best_loss,best_epoch=best_epoch,lr=scheduler.get_last_lr()[0],
                 peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-started)
        history.append(row)
        save(path/'last_full.pt',dict(**weights,optimizer_state_dict=optimizer.state_dict(),
             scheduler_state_dict=scheduler.state_dict(),best_loss=best_loss,best_epoch=best_epoch,history=history))
        write_json(path/'history.json',history)
        print(json.dumps(row),flush=True)
    write_json(path/'complete.json',dict(status='PASS',variant=args.variant,seed=args.seed,epochs=epochs,
               updates=epochs*len(train_loader),selected_epoch=best_epoch))

if __name__=='__main__':
    main()
