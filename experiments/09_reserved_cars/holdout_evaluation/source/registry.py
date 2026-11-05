import json
from pathlib import Path

import torch

ROOT = Path('/workspace/runs/areals_fresh_car_holdout_20261024')
AUDIT = Path('/workspace/runs/areals_architecture_controls_20261024')
OLD = Path('/workspace/runs/areal_drivaerml_420_epoch14_20260927T162928Z')
ARMS = ['reference','ordinary_attention','unanchored_latents','oriented_normal_target',
        'unoriented_vector_target','anchored_no_rotary','unanchored_no_rotary']


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def main():
    split=json.loads((OLD/'split.json').read_text())
    train,val,test=map(set,[split['train'],split['validation'],split['benchmark_test_excluded']])
    assert (len(train),len(val),len(test))==(420,16,48)
    assert not(train&val or train&test or val&test)
    models=[]
    def add(family,arm,seed,name,path,primary=False,lr=None):
        state=torch.load(path,map_location='cpu',weights_only=False)
        if family!='frozen':
            assert state['benchmark']['train_cars']==420
            if family=='architecture':
                assert state['benchmark']['seed']==0
                original=torch.load(state['benchmark']['base_checkpoint'],map_location='cpu',weights_only=True)
                assert original['seed']==seed
            else:assert state['benchmark']['seed']==seed
            assert state['benchmark']['normalise']=='dataset'
            ancestry={k:state['benchmark'].get(k) for k in ['base_checkpoint','mean','std','scratch','lr','new_lr','seed']}
        else:
            run=json.loads((path.parent/'run.json').read_text())
            assert run['train_cars']==420 and run['val_cars']==16 and run['total_updates']==1060
            ancestry=dict(geometry_checkpoint=run['config']['comparison'].get('areal_checkpoint'))
        models.append(dict(task=len(models),family=family,arm=arm,seed=seed,name=name,
                      checkpoint=str(path),primary=primary,
                      backbone_lr=lr,ancestry=ancestry))
    # Primary comparisons use choices fixed from the old 16-car validation only.
    for seed in range(3):
        for arm,lr in [('areals','1e-4'),('pointmae','1e-3')]:
            name=f'online_{arm}_lr{lr}_seed{seed}'
            add('online',arm,seed,name,AUDIT/'architecture_transfer/bench/models'/name/'field_best.pt',True,lr)
        for arm in ['pretrained','scratch']:
            name=f'{arm}_seed{seed}'
            add('replication',arm,seed,name,AUDIT/'replication/bench/models'/name/'field_best.pt',True)
    for seed in range(3):
        for arm,lr in [('areals','1e-3'),('pointmae','1e-4')]:
            name=f'online_{arm}_lr{lr}_seed{seed}'
            add('online',arm,seed,name,AUDIT/'architecture_transfer/bench/models'/name/'field_best.pt',False,lr)
    for arm in ARMS:
        for seed in range(3):
            name=f'{arm}_seed{seed}'
            add('architecture',arm,seed,name,AUDIT/'architecture_transfer/bench/models'/name/'field_best.pt')
    for arm in ['native','areal','native_normals']:
        for seed in range(3):
            name=f'{arm}_seed{seed}'
            add('frozen',arm,seed,name,AUDIT/'frozen_replication/runs'/name/'final_live.pt')
    assert len(models)==len({m['checkpoint'] for m in models})==48
    protocol=dict(run='areals_fresh_car_holdout_20261024',
          train_ids=sorted(train),validation_ids=sorted(val),test_ids=sorted(test),
          primary_comparisons=['Pretrained versus scratch field completion, three seeds, both learning rates 1e-3, same initial head, original 25 visibility settings equally weighted.',
                               'Geometry-only online encoders: AREALS backbone lr1e-4 versus Point-MAE backbone lr1e-3, chosen from old validation; three seeds, same fresh decoder/head.'],
          secondary=['Both learning rates for each online encoder.',
                     'Seven architecture variants, three seeds each; geometry-only full-surface nRMSE.',
                     'Frozen AREALS versus tutorial AeroJEPA XYZ and XYZ+normals, three seeds; fixed 2048-query raw-Cp rL2.',
                     'Shape-only (no-observation) diagnostic for the pretrained/scratch completion models, excluded from the 25-setting aggregate.'],
          source_prefix='datasets/DrivAerML/surface/d8184e249d01bf5230ab05627ce5c03e186dafda9c11ddf33df549a7504e7835/')
    write(ROOT/'models.json',models)
    write(ROOT/'protocol.json',protocol)
    print(json.dumps(dict(models=len(models),test_cars=len(test))))


if __name__=='__main__':main()
