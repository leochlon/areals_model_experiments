import contextlib
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path('/workspace/runs/areals_decoder_transfer_20261024')
OLD=Path('/workspace/runs/areals_architecture_controls_20261024')
DATA=Path('/workspace/runs/areal_drivaerml_420_epoch14_20260927T162928Z')
sys.path.insert(0,'/workspace/runs/field_benchmark_4gpu_20261023')
import field_benchmark as fb


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def run_check(fb,argv,log):
    """The harness's hidden-value check, with its output in `log`; the scoring scripts count its PASS lines."""
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as stream,contextlib.redirect_stdout(stream):
        fb.main(argv)


def main():
    manifest=json.loads((DATA/'manifest.json').read_text())
    # 40 of the 420 training cars, in a fixed hash order, become the test cars of this study.
    train=sorted((r for r in manifest['records'] if r['split']=='train'),
                 key=lambda r:hashlib.sha256(('arch_rca_confirm_20261024:'+r['id']).encode()).hexdigest())
    assert len(train)==420
    held={r['id'] for r in train[:40]}
    for record in manifest['records']:
        if record['id'] in held:record['split']='test'
    groups={s:[r['id'] for r in manifest['records'] if r['split']==s] for s in ['train','val','test']}
    assert [len(groups[s]) for s in groups]==[380,16,40]
    assert not(set(groups['train'])&set(groups['test']))
    write(ROOT/'car_manifest.json',manifest)
    (ROOT/'production_config.json').write_bytes((OLD/'production_config.json').read_bytes())
    # car_manifest.json keeps the relative geometry paths geometry/<id>.npz of the 420-car data.
    for link,target in [(ROOT/'architecture_manifest.json',OLD/'architecture_manifest.json'),(ROOT/'geometry',DATA/'geometry')]:
        if not link.exists():link.symlink_to(target)
    tasks=[]
    for seed in range(3):
        for arm in ['reference','ordinary_attention']:
            tasks.append(dict(task=len(tasks),arm=arm,seed=seed,group='isotropic_'+arm,
                pretrained=str(ROOT/'results/architecture'/arm/f'seed{seed}/best.pt'),retrain_geometry=True))
    for arm in ['reference','ordinary_attention','oriented_normal_target']:
        for seed in range(3):
            path=OLD/'results/architecture'/arm/f'seed{seed}/best.pt'
            tasks.append(dict(task=len(tasks),arm=arm,seed=seed,group='original_'+arm,
                pretrained=str(path),retrain_geometry=False))
    for task in tasks:task['name']=task['group']+'_seed'+str(task['seed'])
    # The vector-normal arm with the same zero-weight head (unit-z bias), so that the head initialisation is not
    # confounded with the normal target.
    for seed in range(3):
        tasks.append(dict(task=len(tasks),arm='oriented_normal_target',seed=seed,
            group='constant_oriented_normal_target',name=f'constant_oriented_normal_target_seed{seed}',
            pretrained=str(ROOT/'results/architecture/oriented_normal_target'/f'seed{seed}/best.pt'),
            retrain_geometry=True))
    protocol=dict(splits=groups,primary_contrasts=[['isotropic_reference','isotropic_ordinary_attention'],
        ['isotropic_reference','constant_oriented_normal_target'],['isotropic_reference','original_reference'],
        ['constant_oriented_normal_target','original_oriented_normal_target']])
    write(ROOT/'confirmation_protocol.json',protocol);write(ROOT/'tasks.json',tasks)
    bench=ROOT/'bench'
    for split in ['val','test']:
        fb.main(['observations','--manifest',str(ROOT/'car_manifest.json'),'--fields',str(DATA/'fields'),
                 '--bench',str(bench),'--split',split,'--unobserved'])
        meta,arrays=fb.read_observations(bench,split)
        entries=[e for e in meta['entries'] if e['setting']=='unobserved']
        arrays={e['key']:arrays[e['key']] for e in entries}
        meta.update(entries=entries,settings=[s for s in meta['settings'] if s['name']=='unobserved'])
        np.savez_compressed(bench/'observations'/f'{split}.npz',**arrays)
        write(bench/'observations'/f'{split}.json',meta)
    print('decoder study prepared',json.dumps(dict(counts={s:len(v) for s,v in groups.items()},tasks=len(tasks))),flush=True)


if __name__=='__main__':main()
