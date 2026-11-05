import csv
import json
from pathlib import Path

import numpy as np
import field_benchmark as fb

ROOT=Path('/workspace/runs/areals_mask_pressure_transfer_20261025')
summary={}
for dataset in ['doubledelta','drivaerml']:
    d=ROOT/'datasets'/dataset
    for seed in range(3):
        updates=[json.loads((d/'models'/f'{arm}_seed{seed}'/'history.json').read_text())[-1]['update'] for arm in ['masked','unmasked']]
        assert updates[0]==updates[1]
    bench=d/'bench'
    fb.main(['score','--bench',str(bench),'--split','test','--final','--reference','masked_seed0'])
    config=fb.bench_config(bench)
    records={r['id']:r for r in fb.records_of(config,'test')}
    with (bench/'scores/test_per_problem.csv').open() as f:rows=list(csv.DictReader(f))
    groups={}
    for row in rows:
        arm,seed=row['method'].rsplit('_seed',1)
        key=(arm,int(seed),row['setting'],records[row['car']]['family'])
        groups.setdefault(key,[]).append(float(row['nrmse']))
    families=sorted({k[3] for k in groups})
    results={}
    for setting in sorted({k[2] for k in groups}):
        matrices={arm:np.asarray([[np.mean(groups[arm,seed,setting,family]) for family in families] for seed in range(3)])
                  for arm in ['masked','unmasked']}
        delta=matrices['masked']-matrices['unmasked']
        rng=np.random.default_rng(57300)
        samples=[]
        for _ in range(5000):
            si=rng.integers(0,3,3);gi=rng.integers(0,len(families),len(families))
            samples.append(float(delta[np.ix_(si,gi)].mean()))
        interval=np.quantile(samples,[.0125,.9875] if setting=='sensors-20' else [.025,.975]).tolist()
        results[setting]=dict(masked_nrmse=float(matrices['masked'].mean()),unmasked_nrmse=float(matrices['unmasked'].mean()),
            masked_minus_unmasked=float(delta.mean()),joint_seed_and_geometry_interval=interval,
            interval_level=.975 if setting=='sensors-20' else .95,
            per_seed={arm:matrices[arm].mean(1).tolist() for arm in matrices},
            verdict='masked_better' if interval[1]<0 else 'unmasked_better' if interval[0]>0 else 'unresolved')
    summary[dataset]=dict(test_geometries=len(families),seeds=[0,1,2],settings=results)
(ROOT/'results.json').write_text(json.dumps(dict(passed=True,datasets=summary,primary_setting='sensors-20'),indent=2)+'\n')
print(json.dumps(summary,indent=2))
