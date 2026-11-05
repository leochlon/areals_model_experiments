import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from functools import lru_cache
import hashlib
import math
import multiprocessing
import shutil

from common import *


@lru_cache(maxsize=1)
def experiment():
    torch.set_num_threads(1)
    return fb.load_model(BASE, 'cpu')[1]


def cache_one(task):
    name, r, fields = task
    f = fb.load_field(fields, r['id'])
    p = fb.prepare_car(r, f.points, experiment())
    out = ROOT / 'cache' / name / (r['cache_key'] + '.npz')
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{'a_' + k: v for k, v in p.arrays.items() if k in fb.CLOUD_KEYS},
             sample_to_field=p.sample_to_field, field_to_sample=p.field_to_sample, anchor_index=p.anchor_index)
    return str(out.relative_to(ROOT))


def prepare(name):
    out = ROOT / 'data' / name
    assert not (out / 'ready.json').exists()
    bench = original_bench(name)
    cfg = fb.bench_config(bench)
    training_bench = HOME_RUNS / 'areals_architecture_controls_20261024/replication/bench'
    rows = fb.records_of(cfg)
    by_id = {r['id']: r for r in rows}
    families = {}
    tasks, unique = [], {}
    for r in rows:
        r['path'] = str(Path(r['path']).resolve())
        families.setdefault(r['family'], set()).add(r['split'])
        geometry = r.get('grid_id', r.get('geometry_id', r['family']))
        fields = cfg['fields']
        if name == 'reserved_cars' and r['split'] != 'test':
            fields = fb.bench_config(training_bench)['fields']
        r['field_path'] = str(Path(fields) / (r['id'] + '.npz'))
        with np.load(r['field_path']) as a:
            point_hash = hashlib.sha256(a['points'].astype(np.float64).tobytes()).hexdigest()
            assert np.isfinite(a['values']).all()
            assert np.isfinite(a['weights']).all() and (a['weights'] > 0).all()
        r['cache_key'] = hashlib.sha256((geometry + point_hash).encode()).hexdigest()[:24]
        if r['cache_key'] not in unique:
            unique[r['cache_key']] = r
            tasks.append((name, r, fields))
    assert all(len(v) == 1 for v in families.values())
    if name == 'nine_angle':
        ref = read(NINE / 'conditions_reference_only.json')
        cond = dict(names=['aoa_degrees'], cases={k:[v['aoa']] for k,v in ref['cases'].items()},
                    mean=[ref['mean']], std=[ref['std']])
        mixture = dict(unobserved=.1, sensors=.45)
    elif name == 'reserved_cars':
        cond = dict(names=['constant_flow'], cases={r['id']:[0.] for r in rows}, mean=[0.], std=[1.])
        mixture = dict(unobserved=.1, sensors=.45)
    else:
        src = EXPANDED / 'datasets' / name
        cond = read(src / 'conditions.json')
        cond.update({k:v for k,v in read(src / 'condition_statistics.json').items() if k in ['mean','std']})
        mixture = dict(unobserved=.1, sensors=.9)
    train_c = np.asarray([cond['cases'][r['id']] for r in rows if r['split'] == 'train'])
    cond['training_bounds'] = [train_c.min(0).tolist(), train_c.max(0).tolist()]
    write(out / 'conditions.json', cond)
    write(out / 'manifest.json', dict(records=rows))
    write(out / 'dataset.json', dict(name=name, original_bench=str(bench), fields=cfg['fields'],
          original_manifest=cfg['manifest'], counts=dict(Counter(r['split'] for r in rows)),
          protocol=cfg['protocol'], mix=mixture, original_updates=40 * math.ceil(sum(r['split']=='train' for r in rows)/8),
          constant_condition=bool(np.all(train_c.max(0) == train_c.min(0)))))
    for split in ['val','test']:
        target = out / 'observations'
        target.mkdir(parents=True, exist_ok=True)
        for ext in ['json','npz']:
            source_bench = training_bench if name == 'reserved_cars' and split == 'val' else bench
            src = source_bench / 'observations' / (split + '.' + ext)
            shutil.copy2(src, target / src.name)
    val = [r for r in rows if r['split'] == 'val']
    rng = np.random.default_rng(0)
    if name == 'nine_angle':
        fam = sorted({r['family'] for r in val})
        chosen = set(rng.choice(fam, min(4,len(fam)), replace=False))
        selected = [r['id'] for r in val if r['family'] in chosen]
    else:
        selected = [val[i]['id'] for i in sorted(rng.choice(len(val), min(64,len(val)), replace=False))]
    write(out / 'selection.json', dict(cases=selected, settings=[5,20,100,500]))
    with ProcessPoolExecutor(12, mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,_ in enumerate(pool.map(cache_one,tasks,chunksize=4),1):
            if i % 100 == 0:
                print('CACHE',name,i,len(tasks),flush=True)
    write(out / 'ready.json', dict(status='PASS',cases=len(rows),geometries=len(families),cached=len(tasks)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=DATASETS,required=True)
    prepare(parser.parse_args().dataset)
