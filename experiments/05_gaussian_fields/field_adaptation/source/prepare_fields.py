import json
import os
from pathlib import Path
import shutil

import numpy as np

import field_benchmark as fb
from surface_fields import SurfaceGRF
from study import ROOT, write

OLD = Path(os.environ['TASK_ROOT']) / 'grf'


def main():
    assert not (ROOT / 'field_data_ready.json').exists()
    original = json.loads((OLD / 'manifest.json').read_text())
    records = []
    for record in original['records']:
        if record['split'] == 'test':
            continue
        source = OLD / 'fields' / (record['id'] + '.npz')
        target = ROOT / 'fields' / source.name
        if not target.exists():
            os.link(source, target)
        records.append({**record, 'path': str((OLD / record['path']).resolve())})
    representatives = {}
    for record in original['records']:
        if record['split'] == 'test':
            representatives.setdefault(record['family'], record)
    assert len(representatives) == 16
    rng = np.random.default_rng(91317)
    generated = []
    for family, record in sorted(representatives.items()):
        with np.load(OLD / 'bases' / (family + '.npz')) as b:
            grf = SurfaceGRF(b['lam'], b['phi'].astype(np.float64), b['mass'],
                             ells=tuple(b['ells']), nu=float(b['nu']), sigma=.001, scale=float(b['scale']))
        with np.load(OLD / 'fields' / (record['id'] + '.npz')) as packed:
            geometry = {k: packed[k] for k in ['points', 'weights', 'normals']}
        values, scales = grf.sample(rng, 8)
        for draw in range(8):
            identity = f'{family}__prodconfirm{draw}'
            target = ROOT / 'clouds' / (identity + '.npz')
            if not target.exists():
                os.link((OLD / record['path']).resolve(), target)
            np.savez(ROOT / 'fields' / (identity + '.npz'), **geometry, values=values[draw].astype(np.float32))
            records.append(dict(id=identity, family=family, split='test', path=str(target)))
            generated.append(dict(id=identity, family=family, length_scale_index=int(scales[draw])))
    write(ROOT / 'field_manifest.json', {**original, 'records': records})
    config = json.loads((OLD / 'bench/config.json').read_text())
    config.update(manifest=str(ROOT / 'field_manifest.json'), fields=str(ROOT / 'fields'))
    write(ROOT / 'bench/config.json', config)
    (ROOT / 'bench/observations').mkdir(exist_ok=True)
    for suffix in ['json', 'npz']:
        shutil.copyfile(OLD / f'bench/observations/val.{suffix}', ROOT / f'bench/observations/val.{suffix}')
    fb.main(['observations', '--bench', str(ROOT / 'bench'), '--manifest', str(ROOT / 'field_manifest.json'),
             '--fields', str(ROOT / 'fields'), '--split', 'test', '--patch-counts', '32', '--hidden', '.6',
             '--sensors', '5', '20', '100', '500', '--seed', '0', '--repeats', '1'])
    if not (ROOT / 'bases').exists():
        (ROOT / 'bases').symlink_to(OLD / 'bases', target_is_directory=True)
    write(ROOT / 'field_data_ready.json', dict(passed=True, test_seed=91317,
        counts={split: sum(r['split'] == split for r in records) for split in ['train', 'val', 'test']},
        fresh_test_fields=generated, reference_sigma=.001))
    print('FIELD_DATA_READY', len(generated), flush=True)


if __name__ == '__main__':
    main()
