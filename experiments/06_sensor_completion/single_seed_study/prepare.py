from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np

ROOT = Path('/workspace/runs/areals_multicondition_wings_20261023')
SOURCE = Path('/workspace/runs/cfd-joint-gno-20261010/data')
VERSION = 'c898ed1aab52c704020a07e27338fac7f7c6874a5d9d4446cd7ee781c736e546'
sys.path.insert(0, '/workspace/runs/cfd-joint-gno-20261010/adapters')
from prepare_automotive_delta import s3_client


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def key_arrays(points, normals):
    values = np.ascontiguousarray(np.concatenate([points, normals], axis=1), dtype=np.float32)
    values[values == 0] = 0
    return values.view(np.dtype((np.void, values.dtype.itemsize * 6))).reshape(-1)


def prepare_family(family, cases, split, client):
    receipt_path = ROOT / 'receipts' / f'{family}.json'
    if receipt_path.exists():
        return json.loads(receipt_path.read_text())
    rows = []
    for identity, info in sorted(cases, key=lambda c: c[1]['physical']['aoa_degrees']):
        key = f'datasets/DoubleDelta/surface/{VERSION}/DoubleDelta__{info["source_id"]}/fields.h5'
        path = ROOT / 'source_fields' / f'{identity}.h5'
        if not path.exists():
            client.download_file('training-data-04a', key, str(path))
        with h5py.File(path) as data:
            p = np.asarray(data['mesh.verts'], np.float32)
            n = np.asarray(data['mesh.verts_normals'], np.float32)
            w = np.asarray(data['mass'], np.float64).reshape(-1)
            y = np.asarray(data['mesh.PressureCoeff'], np.float64).reshape(-1)
        assert p.ndim == 2 and p.shape[1] == 3 and p.shape == n.shape
        assert len(p) == len(w) == len(y)
        assert all(np.isfinite(a).all() for a in [p, n, w, y]) and np.all(w > 0)
        assert np.max(np.abs(np.linalg.norm(n, axis=1) - 1)) < 1e-4
        keys = key_arrays(p, n)
        unique, index, inverse, count = np.unique(keys, return_index=True, return_inverse=True, return_counts=True)
        if np.any(count > 1):
            for i in np.flatnonzero(count > 1):
                assert np.ptp(y[inverse == i]) < 1e-4, 'Conflicting pressure at identical position and normal'
            w = np.bincount(inverse, weights=w)
            p, n, y, keys = p[index], n[index], y[index], unique
        rows.append(dict(identity=identity, info=info, points=p, normals=n, weights=w, values=y,
                         keys=keys, source_key=key))
    angles = [r['info']['physical']['aoa_degrees'] for r in rows]
    assert angles == list(range(11, 20)), angles
    reference = rows[4]
    common = reference['keys']
    for row in rows:
        common = np.intersect1d(common, row['keys'], assume_unique=True)
    assert len(common) >= 8192, f'Insufficient exact common geometry: {len(common)}'
    _, base_index, _ = np.intersect1d(reference['keys'], common, return_indices=True)
    coverage = float(reference['weights'][base_index].sum() / reference['weights'].sum())
    assert coverage >= .90, f'Common physical-area coverage only {coverage}'
    rng = np.random.default_rng(int(hashlib.sha256(family.encode()).hexdigest()[:16], 16))
    if len(common) > 32000:
        common = common[np.sort(rng.choice(len(common), 32000, replace=False))]
    _, base_index, _ = np.intersect1d(reference['keys'], common, return_indices=True)
    query_points = reference['points'][base_index]
    query_normals = reference['normals'][base_index]
    query_weights = reference['weights'][base_index]
    geometry_path = ROOT / 'geometry_families' / f'{family}.npz'
    np.savez_compressed(geometry_path, points=reference['points'], normals=reference['normals'],
                        weights=reference['weights'].astype(np.float32))
    canonical = np.concatenate([reference['points'], reference['normals']], axis=1)
    canonical = canonical[np.lexsort(canonical.T[::-1])]
    signature = hashlib.sha256(canonical.tobytes()).hexdigest()
    records, conditions, pressures = [], {}, []
    for row in rows:
        _, at, _ = np.intersect1d(row['keys'], common, return_indices=True)
        assert np.array_equal(row['points'][at], query_points)
        assert np.array_equal(row['normals'][at], query_normals)
        identity = 'dd_' + hashlib.sha256(row['identity'].encode()).hexdigest()[:20]
        path = ROOT / 'geometry' / f'{identity}.npz'
        if not path.exists():
            os.link(geometry_path, path)
        assert os.stat(path).st_ino == os.stat(geometry_path).st_ino
        y = row['values'][at]
        pressures.append(y)
        np.savez_compressed(ROOT / 'fields' / f'{identity}.npz', points=query_points, normals=query_normals,
                            weights=query_weights.astype(np.float32), values=y.astype(np.float32))
        records.append(dict(id=identity, family='dd_' + family, split=split, path=f'geometry/{identity}.npz'))
        conditions[identity] = dict(aoa=float(row['info']['physical']['aoa_degrees']),
                                    source_id=row['identity'], family='dd_' + family,
                                    physical=row['info']['physical'])
    pressures = np.stack(pressures)
    receipt = dict(family=family, split=split, records=records, conditions=conditions,
                   geometry_signature=signature,
                   reference_angle=15, original_points=len(reference['points']),
                   common_points_before_sampling=len(base_index) if len(common) >= 32000 else len(common),
                   field_points=len(query_points), common_area_coverage=coverage,
                   angle_variation_rms=float(np.sqrt(np.average(np.var(pressures, axis=0), weights=query_weights))),
                   source_objects=[dict(identity=r['identity'], key=r['source_key']) for r in rows])
    write(receipt_path, receipt)
    return receipt


def main():
    start = time.perf_counter()
    for name in ['source_fields', 'geometry_families', 'geometry', 'fields', 'receipts']:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    source = json.loads((SOURCE / 'conditions-training.json').read_text())['cases']
    families = defaultdict(list)
    for identity, row in source.items():
        if identity.startswith('double_delta__'):
            families[row['geom_id']].append((identity, row))
    assert len(families) == 272 and all(len(c) == 9 for c in families.values())
    assert len({r['physical']['mach'] for c in families.values() for _, r in c}) == 1
    assert len({r['physical']['reynolds'] for c in families.values() for _, r in c}) == 1
    original_train = sorted(g for g, c in families.items() if c[0][1]['split'] == 'train')
    original_test = sorted(set(families) - set(original_train))
    assert len(original_train) == 256 and len(original_test) == 16
    chosen = set(np.random.default_rng(20261023).choice(original_train, 16, replace=False))
    split = {g: ('test' if g in original_test else 'val' if g in chosen else 'train') for g in families}
    write(ROOT / 'preparation_protocol.json', dict(
        dataset='DoubleDelta', seed=20261023, families=272, conditions_per_family=9,
        split_families=dict(Counter(split.values())), splits=split, reference_angle=15,
        minimum_common_points=8192, minimum_common_area_fraction=.90, maximum_field_points=32000))
    client = s3_client()
    results, rejected = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        tasks = {pool.submit(prepare_family, g, c, split[g], client): g for g, c in families.items()}
        for task in as_completed(tasks):
            family = tasks[task]
            try:
                result = task.result()
                results.append(result)
                print(json.dumps(dict(done=len(results), family=family, split=split[family],
                                      field_points=result['field_points'], coverage=result['common_area_coverage'],
                                      seconds=round(time.perf_counter() - start))), flush=True)
            except Exception as error:
                rejected.append(dict(family=family, error=repr(error), split=split[family]))
                print(json.dumps(dict(rejected=family,error=repr(error))), flush=True)
    write(ROOT / 'rejected_families.json', rejected)
    assert not rejected, 'some families failed; see rejected_families.json'
    signatures = defaultdict(set)
    for result in results:
        signatures[result['geometry_signature']].add(result['split'])
    assert all(len(s) == 1 for s in signatures.values()), 'Duplicate physical geometry crosses splits'
    records = sorted([r for result in results for r in result['records']], key=lambda r: r['id'])
    schema = json.loads(Path('/workspace/runs/areal_drivaerml_420_epoch14_20260927T162928Z/manifest.json').read_text())['schema']
    write(ROOT / 'manifest.json', dict(schema=schema, units='metres', records=records))
    conditions = {k:v for result in results for k,v in result['conditions'].items()}
    train_angles = [conditions[r['id']]['aoa'] for r in records if r['split'] == 'train']
    write(ROOT / 'conditions_reference_only.json', dict(
        mean=float(np.mean(train_angles)), std=float(np.std(train_angles)), cases=conditions))
    write(ROOT / 'preparation_complete.json', dict(
        status='PASS', families=len(results), records=len(records), split_cases=dict(Counter(r['split'] for r in records)),
        minimum_common_area_coverage=min(r['common_area_coverage'] for r in results),
        minimum_angle_variation_rms=min(r['angle_variation_rms'] for r in results),
        seconds=round(time.perf_counter() - start)))
    print('preparation complete', flush=True)


if __name__ == '__main__':
    main()
