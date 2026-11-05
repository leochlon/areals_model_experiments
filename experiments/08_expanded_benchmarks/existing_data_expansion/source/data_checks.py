import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import field_benchmark as fb
from areal_tokens.data import read_manifest, load_cloud_file
from util import ROOT, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()
    root = ROOT / 'datasets' / args.dataset
    records, _ = read_manifest(root / 'manifest.json')
    families, geometry, geometry_splits = {}, {}, {}
    for record in records:
        families.setdefault(record['family'], set()).add(record['split'])
        stat = Path(record['path']).stat()
        identity = record.get('geometry_id', record['family'])
        geometry.setdefault(identity, {})[(stat.st_dev, stat.st_ino)] = record['path']
        geometry_splits.setdefault(identity, set()).add(record['split'])
        field = fb.load_field(root / 'fields', record['id'])
        assert len(field.points) > 500
        assert np.isfinite(field.values).all() and np.isfinite(field.points).all()
        assert np.isfinite(field.weights).all() and (field.weights > 0).all()
    assert all(len(v) == 1 for v in families.values()), 'Physical geometry crosses splits'
    assert all(len(v) == 1 for v in geometry.values()), 'Condition-specific geometry input'
    hashes = {}
    for identity, paths in geometry.items():
        cloud, _ = load_cloud_file(next(iter(paths.values())))
        h = hashlib.sha256()
        for array in [cloud.points, cloud.normals, cloud.weights]:
            h.update(np.ascontiguousarray(array).tobytes())
        hashes.setdefault(h.hexdigest(), set()).update(geometry_splits[identity])
    assert all(len(splits) == 1 for splits in hashes.values()), 'Identical prepared geometry crosses splits'
    cp = root / 'conditions.json'
    stats = None
    if cp.exists():
        condition = json.loads(cp.read_text())
        values = np.array([condition['cases'][r['id']] for r in records if r['split'] == 'train'])
        assert np.isfinite(values).all()
        stats = dict(names=condition['names'], mean=values.mean(0).tolist(),
                     std=np.maximum(values.std(0), 1e-8).tolist(), fitted_on='training records only')
        write(root / 'condition_statistics.json', stats)
        coverage = {}
        for split in ['train', 'val', 'test']:
            selected = np.asarray([condition['cases'][r['id']] for r in records if r['split'] == split])
            coverage[split] = {name: dict(min=float(selected[:, i].min()), max=float(selected[:, i].max()),
                                         unique=int(len(np.unique(selected[:, i]))))
                               for i, name in enumerate(condition['names'])}
        write(root / 'condition_coverage.json', coverage)
        if args.dataset in ['blendednet']:
            inventory = json.loads((root / 'inventory.json').read_text())
            low, high = inventory['bands']['train_max_mach'], inventory['bands']['val_max_mach']
            assert coverage['train']['mach']['max'] <= low + 1e-6
            assert coverage['val']['mach']['min'] > low - 1e-6
            assert coverage['val']['mach']['max'] <= high + 1e-6
            assert coverage['test']['mach']['min'] >= high - 1e-6
            assert coverage['train']['mach']['max'] < coverage['test']['mach']['min']
    if args.dataset == 'doubledelta':
        angles = {s: sorted({condition['cases'][r['id']][0] for r in records if r['split'] == s})
                  for s in ['train', 'val', 'test']}
        assert angles == dict(train=[11, 12, 13, 14, 15, 16], val=[17], test=[18, 19]), angles
    counts = {s: sum(r['split'] == s for r in records) for s in ['train', 'val', 'test']}
    assert min(counts.values()) > 0
    write(root / 'data_checks.json', dict(passed=True, counts=counts, physical_families=len(families),
        condition_statistics=stats))
    print('data checks passed', args.dataset, counts, flush=True)


if __name__ == '__main__':
    main()
