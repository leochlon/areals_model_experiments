import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FULL = Path('/workspace/runs/objaverse_target_comparison_20261030')
sys.path.insert(0, str(FULL / 'source'))
from prepare_corpus import SETTINGS, canonical_mesh, remesh, patch_arrays, patch, write


def one(index):
    row = json.loads((ROOT / 'protocol/cohort.json').read_text())['objects'][index]
    dest = ROOT / 'prepared/shards' / (row['id'] + '.npz')
    receipt = dest.with_suffix('.json')
    if receipt.exists():
        assert patch.sha256(dest) == json.loads(receipt.read_text())['sha256']
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Reuse the object's patches from a closed production pack when one exists.
    pack = FULL / 'prepared' / f"pack_{row['index'] // 128:04d}.h5"
    record = None
    if pack.with_suffix('.json').exists():
        doc = json.loads(pack.with_suffix('.json').read_text())
        record = next(r for r in doc['records'] if r['id'] == row['id'])
        assert record['patch']['status'] == 'ok'
        with h5py.File(pack, 'r') as f:
            group = f[record['patch']['group']]
            arrays = {k: group[k][:] for k in group}
            meta = json.loads(group.attrs['metadata'])
        method = 'reused_completed_production_pack'
    else:
        native = canonical_mesh(row['source_mesh'])
        fitted, stats = remesh(native)
        args = patch.parser().parse_args(['prepare', '--meshes', row['source_mesh'], '--out', str(dest)])
        for key in ['patches', 'max_faces', 'coverage_points', 'target_coverage']:
            setattr(args, key, SETTINGS[key])
        patches, meta = patch.build_mesh_patches(row['source_mesh'], args,
            np.random.default_rng(row['seed']), mesh=fitted, reference_mesh=native)
        meta.update(split=row['split'], group=row['family'], id=row['id'], remeshing=stats)
        arrays = patch_arrays(patches, meta)
        method = 'same_recipe_and_seed_rebuilt_without_hessian_while_production_pack_open'
    assert meta['sampled_coverage'] >= SETTINGS['minimum_coverage'], row['id']
    assert meta['split'] == row['split']
    metadata = dict(version=patch.VERSION, degree=3, meshes=[meta],
        scope='complete_geometry_local_charts', offset_is_global_distance=False,
        spatial_coordinate_units='mesh_diagonal_normalized_lengths_s1_s2_t')
    temporary = dest.with_suffix('.tmp.npz')
    np.savez(temporary, **arrays, metadata=np.array(json.dumps(metadata)))
    temporary.replace(dest)
    write(receipt, dict(id=row['id'], split=row['split'], method=method,
        path=dest.name, sha256=patch.sha256(dest), patches=len(arrays['inputs']), mesh=meta))
    print(json.dumps(dict(id=row['id'], split=row['split'], patches=len(arrays['inputs']),
        coverage=meta['sampled_coverage'], method=method)), flush=True)


def consolidate():
    selected = json.loads((ROOT / 'protocol/cohort.json').read_text())['objects']
    objects, shards, counts = [], [], Counter()
    dest = ROOT / 'prepared/pilot.h5'
    with h5py.File(str(dest) + '.tmp', 'w') as f:
        for row in selected:
            path = ROOT / 'prepared/shards' / (row['id'] + '.npz')
            receipt = json.loads(path.with_suffix('.json').read_text())
            assert patch.sha256(path) == receipt['sha256']
            meta = receipt['mesh']
            assert meta['split'] == row['split'] and meta['sampled_coverage'] >= .95
            group_name = 'patch/' + row['id']
            group = f.create_group(group_name)
            group.attrs['metadata'] = json.dumps(meta)
            with np.load(path, allow_pickle=False) as z:
                for key in z.files:
                    if key != 'metadata':
                        group.create_dataset(key, data=z[key], compression='lzf')
            objects.append(dict(row, pack=str(dest), patch=dict(status='ok', group=group_name, metadata=meta),
                query=dict(status='not_requested')))
            shards.append(dict(path=str(path.relative_to(ROOT / 'prepared')), sha256=receipt['sha256'],
                patches=receipt['patches'], mesh=meta))
            counts[row['split']] += receipt['patches']
    Path(str(dest) + '.tmp').replace(dest)
    write(ROOT / 'prepared/index.json', dict(complete=True, objects=objects, pack_sha256=patch.sha256(dest)))
    write(ROOT / 'prepared/geometry.json', dict(version=patch.VERSION, degree=3, shards=shards))
    hashes = sorted({s['mesh']['sha256'] for s in shards if s['mesh']['split'] == 'train'})
    assert len(hashes) == 385
    write(ROOT / 'prepared/receipt.json', dict(objects=dict(Counter(r['split'] for r in selected)),
        patches=dict(counts), geometry_sha256=patch.sha256(ROOT / 'prepared/geometry.json'),
        training_source_meshes_sha256=hashlib.sha256(json.dumps(hashes, separators=(',', ':')).encode()).hexdigest()))
    print(json.dumps(dict(patches=dict(counts), objects=len(objects))), flush=True)


def windsor():
    bench = Path('/workspace/runs/areals_all_simulation_efficiency_20261028/datasets/windsor')
    doc = json.loads((FULL / 'windsor/index.json').read_text())
    out = ROOT / 'windsor'
    for name in ['geometry', 'fields']: (out / name).mkdir(parents=True, exist_ok=True)
    cases, audits = [], []
    for row in doc['records']:
        name = row['id']
        geometry = out / 'geometry' / (name + '.npz')
        with h5py.File(row['pack'], 'r') as f:
            group = f[row['group']]
            arrays = {k: group[k][:] for k in ['inputs', 'origin', 'frame', 'radius']}
        metadata = dict(version=patch.VERSION, degree=3, meshes=[row['metadata']])
        np.savez(geometry, **arrays, mesh_index=np.zeros(len(arrays['inputs']), np.int32),
            metadata=np.array(json.dumps(metadata)))
        source = bench / 'data/fields' / (name + '.npz')
        with np.load(source, allow_pickle=False) as z:
            xyz, cp, weights = z['points'].copy(), z['values'].reshape(-1).copy(), z['weights'].reshape(-1).copy()
        assert np.all(weights > 0) and np.isfinite(cp).all()
        field = out / 'fields' / source.name
        np.savez(field, xyz=xyz, cp=cp, weights=weights,
            metadata=np.array(json.dumps(dict(weight_measure='area', source=str(source),
                provenance='Unchanged native area weights and pressure values from frozen WindsorML paper benchmark',
                geometry_axes='x streamwise, y spanwise, z vertical; source STL axes [0,2,1] already applied'))))
        cases.append(dict(id=name, car_id=name, split=row['split'], geometry=str(geometry),
            field=str(field), conditions=[]))
        audits.append(dict(id=name, queries=len(cp)))
    assert Counter(c['split'] for c in cases) == dict(train=274, val=35, test=35)
    write(out / 'cases.json', dict(version='patch-pressure-data-v1', evaluation_scope='held_out_cars',
        condition_names=[], cases=cases))
    write(out / 'conversion.json', dict(cases=audits, counts=dict(Counter(c['split'] for c in cases))))
    print('Converted all 344 Windsor cases without changing pressure samples or weights', flush=True)


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['one', 'consolidate', 'windsor'])
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.command == 'one': one(args.index)
    elif args.command == 'consolidate': consolidate()
    else: windsor()
