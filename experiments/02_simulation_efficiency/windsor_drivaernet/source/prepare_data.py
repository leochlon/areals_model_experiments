import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re

import h5py
import numpy as np

from storage import RangeFile, client
from util import ROOT, stable, write

INVENTORY = Path('/workspace/runs/areals_dataset_inventory_20261025')
SOURCES = {'hilift': 'HiLiftAeroML', 'windsor': 'WindsorML', 'ahmed': 'AhmedML', 'emmiwing': 'EmmiWing'}


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(points):
    scale = max(float(np.ptp(points, axis=0).max()), 1e-12)
    q = np.rint(points / (scale * 1e-7)).astype(np.int64)
    order = np.lexsort((q[:, 2], q[:, 1], q[:, 0]))
    return hashlib.sha256(q[order].tobytes()).hexdigest()


def plan(dataset):
    out = ROOT / 'datasets' / dataset
    assert not (out / 'data_plan.json').exists()
    source = read(INVENTORY / (SOURCES[dataset] + '_objects.json'))
    official, designs = {}, {}
    if dataset != 'emmiwing':
        key = {'hilift': 'hiliftaeroml', 'windsor': 'windsorml', 'ahmed': 'ahmedml'}[dataset]
        manifest = read(ROOT / 'public_sources' / key / 'splits_manifest.json')
        prefix = 'geometry' if dataset == 'hilift' else 'full'
        official = {identity: split for split in ('train', 'val', 'test') for identity in manifest[prefix + '_' + split]}
    else:
        metadata = Path('/workspace/runs/cfd-native-data-20261014/sources/EmmiWing/collated/metadata.npz')
        with np.load(metadata, allow_pickle=False) as a:
            assert a['cond_cols'][:5].tolist() == ['chord_root', 'span', 'taper_ratio', 'sweep', 'dihedral']
            for identity, parameters in zip(a['stems'], a['cond'][:, :5]):
                designs[str(identity)] = hashlib.sha256(np.round(parameters.astype(np.float64), 6).tobytes()).hexdigest()[:24]
    entries = []
    for e in source['fields']:
        identity = e['key'].split('/')[-2]
        native = identity.split('__', 1)[-1]
        family = native.split('_AoA_')[0] if dataset == 'hilift' else designs[native] if dataset == 'emmiwing' else native
        if official:
            assert native in official, native
            split = official[native]
        else:
            fold = stable('ase-family-split-v1|' + family) % 100
            split = 'train' if fold < 70 else 'val' if fold < 85 else 'test'
        entries.append(dict(id=identity, native_id=native, family=family, split=split,
                            bucket=source['result']['bucket'], key=e['key'], bytes=e['bytes'], dataset=dataset))
    groups = defaultdict(list)
    for e in entries:
        groups[e['family']].append(e)
    for family, members in groups.items():
        assert len({e['split'] for e in members}) == 1
        reference = min(members, key=lambda e: e['id'])['id']
        for e in members:
            e['geometry_reference'] = reference
    entries.sort(key=lambda e: e['id'])
    absent = sorted(set(official) - {e['native_id'] for e in entries})
    result = dict(dataset=dataset, entries=entries, missing_public_cases=absent,
        counts=dict(Counter(e['split'] for e in entries)),
        family_counts={s: len({e['family'] for e in entries if e['split'] == s}) for s in ('train', 'val', 'test')},
        grouping='Published HiLift geometry ID' if dataset == 'hilift' else 'Five published geometry parameters, rounded to six decimals' if dataset == 'emmiwing' else 'Official case IDs, followed by exact native geometry duplicate check',
        split='Official geometry split intersected with existing cases' if dataset == 'hilift' else 'Official full split intersected with existing cases' if official else 'Geometry-group hash split, 70/15/15',
        field_grid='32,000 uniform native rows, with native positive area weights; no target clipping or outlier filtering',
        geometry_input='One fixed geometry sample per physical family, shared across all flow conditions',
        geometry_rows='Finite coordinates and normals; normal length > 1e-8; finite positive area',
        bad_labels='A nonfinite pressure on valid geometry fails the case; no silent case exclusion')
    write(out / 'data_plan.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'entries'}), flush=True)


def export(entry):
    os.umask(0o077)
    dataset, identity = entry['dataset'], entry['id']
    out = ROOT / 'datasets' / dataset
    receipt = out / 'exports' / (identity + '.json')
    if receipt.exists():
        old = read(receipt)
        if old['passed']:
            return old
    try:
        remote = RangeFile(client(), entry['bucket'], entry['key'], entry['bytes'])
        with h5py.File(remote, 'r') as h:
            points = np.asarray(h['mesh.verts'], dtype=np.float32)
            normals = np.asarray(h['mesh.verts_normals'], dtype=np.float32)
            weights = np.asarray(h['mass'], dtype=np.float32).reshape(-1)
            values = np.asarray(h['mesh.PressureCoeff'], dtype=np.float32).reshape(-1)
        n = len(points)
        assert points.shape == normals.shape == (n, 3) and len(values) == len(weights) == n
        length = np.linalg.norm(normals, axis=1)
        good = np.isfinite(points).all(1) & np.isfinite(normals).all(1) & np.isfinite(weights) & (weights > 0) & (length > 1e-8)
        assert good.sum() >= 1000
        assert np.isfinite(values[good]).all(), 'Nonfinite pressure on valid geometry'
        points, normals, weights, values = points[good], normals[good], weights[good], values[good]
        normals = normals / np.linalg.norm(normals, axis=1)[:, None]
        rng = np.random.default_rng(stable('ase-field-grid-v1|' + identity))
        selected = np.sort(rng.choice(len(points), min(32000, len(points)), replace=False))
        path = out / 'data/fields' / (identity + '.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, points=points[selected], normals=normals[selected],
                            weights=weights[selected], values=values[selected])
        geometry_hash = None
        if identity == entry['geometry_reference']:
            geometry_hash = fingerprint(points)
            p = weights.astype(np.float64); p /= p.sum()
            selected_geometry = rng.choice(len(points), 16384, replace=True, p=p)
            path = out / 'data/geometry' / (identity + '.npz')
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, points=points[selected_geometry], normals=normals[selected_geometry],
                                weights=np.full(16384, float(weights.sum(dtype=np.float64)) / 16384, np.float32))
        result = dict(**entry, passed=True, native_rows=n, retained_rows=len(points),
            invalid_geometry_rows=int(n-good.sum()), field_rows=len(selected),
            full_native_geometry_hash=geometry_hash, object_etag=remote.etag, bytes_read=remote.transferred,
            cp_min=float(values.min()), cp_max=float(values.max()), area=float(weights.sum(dtype=np.float64)))
    except Exception as error:
        result = dict(**entry, passed=False, error=type(error).__name__ + ': ' + str(error)[:500])
    write(receipt, result)
    return result


def finish(dataset):
    out = ROOT / 'datasets' / dataset
    planned = read(out / 'data_plan.json')
    reports = [read(out / 'exports' / (e['id'] + '.json')) for e in planned['entries']]
    failures = [e for e in reports if not e['passed']]
    write(out / 'data_failures.json', failures)
    assert not failures, f'{len(failures)} failed cases; no cohort reduction is automatic'
    duplicates = []
    if dataset == 'windsor':
        amendment = read(out / 'cohort_amendment.json')
        assert amendment['rule'] == 'Keep one case per native geometry; preserve test, then validation, then training; smallest numerical run ID breaks ties.'
        by_geometry = defaultdict(list)
        for e in reports:
            by_geometry[e['full_native_geometry_hash']].append(e)
        retained = []
        for geometry_hash, members in by_geometry.items():
            chosen = min(members, key=lambda e: (-{'train': 0, 'val': 1, 'test': 2}[e['split']],
                                                 int(e['native_id'].split('_')[-1])))
            retained.append(chosen)
            if len(members) > 1:
                duplicates.append(dict(geometry_hash=geometry_hash, retained=chosen['id'], split=chosen['split'],
                    omitted=[e['id'] for e in members if e is not chosen]))
        reports = retained
        write(out / 'geometry_duplicates.json', duplicates)
    geometry_folds = defaultdict(set)
    for e in reports:
        if e['full_native_geometry_hash']:
            geometry_folds[e['full_native_geometry_hash']].add(e['split'])
    assert all(len(s) == 1 for s in geometry_folds.values()), 'Identical native geometry crosses splits'
    records = []
    for e in reports:
        reference = out / 'data/geometry' / (e['geometry_reference'] + '.npz')
        path = out / 'data/geometry' / (e['id'] + '.npz')
        if not path.exists():
            os.link(reference, path)
        records.append(dict(id=e['id'], family=e['family'], geometry_id=e['family'], split=e['split'],
                            path='geometry/' + e['id'] + '.npz'))
    write(out / 'data/manifest.json', dict(schema='geometry_latents_v1', units='metres', records=records))
    counts = dict(Counter(e['split'] for e in reports))
    families = {s: len({e['family'] for e in reports if e['split'] == s}) for s in counts}
    write(out / 'data_ready.json', dict(passed=True, counts=counts, families=families,
        native_rows_read=sum(e['native_rows'] for e in reports), invalid_geometry_rows=sum(e['invalid_geometry_rows'] for e in reports),
        missing_public_cases=planned['missing_public_cases'], pressure_outliers_retained=True,
        geometry_duplicate_check=True, duplicate_groups=duplicates, prepared_cases=len(records)))
    print('DATA_READY', dataset, counts, flush=True)


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['plan', 'export', 'finish'])
    p.add_argument('--dataset', choices=list(SOURCES), required=True)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    if a.command == 'plan':
        plan(a.dataset)
    elif a.command == 'finish':
        finish(a.dataset)
    else:
        entries = read(ROOT / 'datasets' / a.dataset / 'data_plan.json')['entries'][a.rank::a.shards]
        with ProcessPoolExecutor(a.workers) as pool:
            for i, result in enumerate(pool.map(export, entries, chunksize=1)):
                if not result['passed']:
                    print('EXPORT_FAILED', result['id'], result['error'], flush=True)
                if (i + 1) % 10 == 0:
                    print('EXPORTED', a.dataset, a.rank, i + 1, '/', len(entries), flush=True)


if __name__ == '__main__':
    main()
