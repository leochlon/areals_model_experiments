import argparse
import hashlib
import io
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

from areal_tokens.data import SCHEMA
from storage import RangeFile, client, objects
from util import ROOT, stable, write

PREFIXES = {
    'blendednet': ('training-data', 'BlendedNetPlusPlus', '27767449179d154b4a3cf48b5400b5f0c2dfc109a73a75194b6989e95fcfa9c2'),
    'drivaerml': ('training-data-04a', 'DrivAerML', 'c4d124eefa153d265acedfd5d88202d9b42ce47440ce9bf3ff81e4c0370c17de'),
    'drivaernet': ('training-data-04a', 'DrivAerNet', 'bf29502fbbd3878ad35bc89358fc5a54c0d57a4b173b37c807c1fdd42e84a616'),
}
LOCAL = threading.local()


def s3():
    if not hasattr(LOCAL, 's3'):
        LOCAL.s3 = client()
    return LOCAL.s3


def scalar_values(bucket, key):
    response = s3().get_object(Bucket=bucket, Key=key)
    try:
        packed = response['Body'].read()
    finally:
        response['Body'].close()
    result = {}
    with h5py.File(io.BytesIO(packed), 'r') as h:
        for name in ['mach_number', 'angle_of_attack', 'angle_of_sideslip', 'altitude', 'reynolds_number',
                     'convergence_flag', 'aircraft_half_span_m', 'reynolds_number_reported', 'freestream_velocity']:
            if name in h:
                array = np.asarray(h[name][()]).reshape(-1)
                if len(array) == 1:
                    value = array[0].item()
                    result[name] = value.decode() if isinstance(value, bytes) else value
    return result


def metadata(entry, dataset, out):
    path = out / 'metadata' / (entry['id'] + '.json')
    if path.exists():
        return json.loads(path.read_text())
    try:
        values = scalar_values(entry['bucket'], entry['key'].replace('/fields.h5', '/scalars.h5'))
        result = dict(entry, conditions=values, passed=True)
    except Exception as error:
        result = dict(entry, passed=False, error=type(error).__name__ + ': ' + str(error)[:300])
    write(path, result)
    return result


def family_name(identity, dataset):
    return identity


def shape_signature(points):
    # Order-independent native coordinates; source design identity additionally groups differently meshed conditions.
    extent = max(float(np.ptp(points, axis=0).max()), 1e-12)
    quantised = np.rint(points / (extent * 1e-7)).astype(np.int64)
    order = np.lexsort((quantised[:, 2], quantised[:, 1], quantised[:, 0]))
    return hashlib.sha256(quantised[order].tobytes()).hexdigest()


def export_one(entry, dataset, out):
    receipt = out / 'exports' / (entry['id'] + '.json')
    if receipt.exists():
        return json.loads(receipt.read_text())
    try:
        remote = RangeFile(s3(), entry['bucket'], entry['key'], entry['bytes'])
        with h5py.File(remote, 'r') as h:
            required = ['mesh.verts', 'mesh.verts_normals', 'mass', 'mesh.PressureCoeff']
            assert all(k in h for k in required), (entry['id'], list(h.keys()))
            points = np.asarray(h['mesh.verts'], dtype=np.float64)
            normals = np.asarray(h['mesh.verts_normals'], dtype=np.float64)
            weights = np.asarray(h['mass'], dtype=np.float64).reshape(-1)
            values = np.asarray(h['mesh.PressureCoeff'], dtype=np.float64).reshape(-1)
        assert points.ndim == 2 and points.shape[1] == 3 and normals.shape == points.shape
        assert len(values) == len(weights) == len(points)
        lengths = np.linalg.norm(normals, axis=1)
        valid = (np.isfinite(points).all(1) & np.isfinite(normals).all(1) & np.isfinite(weights)
                 & (weights > 0) & (lengths > 1e-8))
        # Field labels never choose geometry/sensor locations. A nonfinite label on otherwise valid geometry fails the case.
        assert np.isfinite(values[valid]).all(), 'Nonfinite pressure on valid surface'
        assert valid.sum() >= 1000
        points, normals, weights, values = points[valid], normals[valid], weights[valid], values[valid]
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        fingerprint = shape_signature(points)
        count = min(32000, len(points))
        identity = family_name(entry['id'], dataset)
        rng = np.random.default_rng(stable('field-grid:' + identity))
        chosen = np.sort(rng.choice(len(points), count, replace=False))
        grid = out / 'family_grids' / (hashlib.sha256(identity.encode()).hexdigest() + '.npz')
        maximum_alignment_gap = 0.
        field_points, field_normals, field_weights, field_values = points[chosen], normals[chosen], weights[chosen], values[chosen]
        field_path = out / 'uncanonical_fields' / (entry['id'] + '.npz')
        field_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(field_path, points=field_points.astype(np.float32),
                normals=field_normals.astype(np.float32), weights=field_weights.astype(np.float32),
                values=field_values.astype(np.float32))
        chosen_geometry = rng.choice(len(points), 16384, replace=True, p=weights / weights.sum())
        geometry_path = out / 'source_geometry' / (entry['id'] + '.npz')
        geometry_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(geometry_path, points=points[chosen_geometry].astype(np.float32),
                normals=normals[chosen_geometry].astype(np.float32),
                weights=np.full(16384, weights.sum() / 16384, dtype=np.float32))
        result = dict(entry, passed=True, source_points=len(valid), valid_points=len(points),
            removed_invalid_geometry=int((~valid).sum()), field_points=count, geometry_fingerprint=fingerprint,
            source_family=identity, source_field=str(field_path), source_geometry=str(geometry_path),
            data_read_bytes=remote.transferred, object_bytes=remote.size,
            extent=np.ptp(points, axis=0).tolist(), area=float(weights.sum()), maximum_alignment_gap=maximum_alignment_gap,
            field_name='mesh.PressureCoeff')
    except Exception as error:
        result = dict(entry, passed=False, error=type(error).__name__ + ': ' + str(error)[:300])
    write(receipt, result)
    return result


class Groups:
    def __init__(self):
        self.parent = {}

    def root(self, key):
        if key not in self.parent:
            self.parent[key] = key
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        self.parent[max(a, b)] = min(a, b)


def build_inventory(dataset, out, workers):
    path = out / 'inventory.json'
    if path.exists():
        return json.loads(path.read_text())
    bucket, name, version = PREFIXES[dataset]
    prefix = f'datasets/{name}/surface/{version}/'
    entries = [dict(id=o['Key'].split('/')[-2], bucket=bucket, key=o['Key'], bytes=o['Size'])
               for o in objects(s3(), bucket, prefix) if o['Key'].endswith('/fields.h5') and o['Size'] > 0]
    print('LIVE_INVENTORY', dataset, len(entries), flush=True)
    live_count = len(entries)
    with ThreadPoolExecutor(workers) as pool:
        reports = list(pool.map(lambda e: metadata(e, dataset, out), sorted(entries, key=lambda e: e['id'])))
    passed = [r for r in reports if r['passed']]
    failures = [r for r in reports if not r['passed']]
    write(out / 'metadata_failures.json', failures)
    if dataset == 'blendednet':
        assert len(entries) == 12480, len(entries)
        mach = np.asarray([r['conditions']['mach_number'] for r in passed])
        low, high = np.quantile(mach, [.70, .85]).tolist()
        bands = dict(train_max_mach=low, val_max_mach=high)  # from the condition metadata alone
    else:
        bands = None
    result = dict(dataset=dataset, bucket=bucket, prefix=prefix, live_cases=live_count, metadata_cases=len(entries), reports=reports, bands=bands)
    write(path, result)
    print('condition bands', dataset, bands, 'metadata_failures', len(failures), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=list(PREFIXES), required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--probe', action='store_true')
    p.add_argument('--all-condition-bands', action='store_true')
    a = p.parse_args()
    out = ROOT / 'datasets' / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    if a.all_condition_bands:
        assert a.dataset == 'blendednet'
        assert not (out / 'setup_checks.json').exists() and not (out / 'models').exists(), 'Cohort cannot change after model work'
        for name in ['manifest.json', 'ready.json', 'data_checks.json', 'split_protocol.json']:
            source = out / name
            if source.exists():
                previous = out / 'initial_hash_intersection' / name
                previous.parent.mkdir(exist_ok=True)
                if not previous.exists():
                    previous.write_bytes(source.read_bytes())
    else:
        assert not (out / 'ready.json').exists(), 'Already prepared'
    inventory = build_inventory(a.dataset, out, a.workers)
    entries = [r for r in inventory['reports'] if r['passed']]
    if a.probe:
        print(export_one(entries[0], a.dataset, out), flush=True)
        return
    reports = []
    source_groups = {}
    for entry in entries:
        source_groups.setdefault(family_name(entry['id'], a.dataset), []).append(entry)
    def export_family(members):
        return [export_one(e, a.dataset, out) for e in sorted(members, key=lambda e: e['id'])]
    with ThreadPoolExecutor(a.workers) as pool:
        for group_reports in pool.map(export_family, source_groups.values()):
            reports.extend(group_reports)
            write(out / 'progress.json', dict(stage='native array export', done=len(reports), total=len(entries),
                                             passed=sum(r['passed'] for r in reports)))
            if len(reports) % 50 < len(group_reports):
                print('EXPORT_PROGRESS', a.dataset, len(reports), len(entries), flush=True)
    valid = [r for r in reports if r['passed']]
    write(out / 'export_failures.json', [r for r in reports if not r['passed']])
    assert len(valid) >= .99 * len(entries), 'More than one percent native exports failed; inspect before fitting'
    groups = Groups()
    split_groups = Groups()
    for r in valid:
        groups.join('source:' + r['source_family'], 'geometry:' + r['geometry_fingerprint'])
        design = r['source_family']
        r['split_design'] = design
        split_groups.join('source:' + design, 'geometry:' + r['geometry_fingerprint'])
    grouped = {}
    alias_designs = {}
    for r in valid:
        root = split_groups.root('source:' + r['split_design'])
        alias_designs.setdefault(root, set()).add(r['split_design'])
    for r in valid:
        r['physical_family'] = hashlib.sha256(groups.root('source:' + r['source_family']).encode()).hexdigest()[:24]
        grouped.setdefault(r['physical_family'], []).append(r)
    official = json.loads((ROOT / 'source/drivaerml_official_splits.json').read_text()) if a.dataset == 'drivaerml' else None
    official_map = {identity: split for split in ['train', 'val', 'test']
                    for identity in official['full_' + split]} if official else {}
    records, conditions, omissions = [], {}, []
    names = ['mach', 'aoa_degrees', 'altitude'] if a.dataset == 'blendednet' else ['mach', 'aoa_degrees', 'reynolds', 'sideslip_degrees']
    if a.dataset.startswith('drivaer'):
        names = ['constant_flow']
    bands = inventory['bands']
    for family, members in sorted(grouped.items()):
        members.sort(key=lambda r: r['id'])
        root = split_groups.root('source:' + members[0]['split_design'])
        split_name = root
        split_family = hashlib.sha256(split_name.encode()).hexdigest()[:24]
        assert len({split_groups.root('source:' + r['split_design']) for r in members}) == 1
        fold = stable('split:' + split_family) % 100
        designated = ('train' if fold < 80 else 'val' if fold < 90 else 'test')
        if a.all_condition_bands:
            band_counts = {split: 0 for split in ['train', 'val', 'test']}
            for r in members:
                mach = float(r['conditions']['mach_number'])
                split = 'train' if mach <= bands['train_max_mach'] else 'val' if mach <= bands['val_max_mach'] else 'test'
                band_counts[split] += 1
            # One-condition geometries use every available case. Multi-condition duplicates stay together.
            designated = max(['train', 'val', 'test'], key=lambda s: (band_counts[s], ['train', 'val', 'test'].index(s)))
        if official:
            splits = set()
            for r in members:
                match = re.search(r'run_\d+', r['id'])
                assert match and match[0] in official_map, r['id']
                splits.add(official_map[match[0]])
            assert len(splits) == 1, ('Official cross-split duplicate geometry', family, splits)
            designated = splits.pop()
        accepted = []
        for r in members:
            c = r['conditions']
            if a.dataset in ['blendednet']:
                mach = float(c['mach_number'])
                if a.dataset == 'blendednet':
                    condition_split = 'train' if mach <= bands['train_max_mach'] else 'val' if mach <= bands['val_max_mach'] else 'test'
                if condition_split != designated:
                    omissions.append(dict(id=r['id'], reason='Condition band and independent geometry split do not both match'))
                    continue
            accepted.append(r)
        if a.dataset.startswith('drivaer') and len(accepted) > 1:
            # Identical native meshes cannot multiply a single-condition case's weight in training.
            omissions.extend(dict(id=r['id'], reason='Duplicate physical geometry in a single-condition corpus') for r in accepted[1:])
            accepted = accepted[:1]
        if not accepted:
            continue
        reference = accepted[0]
        canonical = out / 'canonical_geometry' / (family + '.npz')
        canonical.parent.mkdir(exist_ok=True)
        if not canonical.exists():
            os.link(reference['source_geometry'], canonical)
        with np.load(reference['source_field']) as first:
            target_points = first['points'].copy()
            target_normals, target_weights = first['normals'].copy(), first['weights'].copy()
        for r in accepted:
            field_path = out / 'fields' / (r['id'] + '.npz')
            field_path.parent.mkdir(exist_ok=True)
            with np.load(r['source_field']) as f:
                if not field_path.exists():
                    os.link(r['source_field'], field_path)
            geometry_path = out / 'geometry' / (r['id'] + '.npz')
            geometry_path.parent.mkdir(exist_ok=True)
            if not geometry_path.exists():
                os.link(canonical, geometry_path)
            records.append(dict(id=r['id'], family=split_family, geometry_id=family, split=designated, path=str(geometry_path),
                                source_design=r['split_design']))
            c = r['conditions']
            if a.dataset == 'blendednet':
                conditions[r['id']] = [float(c['mach_number']), float(np.degrees(c['angle_of_attack'])), float(c['altitude'])]
            else:
                conditions[r['id']] = [0.]
    counts = {s: sum(r['split'] == s for r in records) for s in ['train', 'val', 'test']}
    families = {s: len({r['family'] for r in records if r['split'] == s}) for s in counts}
    if a.dataset == 'drivaerml':
        assert counts == dict(train=400, val=34, test=50), ('Official cohort incomplete', counts)
    if a.dataset in ['blendednet']:
        minimum = 30 if a.dataset == 'blendednet' else 20
        assert families['val'] >= minimum and families['test'] >= minimum, ('Insufficient independent held-out geometries', families)
    write(out / 'manifest.json', dict(schema=SCHEMA, units='metres', records=records))
    write(out / 'conditions.json', dict(names=names, cases=conditions))
    write(out / 'split_protocol.json', dict(bands=bands, counts=counts, families=families,
          assignment='most populated condition band per family' if a.all_condition_bands else 'geometry folds and condition bands',
          official_split=official, omitted=omissions))
    write(out / 'ready.json', dict(passed=True, counts=counts, families=families, full_inventory_cases=inventory['live_cases'],
          export_failed=len(reports) - len(valid)))
    print('export complete', a.dataset, counts, families, flush=True)


if __name__ == '__main__':
    main()
