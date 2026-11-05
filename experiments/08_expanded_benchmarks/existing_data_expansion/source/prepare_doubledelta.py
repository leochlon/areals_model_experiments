import json
import os
from pathlib import Path

from areal_tokens.data import SCHEMA
from util import ROOT, write


def main():
    source = Path('/workspace/runs/areals_multicondition_wings_20261023')
    out = ROOT / 'datasets/doubledelta'
    assert not (out / 'ready.json').exists()
    records = json.loads((source / 'manifest.json').read_text())['records']
    reference = json.loads((source / 'conditions_reference_only.json').read_text())['cases']
    groups = {}
    for record in records:
        groups.setdefault(record['family'], set()).add(record['grid_id'])
    representatives = {family: min(grids) for family, grids in groups.items()}
    selected, conditions = [], {}
    for record in records:
        if record['grid_id'] != representatives[record['family']]:
            continue
        angle = reference[record['id']]['aoa']
        split = record['split']
        if (split == 'train' and angle > 16) or (split == 'val' and angle != 17) or (split == 'test' and angle < 18):
            continue
        geometry = Path(record['path'])
        if not geometry.is_absolute():
            geometry = source / geometry
        assert geometry.is_file()
        field = source / 'fields' / (record['id'] + '.npz')
        destination = out / 'fields' / field.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            os.link(field, destination)
        selected.append({**record, 'path': str(geometry), 'geometry_id': record['family']})
        conditions[record['id']] = [float(angle)]
    counts = {s: sum(r['split'] == s for r in selected) for s in ['train', 'val', 'test']}
    assert counts == dict(train=1368, val=16, test=32), counts
    manifest = dict(schema=SCHEMA, units='source_length', records=selected)
    write(out / 'manifest.json', manifest)
    write(out / 'conditions.json', dict(names=['aoa_degrees'], cases=conditions))
    # One source grid per physical wing (the lexicographically first), so aliases do not multiply examples.
    write(out / 'split_protocol.json', dict(train_angles=[11, 12, 13, 14, 15, 16], val_angles=[17], test_angles=[18, 19],
        physical_families=dict(train=228, val=16, test=16)))
    write(out / 'ready.json', dict(passed=True, counts=counts))
    print('doubledelta prepared', counts, flush=True)


if __name__ == '__main__':
    main()
