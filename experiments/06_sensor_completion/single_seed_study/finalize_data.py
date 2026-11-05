from collections import Counter, defaultdict
import json
import os
from pathlib import Path

import numpy as np

from prepare import ROOT, write


def main():
    receipts = [json.loads(p.read_text()) for p in sorted((ROOT / 'receipts').glob('*.json'))]
    assert len(receipts) == 272
    assert not json.loads((ROOT / 'rejected_families.json').read_text())
    groups = defaultdict(list)
    for receipt in receipts:
        groups[receipt['geometry_signature']].append(receipt)
    test = {signature for signature, rows in groups.items() if any(r['split'] == 'test' for r in rows)}
    assert len(test) == 16
    candidates = sorted(set(groups) - test)
    validation = set(np.random.default_rng(20261023).choice(candidates, 16, replace=False))
    assert not validation & test
    records, conditions, grouping = [], {}, []
    for signature, aliases in sorted(groups.items()):
        aliases.sort(key=lambda r: r['family'])
        physical_family = 'ddg_' + signature[:20]
        split = 'test' if signature in test else 'val' if signature in validation else 'train'
        representative = aliases[0]['family']
        geometry = ROOT / 'geometry_families' / f'{representative}.npz'
        for receipt in aliases:
            for record in receipt['records']:
                destination = ROOT / record['path']
                if destination.stat().st_ino != geometry.stat().st_ino:
                    destination.unlink()
                    os.link(geometry, destination)
                records.append(dict(record, family=physical_family, split=split, grid_id=receipt['family']))
                info = receipt['conditions'][record['id']]
                conditions[record['id']] = dict(info, family=physical_family, grid_id=receipt['family'])
        grouping.append(dict(family=physical_family, split=split, signature=signature,
                             representative=representative, aliases=[r['family'] for r in aliases],
                             geometry_path=str(geometry)))
    records.sort(key=lambda r: r['id'])
    schema = json.loads(Path('/workspace/runs/areal_drivaerml_420_epoch14_20260927T162928Z/manifest.json').read_text())['schema']
    write(ROOT / 'manifest.json', dict(schema=schema, units='metres', records=records))
    angles = [conditions[r['id']]['aoa'] for r in records if r['split'] == 'train']
    write(ROOT / 'conditions_reference_only.json', dict(mean=float(np.mean(angles)), std=float(np.std(angles)), cases=conditions))
    write(ROOT / 'geometry_groups.json', grouping)
    protocol = json.loads((ROOT / 'preparation_protocol.json').read_text())
    protocol.update(split_revision=2, named_geometries=272, physical_geometries=len(groups),
                    split_physical_geometries=dict(Counter(r['split'] for r in grouping)),
                    split_cases=dict(Counter(r['split'] for r in records)))
    write(ROOT / 'preparation_protocol.json', protocol)
    write(ROOT / 'study_protocol.json', dict(
        seed=0, epochs=40, global_batch=8, base_lr=1e-4, new_lr=1e-3,
        roles=['geometry', 'measurements', 'conditioned'],
        selection_settings=dict(geometry=['sensors-5 (ignored by model)'], conditioned=['sensors-5 (ignored by model)'],
                                measurements=['sensors-20','sensors-500','patch-64-0.60']),
        evaluation=dict(patch_counts=[64], hidden=[.25,.6,.9], sensors=[5,20,100,500], unobserved=True),
        gap_formula='(MSE_geometry - MSE_measurements)/(MSE_geometry - MSE_conditioned); undefined if denominator <= 0.'))
    write(ROOT / 'preparation_complete.json', dict(
        status='PASS', named_geometries=272, physical_geometries=len(groups), records=len(records),
        split_geometries=dict(Counter(r['split'] for r in grouping)),
        split_cases=dict(Counter(r['split'] for r in records)),
        minimum_common_area_coverage=min(r['common_area_coverage'] for r in receipts),
        minimum_angle_variation_rms=min(r['angle_variation_rms'] for r in receipts)))
    print(json.dumps(json.loads((ROOT / 'preparation_complete.json').read_text())), flush=True)


if __name__ == '__main__':
    main()
