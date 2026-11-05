"""Snapshot the first 35,493 prepared base-corpus meshes and validate them.

Writes manifest.json (the snapshot), clean.json (the snapshot without the files that fail validation),
audit.initial.json and audit.json next to this script. The corpus expansion reads clean.json and audit.json.

    PYTHONPATH=<study>/geometry_latents_minimal/src python make_snapshot.py
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

from geom_latents.data import read_manifest
from geom_latents.validation import validate_manifest, write_filtered_manifest

PILOT = Path(__file__).resolve().parent
ROOT = PILOT.parents[1]
PREPARED = Path('/workspace/runs/objaverse_base_030_20260924T172527Z/prepared')
LIMIT = 35493


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def make_snapshot():
    """The first LIMIT prepared files by source-manifest index, with their original split assignments."""
    source_path = ROOT / 'manifests/meshes.json'
    source = json.loads(source_path.read_text())
    records = source['records']
    ledger = json.loads((PREPARED / 'prepare.records.json').read_text())
    if ledger != [[i, record['id']] for i, record in enumerate(records)]:
        raise ValueError('Prepared-file ledger does not match the source manifest')
    completed = []
    with os.scandir(PREPARED) as entries:
        for entry in entries:
            match = re.fullmatch(r'shape_(\d{7})\.npz', entry.name)
            if match:
                completed.append((int(match.group(1)), entry.name))
    completed.sort()
    if len(completed) < LIMIT:
        raise ValueError(f'Only {len(completed)} complete files; expected at least {LIMIT}')
    selected = []
    for index, name in completed[:LIMIT]:
        record = records[index]
        selected.append(dict(record, path=str(PREPARED / name), source_mesh=record['path']))
    splits = dict(Counter(record['split'] for record in selected))
    if set(splits) != {'train', 'val', 'test'}:
        raise ValueError(f'Missing split in the snapshot: {splits}')
    write_json(PILOT / 'manifest.json', dict(source, records=selected, source_manifest=str(source_path),
                                             target_source='point-to-triangle unsigned distance',
                                             snapshot_policy=f'First {LIMIT} completed files by source-manifest index'))
    print(json.dumps(dict(selected_meshes=len(selected), completed_at_snapshot=len(completed), splits=splits)))


def audit_snapshot():
    """Validate every snapshot file, drop the failures, and write the audit for the filtered manifest."""
    report = validate_manifest(PILOT / 'manifest.json', workers=8, bad_records='skip')
    write_json(PILOT / 'audit.initial.json', report)
    write_filtered_manifest(PILOT / 'manifest.json', report, PILOT / 'clean.json')
    records, metadata = read_manifest(PILOT / 'clean.json')
    source_records, _ = read_manifest(PILOT / 'manifest.json')
    rejected = {row['id'] for row in report['failures']}
    if records != [row for row in source_records if row['id'] not in rejected]:
        raise ValueError('Filtering changed a retained record or its resolved path')
    if [row['id'] for row in report['shapes']] != [row['id'] for row in records]:
        raise ValueError('Validated records and filtered manifest differ')
    # Every retained file was checked above; reuse those results for the filtered manifest.
    write_json(PILOT / 'audit.json', dict(report, **metadata, failures=[],
                                          filtered_from_audit=str(PILOT / 'audit.initial.json')))
    print(json.dumps(dict(retained=len(records), splits=report['splits'], rejected=report['failures'])))


if __name__ == '__main__':
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ.setdefault(name, '1')
    make_snapshot()
    audit_snapshot()
