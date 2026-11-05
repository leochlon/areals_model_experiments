"""The matched cohort: every object that passed both preparations, and its pack index."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
FULL = Path('/workspace/runs/objaverse_target_comparison_20261030')


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n'); tmp.replace(path)


def snapshot():
    assert not (ROOT / 'protocol/cohort.json').exists(), 'Snapshot already frozen'
    original = json.loads((FULL / 'protocol/objects.json').read_text())
    observed = {}
    for path in sorted((FULL / 'logs').glob('prepare-1078785_*.log')):
        for line in path.read_text(errors='replace').splitlines():
            try: record = json.loads(line)
            except ValueError: continue
            if isinstance(record, dict) and all(k in record for k in ['id', 'patch', 'query', 'done']):
                observed[record['id']] = record
    chosen = [r for r in original if r['id'] in observed and
              observed[r['id']]['patch'] == observed[r['id']]['query'] == 'ok']
    assert len({r['id'] for r in chosen}) == len(chosen)
    counts = dict(Counter(r['split'] for r in chosen))
    assert all(counts.get(s, 0) > 0 for s in ['train', 'val', 'test'])
    families = {}
    for row in chosen:
        assert families.setdefault(row['family'], row['split']) == row['split']
    write(ROOT / 'protocol/objects.json', chosen)
    write(ROOT / 'protocol/cohort.json', dict(created_unix=time.time(), counts=counts,
        processed=len(observed), patch_ok=sum(r['patch'] == 'ok' for r in observed.values()),
        query_ok=sum(r['query'] == 'ok' for r in observed.values()),
        rule='All patch-ok AND query-ok objects reported at snapshot time, preserving original splits and ordering.',
        exclusions='Neither failed nor unfinished objects enter any arm. No replacement objects.',
        selected_pack_numbers=sorted({r['index'] // 128 for r in chosen})))
    write(ROOT / 'windsor/index.json', json.loads((FULL / 'windsor/index.json').read_text()))
    print(json.dumps(json.loads((ROOT / 'protocol/cohort.json').read_text())), flush=True)


def aggregate():
    rows = json.loads((ROOT / 'protocol/objects.json').read_text())
    packs = sorted({r['index'] // 128 for r in rows})
    started = time.monotonic()
    while True:
        missing = [i for i in packs if not (FULL / f'prepared/pack_{i:04d}.json').exists()]
        write(ROOT / 'prepared/status.json', dict(waiting_for_closed_packs=missing, selected_objects=len(rows),
            elapsed_seconds=time.monotonic() - started))
        if not missing: break
        if time.monotonic() - started > 7 * 3600: raise TimeoutError(missing)
        print('Waiting for immutable production packs:', missing, flush=True); time.sleep(30)
    records = {}; receipts = []
    for i in packs:
        path = FULL / f'prepared/pack_{i:04d}.json'; doc = json.loads(path.read_text())
        pack = path.parent / doc['path']
        receipts.append(dict(path=str(pack), receipt=str(path)))
        for record in doc['records']: records[record['id']] = dict(record, pack=str(pack))
    selected = []
    for row in rows:
        record = records[row['id']]
        assert record['split'] == row['split'] and record['family'] == row['family']
        assert record['patch']['status'] == record['query']['status'] == 'ok'
        selected.append(dict(row, **{k: record[k] for k in ['pack', 'patch', 'query']}))
    write(ROOT / 'prepared/index.json', dict(complete=True, objects=selected, packs=receipts,
        identical_ids_all_arms=True, counts=dict(Counter(r['split'] for r in selected))))
    # The three arms must see the same objects in the same order, including epoch boundaries.
    from geometry_data import object_order
    import numpy as np
    length = 10626 * 64; sequences = {}
    for arm in ['patch', 'query', 'areals']:
        eligible = [r for r in selected if r['split'] == 'train'] if arm == 'areals' else selected
        order = object_order(eligible, 'train', 17, length)
        sequences[arm] = np.array([r['index'] for r in eligible], dtype=np.int64)[order]
    assert all(np.array_equal(sequences['patch'], s) for s in sequences.values())
    write(ROOT / 'checks/cohort.json', dict(passed=True, identical_object_sequences=True,
        exposures=length, counts=dict(Counter(r['split'] for r in selected))))
    print(json.dumps(json.loads((ROOT / 'checks/cohort.json').read_text())), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['snapshot', 'aggregate'])
    globals()[p.parse_args().command]()
