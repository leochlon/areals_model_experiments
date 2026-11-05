import collections
import datetime
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/workspace/runs/objaverse_target_comparison_20261030')
PREVIOUS = Path('/workspace/runs/matched_priors_windsor_20261030')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    if (ROOT / 'protocol/cohort.json').exists():
        raise FileExistsError('protocol/cohort.json exists; the cohort was already built')
    observed = datetime.datetime.now(datetime.timezone.utc).isoformat()
    original_path = SOURCE / 'protocol/objects.json'
    original = json.loads(original_path.read_text())
    receipts = sorted((SOURCE / 'prepared').glob('pack_*.json'))
    successes, packs = {}, []
    outcomes = {branch: collections.Counter() for branch in ('patch', 'query')}
    for path in receipts:
        doc = json.loads(path.read_text())
        pack = path.parent / doc['path']
        if not pack.is_file() or pack.stat().st_size != doc['bytes']:
            raise ValueError(f'Incomplete closed pack: {pack}')
        packs.append(dict(path=str(pack), receipt=str(path)))
        for row in doc['records']:
            for branch in outcomes:
                outcomes[branch][row[branch]['status']] += 1
            if row['patch']['status'] == row['query']['status'] == 'ok':
                if row['id'] in successes:
                    raise ValueError('Duplicate prepared identity')
                successes[row['id']] = dict(row, pack=str(pack))
    native, prepared = [], []
    families = {}
    for row in original:
        if row['id'] not in successes:
            continue
        target = successes[row['id']]
        for key in ('split', 'family'):
            assert row[key] == target[key], (row['id'], key)
        assert families.setdefault(row['family'], row['split']) == row['split']
        native.append(row)
        prepared.append(dict(row, **{k: target[k] for k in ('pack', 'patch', 'query')}))
    assert len(native) == len(successes)
    counts = dict(collections.Counter(r['split'] for r in native))
    assert all(counts.get(s, 0) for s in ('train', 'val', 'test'))
    write(ROOT / 'protocol/objects.json', native)
    write(ROOT / 'prepared/index.json', dict(complete=True, counts=counts, objects=prepared, packs=packs,
                                           identical_ids_all_arms=True))
    for name in ('windsor/index.json', 'windsor_protocol/subsets.json'):
        destination = ROOT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PREVIOUS / name, destination)
    small_ids = set()
    for split, n in [('train', 8), ('val', 2), ('test', 2)]:
        small_ids.update([r['id'] for r in native if r['split'] == split][:n])
    write(ROOT / 'checks/real_data/objects.json', [r for r in native if r['id'] in small_ids])
    write(ROOT / 'checks/real_data/index.json', dict(complete=True, objects=[r for r in prepared if r['id'] in small_ids]))
    report = dict(observed_utc=observed, counts=counts, closed_packs=len(packs), outcomes=outcomes,
                  rule='All patch-ok AND query-ok objects in closed packs at snapshot time; original splits retained.')
    write(ROOT / 'protocol/cohort.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
