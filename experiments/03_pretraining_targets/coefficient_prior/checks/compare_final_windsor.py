import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'AREALS_shared_query'))
import shared_query
from shared_query.pressure import read_json, write_json


def main():
    old_root = Path('/workspace/runs/matched_priors_windsor_20261030')
    old_path = old_root / 'results/comparison.json'
    current_path = ROOT / 'results/operator32_comparison.json'
    for relative in ('windsor/index.json', 'windsor_protocol/subsets.json'):
        old, current = read_json(old_root / relative), read_json(ROOT / relative)
        if relative.endswith('index.json'):
            old = {r['id']: r['split'] for r in old['records']}
            current = {r['id']: r['split'] for r in current['records']}
        assert old == current, relative
    previous = {r['budget']: r for r in read_json(old_path)['comparisons'] if r['arm'] == 'patch'}
    current = read_json(current_path)
    assert current['complete']
    rows = []
    for row in current['results']:
        old = previous[row['training_cars']]
        rows.append(dict(training_cars=row['training_cars'],
                         shared_query_pretrained_mean=row['pretrained_mean_nrmse'],
                         shared_query_scratch_mean=row['scratch_mean_nrmse'],
                         previous_patch_pretrained_mean=old['pretrained_mean'],
                         previous_patch_scratch_mean=old['scratch_mean'],
                         pretrained_relative_change=row['pretrained_mean_nrmse'] / old['pretrained_mean'] - 1,
                         shared_query_seeds=row['seeds'], previous_patch_pretrained_seeds=old['pretrained_seeds'],
                         previous_patch_scratch_seeds=old['scratch_seeds']))
    report = dict(complete=True, metric=current['metric'], test_cars=35, seeds=[0, 1, 2],
                  geometry_updates_each=10626, matched_windsor_splits_and_subsets=True,
                  sources=[str(old_path), str(current_path)], results=rows)
    write_json(ROOT / 'results/final_vs_previous_patch.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
