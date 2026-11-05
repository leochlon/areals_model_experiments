import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.umask(0o077)

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path('/workspace/runs/gabi_graph_ablation_20261028')
data = Path('/workspace/runs/areals_multicondition_wings_20261023')
original = Path('/workspace/runs/gabi_doubledelta_20261023')

def read(path):
    return json.loads(path.read_text())

report = read(root / 'results/test_comparison.json')
observations = read(original / 'bench/observations/test.json')
primary = {(e['car'], e['setting']): e for e in observations['entries'] if e['setting'] == 'sensors-20'}
assert len(primary) == 144
with np.load(original / 'bench/observations/test.npz') as arrays:
    masks = {e['key']: arrays[e['key']] for e in primary.values()}
fields = {}
for case, _ in primary:
    with np.load(data / 'fields' / (case + '.npz')) as arrays:
        truth, areas, normals = [arrays[k].astype(np.float64) for k in ['values', 'weights', 'normals']]
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    spread = np.sqrt(np.average((truth - np.average(truth, weights=areas)) ** 2, weights=areas))
    force = np.sum((truth * areas)[:, None] * normals, axis=0)
    fields[case] = truth, areas, normals, spread, force

checked = 0
max_error = {'nrmse': 0.0, 'force_rel': 0.0}
verified_runs = []
for seed in range(3):
    for variant in ['euclidean', 'surface']:
        name = f'{variant}_seed{seed}'
        path = root / 'runs' / name
        done = read(path / 'complete.json')
        assert done['status'] == 'PASS' and done['epochs'] == 455 and done['updates'] == 10010
        history = read(path / 'history.json')
        assert len(history) == 455 and history[-1]['updates'] == 10010
        assert min(history, key=lambda r: r['validation'][0])['epoch'] == done['selected_epoch']
        scores = read(path / 'test/scores.json')
        assert scores['status'] == 'PASS' and len(scores['rows']) == 1152
        groups = defaultdict(list)
        selected = [r for r in scores['rows'] if r['setting'] == 'sensors-20']
        assert {(r['case'], r['setting']) for r in selected} == set(primary)
        for row in selected:
            groups[row['grid_id']].append(row)
        actual = {'nrmse': [], 'force_rel': []}
        geometry_counts = defaultdict(int)
        for grid, rows in groups.items():
            archive = path / 'test' / (grid + '.npz')
            with np.load(archive) as predictions:
                for row in rows:
                    entry = primary[row['case'], row['setting']]
                    prediction = predictions[entry['key']].astype(np.float64)
                    truth, areas, normals, spread, force = fields[row['case']]
                    hidden = np.ones(truth.size, dtype=bool)
                    hidden[masks[entry['key']]] = False
                    nrmse = np.sqrt(np.average((prediction[hidden] - truth[hidden]) ** 2, weights=areas[hidden])) / spread
                    completed = np.where(hidden, prediction, truth)
                    predicted_force = np.sum((completed * areas)[:, None] * normals, axis=0)
                    force_rel = np.linalg.norm(predicted_force - force) / np.linalg.norm(force)
                    for metric, value in [('nrmse', nrmse), ('force_rel', force_rel)]:
                        assert np.isfinite(value)
                        max_error[metric] = max(max_error[metric], abs(float(value) - row[metric]))
                        actual[metric].append(float(value))
                    geometry_counts[row['physical_geometry']] += 1
                    checked += 1
        assert len(geometry_counts) == 16 and set(geometry_counts.values()) == {9}
        means = {m: float(np.mean(values)) for m, values in actual.items()}
        for metric, value in means.items():
            pair = next(r for r in report['results']['sensors-20'][metric]['paired_seed_results'] if r['seed'] == seed)
            assert abs(value - pair[variant]) < 1e-10
        verified_runs.append(dict(run=name, selected_epoch=done['selected_epoch'], means=means))
assert checked == 864 and max(max_error.values()) < 1e-10
result = dict(status='PASS', scope='recomputation of the 20-sensor test nRMSE and force error from saved predictions, all six runs',
    predictions_checked=checked, test_cases=144, physical_geometries=16, angles_per_geometry=9,
    maximum_difference_from_recorded_scores=max_error, verified_runs=verified_runs)
print(json.dumps(result, indent=2, allow_nan=False))
