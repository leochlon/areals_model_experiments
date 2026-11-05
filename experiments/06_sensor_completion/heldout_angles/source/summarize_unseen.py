import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/workspace/runs/areals_architecture_controls_20261024/unseen_conditions')
sys.path.insert(0, '/workspace/runs/areals_multicondition_wings_20261023')
import field_benchmark as fb


def main():
    bench = ROOT / 'bench'
    config = fb.bench_config(bench)
    records = {r['id']: r for r in fb.records_of(config)}
    methods = ['geometry', 'measurements', 'conditioned']
    for split in ['val', 'test']:
        command = ['score', '--bench', str(bench), '--split', split, '--methods', *methods, '--reference', 'geometry']
        if split == 'test':
            command.append('--final')
        fb.main(command)
        rows = list(csv.DictReader((bench / 'scores' / (split + '_per_problem.csv')).open()))
        meta, arrays = fb.read_observations(bench, split)
        entry_keys = {(e['car'], e['setting'], str(e['repeat'])): e['key'] for e in meta['entries']}
        # Recompute every score from the saved predictions and pressure targets.
        maximum = 0.0
        for method in methods:
            with np.load(bench / 'predictions' / method / (split + '.npz')) as predictions:
                field = None
                previous = None
                for row in [r for r in rows if r['method'] == method]:
                    identity = row['car']
                    if identity != previous:
                        field = fb.load_field(config['fields'], identity)
                        previous = identity
                    key = entry_keys[(row['car'], row['setting'], row['repeat'])]
                    visible = arrays[key]
                    hidden = np.ones(len(field.values), bool)
                    hidden[visible] = False
                    w = field.weights / field.weights.sum()
                    mean = w @ field.values
                    spread2 = w @ ((field.values - mean) ** 2)
                    mse = np.average((predictions[key][hidden].astype(np.float64) - field.values[hidden]) ** 2,
                                     weights=w[hidden])
                    independent = float(np.sqrt(mse / spread2))
                    maximum = max(maximum, abs(independent - float(row['nrmse'])))
        assert maximum < 1e-10
        families = sorted({records[r['car']]['family'] for r in rows})
        draws = np.random.default_rng(20261024).integers(0, len(families), (4000, len(families)))
        summary = []
        for setting in sorted({r['setting'] for r in rows}):
            matrix = np.asarray([[np.mean([float(r['nrmse']) for r in rows
                       if r['method'] == method and r['setting'] == setting
                       and records[r['car']]['family'] == family])
                       for method in methods] for family in families])
            square = np.asarray([[np.mean([float(r['nrmse']) ** 2 for r in rows
                       if r['method'] == method and r['setting'] == setting
                       and records[r['car']]['family'] == family])
                       for method in methods] for family in families])
            b, bs = matrix[draws].mean(1), square[draws].mean(1)
            means = square.mean(0)
            denominator = means[0] - means[2]
            stable = np.quantile(bs[:, 0] - bs[:, 2], .025) > 0
            gap = float((means[0] - means[1]) / denominator) if denominator > 0 else None
            intervals = np.quantile((bs[:, 0] - bs[:, 1]) / (bs[:, 0] - bs[:, 2]), [.025, .975]).tolist() if stable else None
            summary.append(dict(setting=setting, nrmse={method: dict(mean=float(matrix[:, i].mean()),
                          low=float(np.quantile(b[:, i], .025)), high=float(np.quantile(b[:, i], .975)))
                          for i, method in enumerate(methods)}, gap_closed=gap, gap_closed_95=intervals,
                          reference_gap_stable=bool(stable)))
        out = ROOT / 'results'
        out.mkdir(exist_ok=True)
        (out / (split + '_grouped.json')).write_text(json.dumps(dict(split=split, physical_geometries=len(families),
             cases=len({r['car'] for r in rows}), settings=summary, independent_maximum_difference=maximum,
             angles=dict(train=[11, 13, 15, 17, 19], val=[11, 13, 15, 17, 19], test=[12, 14, 16, 18])), indent=2)+'\n')
        print('GROUPED', split, json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
