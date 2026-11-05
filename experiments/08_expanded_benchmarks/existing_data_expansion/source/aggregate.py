import argparse
import csv
import json

import numpy as np

import field_benchmark as fb
from util import ROOT, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    a = p.parse_args()
    data = ROOT / 'datasets' / a.dataset
    records = {r['id']: r for r in json.loads((data / 'manifest.json').read_text())['records']}
    for split in ['val', 'test']:
        args = ['score', '--bench', str(data / 'bench'), '--split', split, '--reference', 'pretrained_seed0']
        if split == 'test':
            args.append('--final')
        fb.main(args)
        with (data / 'bench/scores' / f'{split}_per_problem.csv').open() as f:
            rows = list(csv.DictReader(f))
        groups = {}
        for row in rows:
            name, seed = row['method'].rsplit('_seed', 1)
            key = (name, int(seed), row['setting'], records[row['car']]['family'])
            groups.setdefault(key, []).append(float(row['nrmse']))
        settings = sorted({k[2] for k in groups})
        families = sorted({k[3] for k in groups})
        names = sorted({k[0] for k in groups})
        summary = {}
        rng = np.random.default_rng(0)
        for setting in settings:
            matrices = {name: np.array([[np.mean(groups[name, seed, setting, family])
                         for family in families] for seed in range(3)]) for name in names}
            result = {name: dict(nrmse=float(values.mean()), per_seed_mean=values.mean(1).tolist())
                      for name, values in matrices.items()}
            for control in ['scratch', 'shape', 'conditioned']:
                paired = (matrices['pretrained'] - matrices[control]).mean(0)
                samples = paired[rng.integers(len(families), size=(5000, len(families)))].mean(1)
                result[f'pretrained_minus_{control}'] = dict(mean=float(paired.mean()),
                    geometry_grouped_95_interval=np.quantile(samples, [.025, .975]).tolist())
            summary[setting] = result
        # Equal weight per geometry and per condition within a geometry, then the mean over seeds.
        write(data / 'bench/scores' / f'{split}_grouped.json', dict(independent_geometry_families=len(families),
              seeds=[0, 1, 2], settings=summary))
    write(data / 'complete.json', dict(passed=True, seeds=[0, 1, 2]))


if __name__ == '__main__':
    main()
