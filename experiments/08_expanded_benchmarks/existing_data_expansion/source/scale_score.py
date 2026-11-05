import numpy as np

import field_benchmark as fb
from util import ROOT, write

data = ROOT / 'datasets/drivaernet'
bench = data / 'bench'
config = fb.bench_config(bench)
records = {r['id']: r for r in fb.records_of(config)}
for split in ['val', 'test']:
    meta, observation = fb.read_observations(bench, split)
    predictions = {name: np.load(bench / 'predictions' / name / (split + '.npz'))
                   for name in [f'{arm}_seed{seed}' for arm in ['pretrained', 'small'] for seed in range(3)]}
    groups = {}
    for car, entries in fb.by_car(meta['entries']).items():
        field = fb.load_field(config['fields'], car)
        for name, packed in predictions.items():
            for entry in entries:
                score = fb.problem_metrics(field, observation[entry['key']], packed[entry['key']])['nrmse']
                groups.setdefault((name, entry['setting'], records[car]['family']), []).append(score)
    families = sorted({k[2] for k in groups})
    summary = {}
    rng = np.random.default_rng(0)
    for setting in sorted({k[1] for k in groups}):
        values = {arm: np.asarray([[np.mean(groups[f'{arm}_seed{seed}', setting, family]) for family in families]
                                   for seed in range(3)]) for arm in ['pretrained', 'small']}
        delta = (values['pretrained'] - values['small']).mean(0)
        boot = delta[rng.integers(len(families), size=(5000, len(families)))].mean(1)
        summary[setting] = dict(full_data_nrmse=float(values['pretrained'].mean()), small_data_nrmse=float(values['small'].mean()),
              full_minus_small=float(delta.mean()), paired_geometry_95_interval=np.quantile(boot, [.025, .975]).tolist())
    write(bench / 'scores' / f'{split}_data_scaling.json', dict(passed=True, independent_geometry_families=len(families),
          small_training_families=400, full_training_families=sum(r['split'] == 'train' for r in records.values()),
          seeds=[0, 1, 2], settings=summary))
    for packed in predictions.values():
        packed.close()
