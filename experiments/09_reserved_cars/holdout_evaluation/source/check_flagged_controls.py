import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT = Path('/workspace/runs/areals_architecture_controls_20261024')
sys.path.insert(0, str(AUDIT / 'source'))


def main():
    from areal_tokens.model import Rotary3D
    import position_control
    import transfer_architecture as architecture

    catalog = json.loads((ROOT / 'models.json').read_text())
    checked = []
    for record in catalog:
        if record['family'] != 'architecture' or not record['arm'].endswith('_no_rotary'):
            continue
        architecture.ARM, architecture.SEED = record['arm'], record['seed']
        model, _, _ = architecture.field_load(record['checkpoint'], 'cpu')
        modules = [m for m in model.modules() if isinstance(m, Rotary3D)]
        assert modules and all(m.forward.__func__ is position_control.identity for m in modules)
        checked.append(dict(model=record['name'], rotary_modules=len(modules)))
    assert len(checked) == 6
    rows = list(csv.DictReader((ROOT / 'results/complete/per_problem.csv').open()))
    ids = json.loads((ROOT / 'protocol.json').read_text())['test_ids']
    values = {(r['group'], int(r['seed']), r['car']): float(r['value']) for r in rows if r['family'] == 'frozen'}
    # Compute the paired differences directly, without importing the registered scorer.
    differences = np.array([sum(values['areal', seed, car] - values['native_normals', seed, car]
                                for seed in range(3)) / 3 for car in ids])
    indices = np.random.default_rng(20261024).integers(0, 48, (4000, 48))
    low, high = np.quantile(differences[indices].mean(axis=1), [.025, .975])
    summary = json.loads((ROOT / 'results/complete/summary.json').read_text())
    saved = next(r for r in summary['comparisons'] if r['family'] == 'frozen' and r['second'] == 'native_normals')['first_minus_second']
    np.testing.assert_allclose([low, high], [saved['low'], saved['high']], atol=1e-14, rtol=0)
    means = summary['summaries']['architecture']
    anchored = means['anchored_no_rotary']['unobserved']['nrmse']['mean']
    unanchored = means['unanchored_no_rotary']['unobserved']['nrmse']['mean']
    assert unanchored < anchored
    out = dict(passed=True, no_rotary_configuration_checks=checked,
               anchored_no_rotary_mean_nrmse=anchored, unanchored_no_rotary_mean_nrmse=unanchored,
               unanchored_mean_nrmse_is_lower=bool(unanchored < anchored),
               frozen_areals_minus_native_normals_car_bootstrap_95=dict(low=float(low), high=float(high),
                   low_decimal=f'{low:.17f}', high_decimal=f'{high:.17f}', includes_zero=bool(low <= 0 <= high)))
    (ROOT / 'results/complete/flagged_controls_check.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
