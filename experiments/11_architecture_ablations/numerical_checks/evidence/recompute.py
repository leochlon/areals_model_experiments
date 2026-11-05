"""Recompute several paper numbers from the recorded per-case scores."""
from pathlib import Path
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parent


def csv_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    manifest = json.loads((ROOT / 'dd_manifest.json').read_text())
    families = {r['id']: r['family'] for r in manifest['records']}
    long_rows = csv_rows(ROOT / 'gabi_test_per_problem.csv')
    dd_rows = csv_rows(ROOT / 'areals_dd_test_per_problem.csv')
    selected = [r for r in long_rows if r['setting'] == 'sensors-20']
    means = {m: float(np.mean([float(r['nrmse']) for r in selected if r['method'] == m]))
             for m in ['geometry', 'measurements', 'conditioned', 'gabi']}
    group_ids = sorted({families[r['car']] for r in selected})
    squared = np.array([[np.mean([float(r['nrmse']) ** 2 for r in dd_rows
                                 if r['setting'] == 'sensors-20' and r['method'] == m
                                 and families[r['car']] == group])
                         for m in ['geometry', 'measurements', 'conditioned']]
                        for group in group_ids])
    mse = squared.mean(0)
    gap = (mse[0] - mse[1]) / (mse[0] - mse[2])
    sample = np.random.default_rng(20261023).integers(0, len(group_ids), (4000, len(group_ids)))
    boot = squared[sample].mean(1)
    ratios = (boot[:, 0] - boot[:, 1]) / (boot[:, 0] - boot[:, 2])
    result = dict(
        sensor_count=20, physical_wing_groups=len(group_ids), cases=len({r['car'] for r in selected}),
        conditions_per_geometry=sorted({len({r['car'] for r in selected if families[r['car']] == group})
                                        for group in group_ids}),
        mean_nrmse=means,
        areals_reduction_against_gabi_percent=100 * (1 - means['measurements'] / means['gabi']),
        mean_casewise_normalized_mse=dict(zip(['geometry', 'measurements', 'conditioned'], mse.tolist())),
        normalized_mse_gap_closed_percent=100 * float(gap),
        normalized_mse_gap_closed_95_percent=(100 * np.quantile(ratios, [.025, .975])).tolist(),
        bootstrap=dict(unit='physical wing geometry retaining all nine angles', draws=4000, seed=20261023),
    )
    density_path = ROOT / 'density_fixed_rows.json'
    density = json.loads(density_path.read_text())
    fixed = [r for r in density if r['mode'] == 'biased_fixed_anchors_frame']
    density_means = {arm: float(np.mean([r['prediction_change'] for r in fixed if r['arm'] == arm]))
                     for arm in ['reference', 'ordinary_attention']}
    result['density_fixed_anchors_and_frame'] = dict(
        source=str(density_path),
        objects=len({r['id'] for r in fixed}), seeds=sorted({r['seed'] for r in fixed}),
        draws=sorted({r['draw'] for r in fixed}), rows=len(fixed),
        metric='RMS difference from the same-object uniform-sample prediction, divided by object RMS radius',
        mean_drift=density_means,
        area_drift_fraction_of_ordinary=density_means['reference'] / density_means['ordinary_attention'],
        area_relative_drift_reduction_percent=100 * (1 - density_means['reference'] / density_means['ordinary_attention']),
        ordinary_drift_divided_by_area=density_means['ordinary_attention'] / density_means['reference'],
    )
    frozen_path = ROOT / 'frozen_48_car_scores.csv'
    frozen = [r for r in csv_rows(frozen_path) if r['family'] == 'frozen' and r['metric'] == 'rel_l2'
              and r['group'] in ['areal', 'native']]
    pair = {}
    for arm in ['areal', 'native']:
        rows = [r for r in frozen if r['group'] == arm]
        pair[arm] = dict(mean_relative_l2=float(np.mean([float(r['value']) for r in rows])),
                         cars=len({r['car'] for r in rows}), seeds=sorted({int(r['seed']) for r in rows}),
                         scores=len(rows))
    result['frozen_48_car_comparison'] = dict(source=str(frozen_path), methods=pair)
    assert result['physical_wing_groups'] == 16 and result['cases'] == 144
    assert result['conditions_per_geometry'] == [9]
    assert round(result['areals_reduction_against_gabi_percent'], 1) == 77.1
    assert round(result['normalized_mse_gap_closed_percent'], 1) == 90.7
    assert [round(x, 1) for x in result['normalized_mse_gap_closed_95_percent']] == [84.4, 96.9]
    assert round(result['density_fixed_anchors_and_frame']['area_relative_drift_reduction_percent'], 1) == 81.5
    assert round(pair['areal']['mean_relative_l2'], 5) == .32885
    assert round(pair['native']['mean_relative_l2'], 5) == .33193
    result['all_assertions_passed'] = True
    (ROOT / 'independent_case_reduction.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
