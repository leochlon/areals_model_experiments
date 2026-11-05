"""Decompose invariant-prediction error and reassess the regional gain contrast."""
from collections import defaultdict
from pathlib import Path
import csv
import json
import math
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'areals-attention-attribution-20261026'
sys.path.insert(0, str(SOURCE))
import theory_tests as tt


def write_json(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False,
                                       default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')


def write_csv(name, rows):
    with (ROOT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def masks(curv, extreme):
    out = {'all': np.ones_like(extreme, bool),
           'local': (curv == 2) & (extreme < 2), 'global': (curv == 0) & (extreme == 2)}
    for c in range(3):
        out[f'curvature_{c}'] = np.broadcast_to(curv == c, extreme.shape)
        for x in range(3):
            out[f'cell_{c}_{x}'] = (curv == c) & (extreme == x)
    for x in range(3):
        out[f'extremity_{x}'] = extreme == x
    return out


def extremity(field, area):
    w = area / area.sum()
    spread = math.sqrt(float(w @ (field - w @ field) ** 2))
    spread2 = spread * spread
    assert spread2 > 0
    deviation = np.abs(field - tt.weighted_quantile(field, area, [.5])[0]) / spread
    cut = tt.weighted_quantile(deviation, area, [1/3, 2/3])
    return np.digitize(deviation, cut), spread2


checks = dict(expected_cases=144, expected_wings=16, predictions_identical=True,
              geometry_and_weights_identical=True, all_points_scored=True,
              max_decomposition_relative_error=0.0, max_arithmetic_identity_relative_error=0.0,
              max_gain_shrinkage_identity_error=0.0)
rows = []
groups = defaultdict(list)
audit = json.loads((SOURCE / 'real/export_audit.json').read_text())
expected = {c['id'] for c in audit['cases']}
assert {p.stem for p in (SOURCE / 'real/attribution').glob('*.npz')} == expected
for case in audit['cases']:
    groups[case['group']].append(case['id'])
assert len(groups) == 16 and all(len(ids) == 9 for ids in groups.values())

for number, (group, ids) in enumerate(sorted(groups.items()), 1):
    fields = []
    reference = None
    for cid in sorted(ids):
        path = SOURCE / 'real/attribution' / f'{cid}.npz'
        with np.load(path, allow_pickle=False) as z:
            current = {k: z[k] for k in ('X', 'N', 'A', 'R', 'pred_finetuned', 'pred_scratch')}
            assert z['scored'].all() and str(z['group']) == group
            if reference is None:
                reference = current
            else:
                for k in current:
                    np.testing.assert_array_equal(current[k], reference[k])
            fields.append(z['f'])
    F = np.stack(fields).astype(np.float64)
    A, X, N, R = (reference[k] for k in ('A', 'X', 'N', 'R'))
    P = np.stack([reference['pred_finetuned'], reference['pred_scratch']]).astype(np.float64)
    assert F.shape == (9, 32000) and np.isfinite(F).all() and np.isfinite(P).all()
    mu9 = F.mean(0)
    kappa = tt.shape_operator_norm(X, N, .25 * float(R)) * float(R)
    assert np.isfinite(kappa).all()
    curv = np.digitize(kappa, tt.weighted_quantile(kappa, A, [1/3, 2/3]))
    bins, spreads = zip(*(extremity(f, A) for f in F))
    per_angle_bins, spreads = np.stack(bins), np.asarray(spreads)
    fixed_bins, mean_spread2 = extremity(mu9, A)

    for scheme, region_masks, normalisers in (
        ('original', masks(curv, per_angle_bins), 1/spreads),
        ('fixed', masks(curv, np.broadcast_to(fixed_bins, F.shape)), np.full(9, 1/mean_spread2)),
    ):
        for region, mask in region_masks.items():
            weight = mask * normalisers[:, None]
            W = weight.sum(0)
            keep = W > 0
            assert keep.any(), (group, scheme, region)
            mean = np.divide((weight * F).sum(0), W, out=np.zeros_like(W), where=keep)
            total = np.array([np.sum(A * np.sum(weight * (p - F) ** 2, axis=0)) for p in P])
            bias = np.array([np.sum(A * W * (p - mean) ** 2) for p in P])
            variance = float(np.sum(A * np.sum(weight * (F - mean) ** 2, axis=0)))
            rel = float(np.max(np.abs(total - bias - variance) / np.maximum(total, 1e-30)))
            checks['max_decomposition_relative_error'] = max(checks['max_decomposition_relative_error'], rel)
            assert rel < 1e-11
            bias9 = np.array([np.sum(A * W * (p - mu9) ** 2) for p in P])
            variance9 = float(np.sum(A * np.sum(weight * (F - mu9) ** 2, axis=0)))
            cross9 = np.array([-2*np.sum(A * (p - mu9) * np.sum(weight * (F - mu9), axis=0)) for p in P])
            rel9 = float(np.max(np.abs(total - bias9 - variance9 - cross9) / np.maximum(total, 1e-30)))
            checks['max_arithmetic_identity_relative_error'] = max(checks['max_arithmetic_identity_relative_error'], rel9)
            assert rel9 < 1e-11 and total[1] > 0 and bias[1] > 0 and bias9[1] > 0
            shrink_error = abs((1-total[0]/total[1]) - (1-bias[0]/bias[1])*(bias[1]/total[1]))
            checks['max_gain_shrinkage_identity_error'] = max(checks['max_gain_shrinkage_identity_error'], shrink_error)
            assert shrink_error < 1e-10
            share = float(np.sum(mask * A) / (9*A.sum()))
            if scheme == 'original':
                options = [('original', total, bias, variance, np.zeros(2)),
                           ('original_exact_bias', bias, bias, variance, np.zeros(2)),
                           ('nine_angle_mean_original_masks', bias9, bias9, variance9, cross9)]
            else:
                np.testing.assert_allclose(bias, bias9, atol=1e-10, rtol=1e-11)
                options = [('fixed_masks_total', total, bias9, variance9, cross9),
                           ('nine_angle_mean_fixed_masks', bias9, bias9, variance9, cross9)]
            for variant, score, components_bias, component_variance, cross in options:
                rows.append(dict(wing=group, region=region, variant=variant, area_share=share,
                                 area_times_conditions=float(share*9*A.sum()),
                                 finetuned_sse=float(score[0]), scratch_sse=float(score[1]),
                                 gain=float(1-score[0]/score[1]),
                                 total_finetuned_sse=float(total[0]), total_scratch_sse=float(total[1]),
                                 bias_finetuned_sse=float(components_bias[0]), bias_scratch_sse=float(components_bias[1]),
                                 angle_variation_sse=component_variance,
                                 cross_finetuned_sse=float(cross[0]), cross_scratch_sse=float(cross[1])))
    print(f'{number}/16 wings: {group}', flush=True)

write_csv('per_wing_decomposition.csv', rows)
index = {(r['variant'], r['region'], r['wing']): r for r in rows}
wings = sorted(groups)
variants = list(dict.fromkeys(r['variant'] for r in rows))
regions = list(dict.fromkeys(r['region'] for r in rows))


def stats(values, with_test=False):
    values = np.asarray(values, float)
    assert len(values) == 16 and np.isfinite(values).all()
    mean, ci = tt.t_interval(values)
    r = dict(mean=mean, ci95=ci, positive_wings=int(np.sum(values > 0)), wings=16)
    if with_test:
        r['p_one_sided'] = tt.sign_flip_pvalue(values)
    return r


results = {}
contrast_rows = []
for variant in variants:
    result = dict(regions={}, contrasts={})
    for region in regions:
        rr = [index[(variant, region, w)] for w in wings]
        s = stats([r['gain'] for r in rr])
        s['pooled_gain'] = 1-sum(r['finetuned_sse'] for r in rr)/sum(r['scratch_sse'] for r in rr)
        s['minimum_wing_area_share'] = min(r['area_share'] for r in rr)
        s['mean_wing_area_share'] = float(np.mean([r['area_share'] for r in rr]))
        s['shared_angle_fraction_of_scratch_total'] = stats([r['angle_variation_sse']/r['total_scratch_sse'] for r in rr])
        s['shared_angle_fraction_of_finetuned_total'] = stats([r['angle_variation_sse']/r['total_finetuned_sse'] for r in rr])
        result['regions'][region] = s
    pairs = [('local_minus_global', 'local', 'global'),
             ('curved_minus_flat_all_pressures', 'curvature_2', 'curvature_0')]
    pairs += [(f'curved_minus_flat_extremity_{x}', f'cell_2_{x}', f'cell_0_{x}') for x in range(3)]
    pairs += [(f'extreme_minus_mild_curvature_{c}', f'cell_{c}_2', f'cell_{c}_0') for c in range(3)]
    for label, a, b in pairs:
        values = [index[(variant, a, w)]['gain'] - index[(variant, b, w)]['gain'] for w in wings]
        s = stats(values, True)
        result['contrasts'][label] = s
        for wing, v in zip(wings, values):
            contrast_rows.append(dict(variant=variant, contrast=label, wing=wing, difference=v))
    result['local_global_coverage_all_wings'] = all(result['regions'][r]['minimum_wing_area_share'] >= .02
                                                  for r in ('local', 'global'))
    results[variant] = result
original = json.loads((SOURCE/'attribution_results.json').read_text())
for region in ('all', 'local', 'global'):
    a, b = results['original']['regions'][region], original[region]
    np.testing.assert_allclose([a['mean'], *a['ci95']], [b['gain'], *b['ci95']], atol=1e-10, rtol=0)
np.testing.assert_allclose(results['original']['contrasts']['local_minus_global']['mean'],
                           original['local_minus_global']['difference'], atol=1e-10, rtol=0)
for c in range(3):
    for x in range(3):
        np.testing.assert_allclose(results['original']['regions'][f'cell_{c}_{x}']['pooled_gain'],
                                  original['cell_gain'][c][x], atol=1e-10, rtol=0)
checks['original_all_region_statistics_and_cells_reproduced'] = True
write_json('checks.json', checks)
write_json('results.json', dict(cases=144, wings=16, angles_per_wing=9, variants=results))
write_csv('per_wing_contrasts.csv', contrast_rows)
summary = []
for variant in variants:
    r = results[variant]
    d = r['contrasts']['local_minus_global']
    summary.append(dict(variant=variant, local_gain=r['regions']['local']['mean'],
                        global_gain=r['regions']['global']['mean'], difference=d['mean'],
                        ci95_low=d['ci95'][0], ci95_high=d['ci95'][1], positive_wings=d['positive_wings'],
                        p_one_sided=d['p_one_sided'], coverage_pass=r['local_global_coverage_all_wings']))
write_csv('summary.csv', summary)
print(json.dumps(summary, indent=2), flush=True)
