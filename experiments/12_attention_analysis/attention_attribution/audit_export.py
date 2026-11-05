"""Audit pairing, completeness and per-wing attribution coverage without excluding cases."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import json

import numpy as np
import theory_tests as tt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'real'


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False,
                               default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')


def table(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


audit = json.loads((OUT / 'export_audit.json').read_text())
expected = {c['id']: c for c in audit['cases']}
assert len(expected) == 144
assert Counter(c['group'] for c in expected.values()) == Counter(audit['group_counts'])
models = ('pretrained', 'finetuned', 'scratch')
attention_rows = []
for component in ('decoder', 'encoder'):
    folder = OUT / f'attention_{component}'
    for name in models:
        assert {p.stem for p in (folder / name).glob('*.npz')} == set(expected)
    for cid, case in expected.items():
        paired = {}
        for name in models:
            with np.load(folder / name / f'{cid}.npz') as z:
                assert str(z['group']) == case['group']
                assert np.all(z['qw'] > 0) and float(z['R']) > 0
                keys = ('dist_mean', 'bin_mass', 'base_mean', 'base_mass', 'edges', 'qw')
                s = {k: z[k] for k in keys}
                assert all(np.isfinite(s[k]).all() for k in keys if k != 'edges')
                assert np.all(s['bin_mass'] >= 0)
                np.testing.assert_allclose(s['bin_mass'].sum(-1), 1, atol=2e-6, rtol=0)
                np.testing.assert_allclose(s['base_mass'].sum(-1), 1, atol=1e-12, rtol=0)
                dimensions = (2, 4, 2048) if component == 'decoder' else (4, 4, 256)
                assert s['dist_mean'].shape == dimensions
                pairing = (str(z['query_sha256']), str(z['anchor_sha256']), float(z['R']))
                if paired:
                    assert pairing == paired['pairing']
                    for k in ('base_mean', 'base_mass', 'qw'):
                        np.testing.assert_array_equal(s[k], paired[k])
                else:
                    paired = dict(pairing=pairing, **{k: s[k] for k in ('base_mean', 'base_mass', 'qw')})
                st = tt._attn_shape_stats(s, (.25, .5, 1.0))
                attention_rows.append(dict(component=component, checkpoint=name, case=cid, group=case['group'],
                                            D_R=st['D'], baseline_D_R=st['D0'], ratio=st['ratio'],
                                            tail05=st['tail0.5'], near025=st['near'],
                                            baseline_tail05=st['tail0.5_base']))
    print(f'{component}: all 144 paired cases, dimensions, weights and probabilities valid', flush=True)
table(ROOT / 'attention_per_case.csv', attention_rows)
group_attention = defaultdict(list)
for row in attention_rows:
    group_attention[(row['component'], row['checkpoint'], row['group'])].append(row)
attention_groups = []
for (component, name, group), rows in group_attention.items():
    assert len(rows) == 9
    attention_groups.append(dict(component=component, checkpoint=name, group=group,
                                  **{k: float(np.mean([r[k] for r in rows])) for k in
                                     ('D_R', 'baseline_D_R', 'ratio', 'tail05', 'near025', 'baseline_tail05')}))
table(ROOT / 'attention_per_wing.csv', attention_groups)

assert {p.stem for p in (OUT / 'attribution').glob('*.npz')} == set(expected)
curvature = {}
group_cells = defaultdict(lambda: np.zeros((3, 3, 3)))
group_area = defaultdict(float)
case_rows = []
local = np.zeros((3, 3), bool)
local[2, :2] = True
global_ = np.zeros((3, 3), bool)
global_[0, 2] = True
all_cells = np.zeros((3, 3, 3))
first_prediction = {}
for cid, case in expected.items():
    with np.load(OUT / 'attribution' / f'{cid}.npz') as z:
        X, N, A, f, pa, pb = (z[k] for k in ('X', 'N', 'A', 'f', 'pred_finetuned', 'pred_scratch'))
        assert str(z['group']) == case['group']
        assert len(X) == 32000 and z['scored'].all()
        assert all(np.isfinite(a).all() for a in (X, N, A, f, pa, pb)) and np.all(A > 0)
        group = case['group']
        for name, p in (('finetuned', pa), ('scratch', pb)):
            # The models see no sensor and no angle, so a wing's nine predictions must be identical.
            assert np.array_equal(first_prediction.setdefault((group, name), p), p), (group, name)
        R = float(z['R'])
        assert abs(R - tt.rms_radius(X, A)) < 1e-12
        if group not in curvature:
            kappa = tt.shape_operator_norm(X, N, .25 * R) * R
            curvature[group] = kappa
        kappa = curvature[group]
        m = np.isfinite(kappa)
        w = A / A.sum()
        spread = np.sqrt(w @ (f - w @ f) ** 2)
        assert spread > 0 and m.sum() >= 30
        ext = np.abs(f - tt.weighted_quantile(f, A, [.5])[0]) / spread
        ci = np.digitize(kappa, tt.weighted_quantile(kappa[m], A[m], [1/3, 2/3]))
        xi = np.digitize(ext, tt.weighted_quantile(ext[m], A[m], [1/3, 2/3]))
        cells = np.zeros((3, 3, 3))
        for c in range(3):
            for x in range(3):
                select = m & (ci == c) & (xi == x)
                cells[c, x] = (A[select] @ ((pa[select] - f[select]) / spread) ** 2,
                               A[select] @ ((pb[select] - f[select]) / spread) ** 2, A[select].sum())
        local_share = float(cells[local, 2].sum() / A.sum())
        global_share = float(cells[global_, 2].sum() / A.sum())
        case_rows.append(dict(case=cid, group=group, local_area_fraction=local_share,
                              global_area_fraction=global_share, curvature_valid_area_fraction=float(A[m].sum()/A.sum()),
                              R1=local_share >= .02 and global_share >= .02))
        group_cells[group] += cells
        group_area[group] += float(A.sum())
        all_cells += cells
group_rows = []
for group, cells in sorted(group_cells.items()):
    row = dict(group=group)
    for label, select in (('all', np.ones((3, 3), bool)), ('local', local), ('global', global_)):
        sums = cells[select].sum(0)
        row[f'{label}_area_fraction'] = float(sums[2] / group_area[group])
        row[f'{label}_gain'] = float(1 - sums[0] / sums[1]) if sums[1] > 0 else None
    row['R1'] = row['local_area_fraction'] >= .02 and row['global_area_fraction'] >= .02
    row['local_minus_global'] = (row['local_gain'] - row['global_gain']
                                 if row['local_gain'] is not None and row['global_gain'] is not None else None)
    group_rows.append(row)
table(ROOT / 'attribution_coverage_per_case.csv', case_rows)
table(ROOT / 'attribution_per_wing.csv', group_rows)
reported = json.loads((ROOT / 'attribution_results.json').read_text())
np.testing.assert_allclose(all_cells[..., 2] / all_cells[..., 2].sum(), reported['area_share'], atol=1e-12, rtol=0)
for label in ('all', 'local', 'global'):
    gains = [r[f'{label}_gain'] for r in group_rows if r[f'{label}_gain'] is not None]
    mean, ci = tt.t_interval(gains)
    np.testing.assert_allclose([mean, *ci], [reported[label]['gain'], *reported[label]['ci95']], atol=1e-10, rtol=0)
strict_R1 = all(r['R1'] for r in group_rows)
verdict = dict(all_expected_cases_present=True, all_attention_pairing_checks_pass=True,
               native_capture_and_prediction_checks=audit['checks'],
               predictions_identical_across_nine_angles=True, cases=144, wings=16,
               curvature_resolution_R=.25,
               minimum_valid_curvature_area=min(r['curvature_valid_area_fraction'] for r in case_rows),
               per_case_R1_pass_count=sum(r['R1'] for r in case_rows),
               per_wing_R1_pass_count=sum(r['R1'] for r in group_rows),
               minimum_case_local_area=min(r['local_area_fraction'] for r in case_rows),
               minimum_case_global_area=min(r['global_area_fraction'] for r in case_rows),
               minimum_wing_local_area=min(r['local_area_fraction'] for r in group_rows),
               minimum_wing_global_area=min(r['global_area_fraction'] for r in group_rows),
               strict_R1_all_wings_pass=strict_R1,
               strict_R2=(reported['local_minus_global']['verdict'] == 'local') if strict_R1 else None,
               independent_attribution_recomputation_matches=True)
dump(ROOT / 'integrity_and_coverage.json', verdict)
print(json.dumps(verdict, indent=2), flush=True)
