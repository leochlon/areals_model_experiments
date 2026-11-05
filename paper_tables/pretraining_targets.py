"""Section 4.3, Table 1 and Appendix M: matched pretraining targets on WindsorML.

Rebuilds Table 1, Tables M.1-M.3 and the numbers quoted in Section 4.3 and Appendix M
(cohort, results, patch pilot, coefficients inside AREALS, pretraining seeds, scope) from

    results/pretraining_targets/{matched,checkpoint_sensitivity,patch_pilot,
                                 coefficient_prior,coefficient_replicates}/
    results/pretraining_targets_extra/   (cohort summary, seed-0 tuning fits, the 150 fit configs,
                                          coefficient-prior run records, patch-pilot receipts and split)

Two bootstrap schemes appear in the paper.

* Table M.2 (pretrained minus scratch, every arm): the intervals are the ones stored in
  matched/comparison.json, written by field_study.py ``report``. That function uses one
  ``numpy.random.default_rng(2091)`` shared across all 15 arm-budget cells (order patch,
  areals, hessian) and stores scratch minus pretrained; the paper prints the negated interval.

* AREALS minus Hessian (Appendix M, Results) and Table M.3: no stored file. They are computed
  here with a fresh ``numpy.random.default_rng(2091)`` for every budget and 5,000 paired
  bootstrap draws. With three fine-tuning seeds (AREALS minus Hessian, Table M.3 left) each
  draw resamples seeds, then cars: ``delta[rng.integers(0, 3, 3)][:, rng.integers(0, 35, 35)].mean()``,
  where ``delta`` is a (seed, car) array of per-car test nRMSE differences (first model minus
  second). Table M.3 right has one fine-tuning seed, so each draw resamples cars only,
  ``d[rng.integers(0, 35, 35)].mean()``, with ``d`` the per-car mean over the three coefficient
  pretraining seeds minus the per-car mean over the two distance-and-normal-line seeds.
  The interval is the 2.5% and 97.5% quantiles of the draws.
"""
from __future__ import annotations

import numpy as np

from paper_tables._core import RESULTS, Table, as_printed, fmt, load_json

BUDGETS = [18, 35, 69, 137, 274]
ARMS = ['areals', 'hessian', 'patch']
ARM_LABEL = {'areals': 'AREALS (d, nn⊤)', 'hessian': 'Hessian (d, nn⊤, Hr)', 'patch': 'Patch (A)'}
BASE = 'pretraining_targets'
EXTRA = 'pretraining_targets_extra'


# ---------------------------------------------------------------- helpers

def signed(value: float, decimals: int = 4) -> str:
    return f'{value:+.{decimals}f}'.replace('-', '−')


def interval(lo: float, hi: float, decimals: int = 4) -> str:
    return f'[{signed(lo, decimals)}, {signed(hi, decimals)}]'


def bootstrap_seeds_cars(delta: np.ndarray, draws: int = 5000, seed: int = 2091) -> np.ndarray:
    """Paired bootstrap over fine-tuning seeds, then cars, with a fresh generator (see module docstring)."""
    rng = np.random.default_rng(seed)
    n_seeds, n_cars = delta.shape
    values = [delta[rng.integers(0, n_seeds, n_seeds)][:, rng.integers(0, n_cars, n_cars)].mean() for _ in range(draws)]
    return np.quantile(values, [.025, .975])


def bootstrap_cars(delta: np.ndarray, draws: int = 5000, seed: int = 2091) -> np.ndarray:
    """Bootstrap over cars only, with a fresh generator (see module docstring)."""
    rng = np.random.default_rng(seed)
    n_cars = len(delta)
    values = [delta[rng.integers(0, n_cars, n_cars)].mean() for _ in range(draws)]
    return np.quantile(values, [.025, .975])


def car_values(metrics: dict, ids: list[str]) -> np.ndarray:
    by_id = {c['id']: c['nrmse'] for c in metrics['cars']}
    assert sorted(by_id) == sorted(ids)
    return np.array([by_id[i] for i in ids])


# ---------------------------------------------------------------- loaders

def matched_test():
    """Per-car test nRMSE of the 90 locked runs: values[arm, init, budget] -> (3 seeds, 35 cars)."""
    rows = [load_json(BASE, 'matched', 'test', f'{i:03d}.json') for i in range(90)]
    ids = [c['id'] for c in rows[0]['metrics']['cars']]
    per = {}
    for r in rows:
        row = r['row']
        per[row['arm'], row['initialization'], row['budget'], row['seed']] = car_values(r['metrics'], ids)
    values = {(a, k, b): np.stack([per[a, k, b, s] for s in range(3)])
              for a in ARMS for k in ['pretrained', 'scratch'] for b in BUDGETS}
    return values, ids


def comparison():
    doc = load_json(BASE, 'matched', 'comparison.json')
    return doc, {(c['arm'], c['budget']): c for c in doc['comparisons']}


def validation_means():
    """Three-seed validation means: seed 0 from the selected tuning fit, seeds 1 and 2 from their own fits."""
    winners = load_json(BASE, 'matched', 'selections.json')['winners']
    out = {}
    for arm in ARMS:
        for init in ['pretrained', 'scratch']:
            for b in BUDGETS:
                seeds = [winners[f'{arm}/{init}/{b}']['validation']]
                seeds += [load_json(BASE, 'matched', 'validation', arm, init, f'n{b}', f'seed{s}-selected',
                                    'best_validation.json')['mean_nrmse'] for s in (1, 2)]
                out[arm, init, b] = float(np.mean(seeds))
    return out


def fit_configs() -> list[dict]:
    """config.json of all 150 matched fine-tuning fits (90 seed-0 tuning fits, 60 follow-ups)."""
    paths = sorted((RESULTS / EXTRA / 'matched_fit_configs').glob('*/*/n*/*/config.json'))
    return [load_json(*p.relative_to(RESULTS).parts) for p in paths]


def uniform(rows: list[dict], key: str):
    """The single value of `key` shared by every row (asserts that it is shared)."""
    values = {r[key] for r in rows}
    assert len(values) == 1, (key, values)
    return values.pop()


def coefficient_prior():
    """Coefficient prior (three fine-tuning seeds): per-car (3, 35) arrays for pretrained and scratch."""
    out = {}
    for b in BUDGETS:
        docs = [load_json(BASE, 'coefficient_prior', 'per_car', f'n{b}', f'seed{s}', 'test_metrics.json') for s in range(3)]
        ids = [c['id'] for c in docs[0]['metrics']['pretrained']['cars']]
        out[b] = {k: np.stack([car_values(d['metrics'][k], ids) for d in docs]) for k in ['pretrained', 'scratch']}
        out[b]['ids'] = ids
        out[b]['scratch_seed0_mean'] = docs[0]['metrics']['scratch']['mean_nrmse']
    return out


def replicates(prior):
    """Fine-tuning seed 0: per-car arrays for the coefficient prior (seeds 17/29/43) and d, nn⊤ alone (17/29)."""
    out = {}
    for b in BUDGETS:
        ids = prior[b]['ids']
        seed17 = prior[b]['pretrained'][0]
        coef = [seed17] + [car_values(load_json(BASE, 'coefficient_replicates', 'per_car', 'patch_query', f'seed{s}',
                                                f'n{b}', 'seed0', 'early_test_metrics.json')['metrics'], ids)
                           for s in (29, 43)]
        dnn_docs = [load_json(BASE, 'coefficient_replicates', 'per_car', 'areals', f'seed{s}', f'n{b}', 'seed0',
                              'early_test_metrics.json') for s in (17, 29)]
        for doc, s in zip(dnn_docs, (17, 29)):
            assert doc['cell']['pretraining_seed'] == s and doc['cell']['seed'] == 0 and doc['cell']['budget'] == b
        out[b] = dict(coef=np.stack(coef), dnn=np.stack([car_values(d['metrics'], ids) for d in dnn_docs]))
    return out


# ---------------------------------------------------------------- tables

def table_1(comp) -> Table:
    t = Table('Table 1', 'Matched pretraining targets on WindsorML: test nRMSE, pretrained / from scratch, '
              'means of three seeds over the 35 test cars',
              ['Cars', 'd, nn⊤ (AREALS)', 'd, nn⊤, Hessian', 'Patch A'])
    paper = {
        'areals': [(0.277, 0.443), (0.228, 0.381), (0.193, 0.265), (0.178, 0.228), (0.158, 0.203)],
        'hessian': [(0.301, 0.475), (0.246, 0.388), (0.206, 0.301), (0.179, 0.215), (0.161, 0.182)],
        'patch': [(0.457, 0.442), (0.384, 0.379), (0.307, 0.273), (0.257, 0.223), (0.221, 0.196)],
    }
    for j, b in enumerate(BUDGETS):
        row = [str(b)]
        for arm in ARMS:
            c = comp[arm, b]
            p = t.check(f'{arm} {b} pretrained', c['pretrained_mean'], paper[arm][j][0], 3)
            s = t.check(f'{arm} {b} scratch', c['scratch_mean'], paper[arm][j][1], 3)
            row.append(f'{fmt(p, 3)} / {fmt(s, 3)}')
        t.rows.append(row)
    return t


def table_m1() -> Table:
    t = Table('Table M.1', 'The three priors: recipe and geometry on the 92 test meshes (per-object means)',
              ['Arm', 'Parameters', 'Targets', 'Peak rate', 'Distance or position', 'Normal line', 'Other'])
    geo = {a: load_json(BASE, 'matched', f'geometry_{a}.json')['splits']['test'] for a in ARMS}
    cfg = {a: load_json(BASE, 'matched', f'pretraining_config_{a}.json') for a in ARMS}
    paper_params = {'areals': 3215503, 'hessian': 3217948, 'patch': 2268298}
    paper_peak = {'areals': 3, 'hessian': 3, 'patch': 1}  # units of 1e-4
    for a in ARMS:
        t.check(f'{a} test meshes', geo[a]['objects'], 92, 0)
        params = t.check(f'{a} parameters', cfg[a]['parameters'], paper_params[a], 0)
        peak = t.check(f'{a} peak rate (x1e-4)', cfg[a]['peak_lr'] / 1e-4, paper_peak[a], 0)
        m = geo[a]['mean']
        if a == 'areals':
            dist = f"distance nRMSE {fmt(t.check('areals distance nRMSE', m['distance_nrmse'], 0.0089, 4), 4)}"
            normal = t.check('areals normal line (deg)', m['normal_degrees'], 11.0, 1)
            other = '—'
        elif a == 'hessian':
            dist = f"distance nRMSE {fmt(t.check('hessian distance nRMSE', m['distance_nrmse'], 0.0183, 4), 4)}"
            normal = t.check('hessian normal line (deg)', m['normal_degrees'], 15.0, 1)
            h = t.check('hessian H error', m['H'], 0.495, 3)
            floor = t.check('hessian H label floor', m['H_label_floor'], 0.007, 3)
            other = f'H error {fmt(h, 3)} (label floor {fmt(floor, 3)})'
        else:
            dist = f"position RMS {fmt(t.check('patch position RMS', m['position_nrmse'], 0.0013, 4), 4)}"
            normal = t.check('patch normal line (deg)', m['normal_degrees'], 1.6, 1)
            other = f"relative A RMS {fmt(t.check('patch relative A RMS', m['relative_A_rms'], 0.019, 3), 3)}"
            t.check('patch: sampled patches per object (caption)', cfg[a]['patches_per_object'], 8, 0)
        t.rows.append([a, f'{params:,}', cfg[a]['targets'], f'{peak:g} × 10⁻⁴', dist, f'{normal:.1f}°', other])
    t.notes.append('Weight decay (0.05, 0.05, 0) is set inside pretrain.py and is not written to any result file, '
                   'so that column is not rebuilt. The same holds for the 200-update warm-up and gradient clipping '
                   'at 1 quoted in Appendix M.')
    return t


def table_m2(comp, val) -> Table:
    t = Table('Table M.2', 'Matched pretraining targets, every comparison: test nRMSE over the 35 test cars '
              '(three fine-tuning seeds), difference pretrained minus scratch with a 95% paired bootstrap interval, '
              'and validation means',
              ['Arm', 'Cars', 'Pretrained', 'Scratch', 'Gain', 'Difference', '95% interval', 'Validation pretrained / scratch'])
    paper = {
        'areals': [(0.2773, 0.4431, 37.4, -0.1658, -0.1929, -0.1447, 0.2649, 0.4413),
                   (0.2281, 0.3809, 40.1, -0.1528, -0.1817, -0.1266, 0.2114, 0.3602),
                   (0.1934, 0.2649, 27.0, -0.0715, -0.0949, -0.0512, 0.1769, 0.2649),
                   (0.1778, 0.2279, 22.0, -0.0501, -0.0733, -0.0324, 0.1506, 0.2037),
                   (0.1582, 0.2026, 21.9, -0.0443, -0.0661, -0.0247, 0.1410, 0.1837)],
        'hessian': [(0.3011, 0.4753, 36.7, -0.1742, -0.2076, -0.1401, 0.2879, 0.4721),
                    (0.2465, 0.3882, 36.5, -0.1417, -0.1817, -0.1077, 0.2331, 0.3727),
                    (0.2061, 0.3006, 31.4, -0.0945, -0.1260, -0.0644, 0.1892, 0.2950),
                    (0.1786, 0.2147, 16.9, -0.0362, -0.0489, -0.0230, 0.1606, 0.2079),
                    (0.1607, 0.1822, 11.8, -0.0215, -0.0327, -0.0103, 0.1452, 0.1735)],
        'patch': [(0.4575, 0.4419, -3.5, 0.0155, -0.0218, 0.0614, 0.4401, 0.4231),
                  (0.3838, 0.3789, -1.3, 0.0049, -0.0456, 0.0521, 0.3748, 0.3649),
                  (0.3068, 0.2730, -12.4, 0.0337, 0.0119, 0.0539, 0.3079, 0.2735),
                  (0.2568, 0.2233, -15.0, 0.0335, 0.0097, 0.0575, 0.2450, 0.2257),
                  (0.2212, 0.1956, -13.1, 0.0256, 0.0077, 0.0474, 0.2174, 0.1894)],
    }
    for arm in ARMS:
        for j, b in enumerate(BUDGETS):
            c, p = comp[arm, b], paper[arm][j]
            tag = f'{arm} {b}'
            pre = t.check(f'{tag} pretrained', c['pretrained_mean'], p[0], 4)
            scr = t.check(f'{tag} scratch', c['scratch_mean'], p[1], 4)
            gain = t.check(f'{tag} gain %', 100 * (1 - c['pretrained_mean'] / c['scratch_mean']), p[2], 1)
            diff = t.check(f'{tag} difference', -c['gain_nrmse'], p[3], 4)
            lo = t.check(f'{tag} interval low', -c['gain_paired_bootstrap_95'][1], p[4], 4)
            hi = t.check(f'{tag} interval high', -c['gain_paired_bootstrap_95'][0], p[5], 4)
            vp = t.check(f'{tag} validation pretrained', val[arm, 'pretrained', b], p[6], 4)
            vs = t.check(f'{tag} validation scratch', val[arm, 'scratch', b], p[7], 4)
            t.rows.append([ARM_LABEL[arm] if j == 0 else '', str(b), fmt(pre, 4), fmt(scr, 4),
                           f'{gain:.1f}%'.replace('-', '−'),
                           signed(diff), interval(lo, hi), f'{fmt(vp, 4)} / {fmt(vs, 4)}'])
    t.notes.append('Intervals are those stored in matched/comparison.json (one generator shared across cells), negated '
                   'to pretrained minus scratch.')
    return t


def table_m3(prior, reps) -> tuple[Table, dict]:
    t = Table('Table M.3', 'Coefficient targets inside AREALS: test nRMSE on the 35 test cars. Left: coefficient prior '
              'vs scratch (three fine-tuning seeds). Right: fine-tuning seed 0, coefficients (pretraining seeds 17/29/43) '
              'vs d, nn⊤ alone (17/29)',
              ['Cars', 'Coefficient prior', 'Scratch', 'Difference [95% interval]',
               'Coefficients', 'd, nn⊤ alone', 'Difference [95% interval]'])
    op = {r['training_cars']: r for r in load_json(BASE, 'coefficient_prior', 'operator32_comparison.json')['results']}
    pq = {r['training_cars']: r for r in load_json(BASE, 'coefficient_replicates', 'patch_query_windsor.json')['results']}
    paper_left = [(0.2789, 0.4359, -0.1569, -0.1812, -0.1331), (0.2304, 0.3729, -0.1424, -0.1645, -0.1213),
                  (0.1896, 0.2649, -0.0753, -0.0963, -0.0572), (0.1690, 0.2279, -0.0589, -0.0759, -0.0439),
                  (0.1565, 0.1915, -0.0350, -0.0487, -0.0216)]
    paper_right = [(0.2637, 0.2648, -0.0010, -0.0063, 0.0041), (0.2222, 0.2248, -0.0027, -0.0090, 0.0041),
                   (0.1874, 0.1876, -0.0002, -0.0080, 0.0062), (0.1730, 0.1688, 0.0043, -0.0021, 0.0112),
                   (0.1552, 0.1482, 0.0069, -0.0004, 0.0168)]
    stats = {}
    for j, b in enumerate(BUDGETS):
        pl, pr = paper_left[j], paper_right[j]
        pre, scr = prior[b]['pretrained'], prior[b]['scratch']
        assert abs(pre.mean() - op[b]['pretrained_mean_nrmse']) < 1e-12 and abs(scr.mean() - op[b]['scratch_mean_nrmse']) < 1e-12
        cp = t.check(f'left {b} coefficient prior', op[b]['pretrained_mean_nrmse'], pl[0], 4)
        sc = t.check(f'left {b} scratch', op[b]['scratch_mean_nrmse'], pl[1], 4)
        delta = pre - scr
        q = bootstrap_seeds_cars(delta)
        dl = t.check(f'left {b} difference', delta.mean(), pl[2], 4)
        ll = t.check(f'left {b} interval low', q[0], pl[3], 4)
        lh = t.check(f'left {b} interval high', q[1], pl[4], 4)

        coef_seed_means = reps[b]['coef'].mean(1)
        assert abs(coef_seed_means.mean() - pq[b]['mean_test_nrmse']) < 1e-12
        cr = t.check(f'right {b} coefficients', pq[b]['mean_test_nrmse'], pr[0], 4)
        dn = t.check(f'right {b} d, nn alone', reps[b]['dnn'].mean(), pr[1], 4)
        d = reps[b]['coef'].mean(0) - reps[b]['dnn'].mean(0)
        qr = bootstrap_cars(d)
        dr = t.check(f'right {b} difference', d.mean(), pr[2], 4)
        rl = t.check(f'right {b} interval low', qr[0], pr[3], 4)
        rh = t.check(f'right {b} interval high', qr[1], pr[4], 4)
        stats[b] = dict(left=(dl, ll, lh), right=(dr, rl, rh), dnn=dn, coef=cr, pretrained=cp, scratch=sc,
                        std=pq[b]['std_test_nrmse'], relative=op[b]['relative_error_change'],
                        scratch_seed0=prior[b]['scratch_seed0_mean'])
        t.rows.append([str(b), fmt(cp, 4), fmt(sc, 4), f'{signed(dl)} {interval(ll, lh)}',
                       fmt(cr, 4), fmt(dn, 4), f'{signed(dr)} {interval(rl, rh)}'])
    t.notes.append('Differences and intervals are computed here (fresh default_rng(2091) per budget, 5,000 draws); '
                   'see the module docstring.')
    return t, stats


def text_numbers(comp_doc, comp, values, m3) -> list[Table]:
    cols = ['Claim', 'Rebuilt', 'Paper']

    # ---- Section 4.3 (and the Section 4.2 sentence on the 1,668-mesh prior)
    s = Table('Section 4.3', 'Numbers quoted in the text (Section 4.2 sentence on the matched prior included)', cols)
    thresholds = comp_doc['smallest_measured_budget']
    cohort = load_json(EXTRA, 'matched_cohort.json')
    v = s.check('Section 4.2: 1,668-mesh prior reaches 0.25 with N cars', thresholds['areals/pretrained'], 35, 0)
    s.rows.append(['1,668-mesh prior reaches nRMSE 0.25 on WindsorML with (cars)', f'{v}', '35'])
    v = s.check('matched cohort training meshes', cohort['counts']['train'], 1668, 0)
    s.rows.append(['Pretraining meshes', f'{v:,}', '1,668'])
    gains = [100 * (1 - comp['areals', b]['pretrained_mean'] / comp['areals', b]['scratch_mean']) for b in BUDGETS]
    lo, hi = s.check('AREALS gain min %', min(gains), 22, 0), s.check('AREALS gain max %', max(gains), 40, 0)
    s.rows.append(['Distance and normal line cut the error by (%)', f'{lo:.1f}–{hi:.1f}', '22–40'])
    better = [int((values['areals', 'pretrained', b].mean(0) < values['areals', 'scratch', b].mean(0)).sum()) for b in BUDGETS]
    v = s.check('AREALS better than scratch on at least N of 35 cars', min(better), 31, 0)
    s.rows.append(['... on at least N of the 35 cars (per budget: ' + '/'.join(map(str, better)) + ')', str(v), '31'])
    v = s.check('budgets where Hessian prior is more accurate than AREALS',
                sum(comp['hessian', b]['pretrained_mean'] < comp['areals', b]['pretrained_mean'] for b in BUDGETS), 0, 0)
    s.rows.append(['Budgets where the Hessian prior beats AREALS ("never")', str(v), '0'])
    a_minus_h = {b: values['areals', 'pretrained', b] - values['hessian', 'pretrained', b] for b in BUDGETS}
    a_minus_h_q = {b: bootstrap_seeds_cars(a_minus_h[b]) for b in BUDGETS}
    v = s.check('smallest three budgets where the AREALS-minus-Hessian interval lies below zero',
                sum(a_minus_h_q[b][1] < 0 for b in BUDGETS[:3]), 3, 0)
    s.rows.append(['Hessian less accurate at the three smallest budgets (AREALS-minus-Hessian interval below zero)', str(v), '3'])
    geo_patch = load_json(BASE, 'matched', 'geometry_patch.json')['splits']['test']['mean']
    v = s.check('patch relative A RMS %', 100 * geo_patch['relative_A_rms'], 1.9, 1)
    w = s.check('patch normals (deg)', geo_patch['normal_degrees'], 1.6, 1)
    s.rows.append(['Patch encoder reproduces coefficients (%) / normals (°)', f'{v:.2f} / {w:.2f}', '1.9 / 1.6'])
    v = s.check('budgets where patch prior is worse than scratch',
                sum(comp['patch', b]['pretrained_mean'] > comp['patch', b]['scratch_mean'] for b in BUDGETS), 5, 0)
    s.rows.append(['Budgets where the patch prior transfers worse than scratch', str(v), '5 (every budget)'])
    v = s.check('patch budgets significantly worse (largest three)',
                sum(comp['patch', b]['gain_paired_bootstrap_95'][1] < 0 for b in BUDGETS[2:]), 3, 0)
    s.rows.append(['... significantly at the three largest (interval excludes zero)', str(v), '3'])
    for key, paper in [('patch/pretrained', 274), ('areals/scratch', 137), ('hessian/scratch', 137),
                       ('patch/scratch', 137), ('areals/pretrained', 35), ('hessian/pretrained', 35)]:
        v = s.check(f'cars to reach 0.25: {key}', thresholds[key], paper, 0)
        s.rows.append([f'Cars to reach nRMSE 0.25, {key}', str(v), str(paper)])
    op_cohort = load_json(BASE, 'coefficient_prior', 'operator32_comparison.json')['geometry_cohort']
    v = s.check('coefficient prior training meshes', op_cohort['train'], 14304, 0)
    s.rows.append(['Coefficient prior trained on (meshes)', f'{v:,}', '14,304'])
    rel = [-100 * m3[b]['relative'] for b in BUDGETS]
    lo, hi = s.check('coefficient prior below scratch min %', min(rel), 18, 0), s.check('coefficient prior below scratch max %', max(rel), 38, 0)
    s.rows.append(['Coefficient prior below scratch (%)', f'{lo:.1f}–{hi:.1f}', '18–38'])
    gap = max(abs(m3[b]['right'][0]) for b in BUDGETS)
    v = s.check('within N nRMSE of d, nn alone', gap, 0.007, 3)
    s.rows.append(['Within N nRMSE of the same network without the coefficients', fmt(v, 4), '0.007'])

    # ---- Appendix M
    m = Table('Appendix M', 'Numbers quoted in the Appendix M text (cohort, results, patch pilot, coefficients, '
              'pretraining seeds, scope)', cols)
    for key, label, paper in [('processed', 'processed', 5287), ('patch_ok', 'passed patch preparation', 2123),
                              ('query_ok', 'passed query preparation', 4529)]:
        v = m.check(f'cohort {key}', cohort[key], paper, 0)
        m.rows.append([f'Snapshot: objects {label}', f'{v:,}', f'{paper:,}'])
    for split, paper in [('train', 1668), ('val', 85), ('test', 92)]:
        v = m.check(f'cohort {split}', cohort['counts'][split], paper, 0)
        m.rows.append([f'Cohort {split} meshes', f'{v:,}', f'{paper:,}'])
    for a in ARMS:
        cfg = load_json(BASE, 'matched', f'pretraining_config_{a}.json')
        steps = m.check(f'{a} updates', cfg['steps'], 10626, 0)
        batch = m.check(f'{a} global batch', cfg['global_object_batch'], 64, 0)
        pres = m.check(f'{a} presentations', cfg['steps'] * cfg['global_object_batch'], 680064, 0)
        seed = m.check(f'{a} seed', cfg['seed'], 17, 0)
        gpus = m.check(f'{a} GPUs', cfg['world_size'], 4, 0)
        final = m.check(f'{a} final rate % of peak', 100 * cfg['final_lr_ratio'], 5, 0)
        m.rows.append([f'{a}: updates / batch / presentations / seed / GPUs / final rate (% of peak)',
                       f'{steps:,} / {batch} / {pres:,} / {seed} / {gpus} / {final:g}',
                       '10,626 / 64 / 680,064 / 17 / 4 / 5'])

    # fine-tuning and testing: the recipe recorded in all 150 fit configs
    configs = fit_configs()
    for c in configs:
        assert len(c['normalization']['training_ids']) == c['budget']
    assert uniform(configs, 'optimizer') == 'AdamW' and uniform(configs, 'precision') == 'fp32'
    assert uniform(configs, 'schedule') == 'constant' and uniform(configs, 'complete_geometry') is True
    w = m.check('WindsorML training cars (largest budget)', max(len(c['normalization']['training_ids']) for c in configs), 274, 0)
    m.rows.append(['WindsorML training cars', str(w), '274'])
    val_cars = {len(load_json(*p.relative_to(RESULTS).parts)['cars']) for p in
                list((RESULTS / EXTRA / 'matched_tuning_validation').glob('*/*/n*/seed0-lr*/best_validation.json'))
                + list((RESULTS / BASE / 'matched' / 'validation').glob('*/*/n*/seed*-selected/best_validation.json'))}
    assert len(val_cars) == 1
    v = m.check('WindsorML validation cars', val_cars.pop(), 35, 0)
    m.rows.append(['WindsorML validation cars', str(v), '35'])
    budgets = sorted({c['budget'] for c in configs})
    for got, paper in zip(budgets, BUDGETS):
        m.check(f'nested budget {paper}', got, paper, 0)
    assert len(budgets) == len(BUDGETS)
    m.rows.append(['Nested training budgets (cars)', ', '.join(map(str, budgets)), '18, 35, 69, 137, 274'])
    sens = m.check('sensors', uniform(configs, 'sensors'), 0, 0)
    upd = m.check('fine-tuning updates', uniform(configs, 'updates'), 8000, 0)
    cars = m.check('cars per update', uniform(configs, 'batch'), 8, 0)
    qry = m.check('queries per car', uniform(configs, 'queries'), 2048, 0)
    wd = m.check('fine-tuning weight decay', uniform(configs, 'weight_decay'), 0, 0)
    new = m.check('new-parameter rate (x1e-4)', uniform(configs, 'new_lr') / 1e-4, 2, 0)
    every = m.check('validation every N updates', uniform(configs, 'validation_every'), 250, 0)
    m.rows.append(['Sensors / updates / cars per update / queries per car / weight decay / new-parameter rate (× 10⁻⁴) '
                   '/ validation every', f'{sens} / {upd:,} / {cars} / {qry:,} / {wd:g} / {new:g} / {every}',
                   '0 / 8,000 / 8 / 2,048 / 0 / 2 / 250'])
    winners = load_json(BASE, 'matched', 'selections.json')['winners']
    tuning_cfg = [c for c in configs if c['name'].split('/')[-1].startswith('seed0-lr')]
    followup_cfg = [c for c in configs if c['name'].endswith('-selected')]
    grid = sorted({c['lr'] for c in tuning_cfg})
    assert len(tuning_cfg) == len(grid) * len(winners) and len(followup_cfg) + len(tuning_cfg) == len(configs)
    for got, (paper, unit) in zip(grid, [(2, 1e-5), (2, 1e-4), (1, 1e-3)]):
        m.check(f'encoder rate grid ({unit:g})', got / unit, paper, 0)
    assert len(grid) == 3
    m.rows.append(['Encoder learning-rate grid', ', '.join(f'{g:g}' for g in grid), '2e-05, 0.0002, 0.001'])
    v = m.check('rate chosen on seed', uniform(tuning_cfg, 'seed'), 0, 0)
    repeat = sorted({c['seed'] for c in followup_cfg})
    for got, paper in zip(repeat, [1, 2]):
        m.check(f'seed {paper} repeats the chosen rate', got, paper, 0)
    assert len(repeat) == 2 and len(followup_cfg) == len(repeat) * len(winners)
    assert all(c['lr'] == winners[f"{c['arm']}/{c['initialization']}/{c['budget']}"]['lr'] for c in followup_cfg)
    m.rows.append(['Rate chosen on seed / seeds repeating the chosen rate (rates verified equal)',
                   f"{v} / {', '.join(map(str, repeat))}", '0 / 1, 2'])
    tuning = list((RESULTS / EXTRA / 'matched_tuning_validation').glob('*/*/n*/seed0-lr*/best_validation.json'))
    followup = list((RESULTS / BASE / 'matched' / 'validation').glob('*/*/n*/seed*-selected/best_validation.json'))
    v = m.check('fits completed', len(tuning) + len(followup), 150, 0)
    m.rows.append(['Fits completed', str(v), '150'])
    v = m.check('locked runs', len(load_json(BASE, 'matched', 'test_lock.json')['runs']), 90, 0)
    m.rows.append(['Selected runs locked before testing', str(v), '90'])
    v = m.check('test cars', len(values['areals', 'pretrained', 18][0]), 35, 0)
    m.rows.append(['Test cars', str(v), '35'])
    v = m.check('AREALS gains excluding zero', sum(comp['areals', b]['gain_paired_bootstrap_95'][0] > 0 for b in BUDGETS), 5, 0)
    w = m.check('Hessian gains excluding zero', sum(comp['hessian', b]['gain_paired_bootstrap_95'][0] > 0 for b in BUDGETS), 5, 0)
    m.rows.append(['Budgets where AREALS / Hessian gains exclude zero', f'{v} / {w}', '5 / 5'])
    v = m.check('patch intervals excluding zero (largest three)',
                sum(comp['patch', b]['gain_paired_bootstrap_95'][1] < 0 for b in BUDGETS), 3, 0)
    m.rows.append(['Budgets where patch intervals exclude zero (the three largest)', str(v), '3'])
    paper_diff = [0.0238, 0.0184, 0.0127, 0.0008, 0.0025]
    for b, p in zip(BUDGETS, paper_diff):
        v = m.check(f'AREALS lower than Hessian by, {b} cars', -a_minus_h[b].mean(), p, 4)
        m.rows.append([f'AREALS minus Hessian, {b} cars: difference {interval(*a_minus_h_q[b])}', signed(-v), f'−{p:.4f}'])
    v = m.check('AREALS-Hessian intervals excluding zero (18-69)', sum(a_minus_h_q[b][1] < 0 for b in BUDGETS), 3, 0)
    m.rows.append(['AREALS-minus-Hessian intervals excluding zero (18, 35, 69 cars)', str(v), '3'])
    for b, (plo, phi) in [(137, (-0.0104, 0.0103)), (274, (-0.0207, 0.0135))]:
        lo = m.check(f'AREALS-Hessian interval low {b}', a_minus_h_q[b][0], plo, 4)
        hi = m.check(f'AREALS-Hessian interval high {b}', a_minus_h_q[b][1], phi, 4)
        m.rows.append([f'AREALS-minus-Hessian interval, {b} cars', interval(lo, hi), interval(plo, phi)])

    # patch pilot
    pilot = load_json(BASE, 'patch_pilot', 'summary.json')
    receipt = load_json(EXTRA, 'patch_pilot', 'export_receipt.json')
    complete = load_json(EXTRA, 'patch_pilot', 'pretraining_complete.json')
    v = m.check('pilot pretraining meshes', receipt['training_meshes'], 385, 0)
    w = m.check('pilot updates', complete['step'], 10626, 0)
    x = m.check('pilot seed', complete['seed'], 17, 0)
    m.rows.append(['Patch pilot: meshes / updates / seed', f'{v} / {w:,} / {x}', '385 / 10,626 / 17'])
    v = m.check('pilot pressure seeds', len(pilot['per_seed']), 1, 0)
    m.rows.append(['Patch pilot: pressure seeds', str(v), '1'])
    pilot_cases = load_json(EXTRA, 'patch_pilot', 'windsor_cases.json')['cases']
    v = m.check('pilot training cars', sum(c['split'] == 'train' for c in pilot_cases), 274, 0)
    w = m.check('pilot test cars', len(pilot['per_seed'][0]['per_case']), 35, 0)
    assert sum(c['split'] == 'test' for c in pilot_cases) == w
    m.rows.append(['Patch pilot: training cars / test cars', f'{v} / {w}', '274 / 35'])
    pilot_protocol = load_json(EXTRA, 'patch_pilot', 'pressure_protocol.json')
    pilot_grid = sorted(g['encoder_lr'] for g in pilot_protocol['grid'])
    assert pilot_grid == grid and pilot_protocol['seeds'] == [0]
    m.rows.append(['Patch pilot: encoder learning-rate grid (same as the full comparison)',
                   ', '.join(f'{g:g}' for g in pilot_grid), 'same grid'])
    for init in ['pretrained', 'scratch']:
        v = m.check(f'pilot {init} encoder rate (x1e-5)', pilot['selected_rates'][init]['encoder_lr'] / 1e-5, 2, 0)
        m.rows.append([f'Patch pilot: {init} chose encoder rate (× 10⁻⁵)', f'{v:g}', '2'])
    comparison_seed0 = pilot['per_seed'][0]
    pre = m.check('pilot pretrained', comparison_seed0['pretrained_case_mean_nrmse'], 0.2176, 4)
    scr = m.check('pilot scratch', comparison_seed0['scratch_case_mean_nrmse'], 0.1917, 4)
    m.rows.append(['Patch pilot: pretrained vs scratch (35 test cars)', f'{fmt(pre, 4)} vs {fmt(scr, 4)}', '0.2176 vs 0.1917'])
    v = m.check('pilot worse %', 100 * (pre / scr - 1), 13.5, 1)
    w = m.check('full comparison worse at 274 %', 100 * (comp['patch', 274]['pretrained_mean'] / comp['patch', 274]['scratch_mean'] - 1), 13.1, 1)
    m.rows.append(['Patch pilot worse than scratch (%) / full comparison at 274 cars (%)', f'{v:.2f} / {w:.2f}', '13.5 / 13.1'])

    # coefficients inside AREALS
    for split, paper in [('train', 14304), ('val', 637), ('test', 632)]:
        v = m.check(f'coefficient cohort {split}', op_cohort[split], paper, 0)
        m.rows.append([f'Coefficient prior cohort, {split}', f'{v:,}', f'{paper:,}'])
    history = load_json(BASE, 'coefficient_prior', 'operator32_comparison.json')['geometry_allocation_history']
    v = m.check('coefficient prior updates', history[-1]['through_step'], 10626, 0)
    m.rows.append(['Coefficient prior updates', f'{v:,}', '10,626'])
    runs = {name: load_json(EXTRA, 'coefficient_pretraining', name, 'run.json')
            for name in ['operator32', 'operator_seed29', 'operator_seed43', 'dnn_alone_seed17', 'dnn_alone_seed29']}
    for name, r in runs.items():
        assert r['mode'] == ('baseline' if name.startswith('dnn') else 'operator') and r['amp'] == 'fp32'
        assert r['config']['data']['augment_rotation'] == 'none' and r['train_provenance']['rotation'] == 'none'
        assert r['train_provenance']['cohort_counts'] == op_cohort and r['steps'] == history[-1]['through_step']
    op = runs['operator32']
    aux = m.check('auxiliary chart-loss weight', op['aux_weight'], 0.1, 1)
    seed = m.check('coefficient prior seed', op['seed'], 17, 0)
    steps = m.check('coefficient prior updates (run record)', op['steps'], 10626, 0)
    batch = m.check('coefficient prior global batch', op['batch_size'], 64, 0)
    mask = m.check('coefficient prior masking fraction', op['config']['data']['mask_fraction'], 0, 0)
    m.rows.append(['Coefficient prior: auxiliary weight / seed / updates / global batch / masking (FP32, no rotation)',
                   f'{aux:g} / {seed} / {steps:,} / {batch} / {mask:g}', '0.1 / 17 / 10,626 / 64 / none'])
    v = m.check('coefficient-prior intervals excluding zero', sum(m3[b]['left'][2] < 0 for b in BUDGETS), 5, 0)
    m.rows.append(['Coefficient prior vs scratch: intervals excluding zero', str(v), '5 (every budget)'])
    prev = [-100 * r['pretrained_relative_change'] for r in load_json(BASE, 'coefficient_prior', 'final_vs_previous_patch.json')['results']]
    lo, hi = m.check('more accurate than patch prior min %', min(prev), 29, 0), m.check('more accurate than patch prior max %', max(prev), 40, 0)
    m.rows.append(['More accurate than the patch prior (%)', f'{lo:.1f}–{hi:.1f}', '29–40'])
    s17 = m.check('d, nn alone pretraining seed 17', runs['dnn_alone_seed17']['seed'], 17, 0)
    s29 = m.check('d, nn alone pretraining seed 29', runs['dnn_alone_seed29']['seed'], 29, 0)
    m.rows.append(['d, nn⊤ alone: pretraining seeds (same network, cohort and update budget)', f'{s17}/{s29}', '17/29'])
    dnn_gain = [100 * (1 - m3[b]['dnn'] / m3[b]['scratch_seed0']) for b in BUDGETS]
    lo, hi = m.check('d, nn alone below seed-0 scratch min %', min(dnn_gain), 25, 0), m.check('d, nn alone below seed-0 scratch max %', max(dnn_gain), 41, 0)
    m.rows.append(['d, nn⊤ alone lowers seed-0 scratch error by (%)', f'{lo:.1f}–{hi:.1f}', '25–41'])
    v = m.check('max |coefficients - d, nn alone|', max(abs(m3[b]['right'][0]) for b in BUDGETS), 0.007, 3)
    m.rows.append(['The two priors differ by at most (nRMSE)', fmt(v, 4), '0.007'])
    v = m.check('right intervals including zero', sum(m3[b]['right'][1] <= 0 <= m3[b]['right'][2] for b in BUDGETS), 5, 0)
    m.rows.append(['Intervals over cars that include zero', str(v), '5 (every budget)'])
    matched_scratch = {b: comp['areals', b]['scratch_seeds'] for b in BUDGETS}
    op_rows = {r['training_cars']: r for r in load_json(BASE, 'coefficient_prior', 'operator32_comparison.json')['results']}
    changed = {b: sum(abs(a - r['scratch']) > 1e-12 for a, r in zip(matched_scratch[b], op_rows[b]['seeds'])) for b in BUDGETS}
    v = m.check('cells where the per-seed rate search changes scratch', sum(changed.values()), 4, 0)
    m.rows.append(['Scratch cells changed by the per-seed rate search (per budget: '
                   + '/'.join(str(changed[b]) for b in BUDGETS) + ')', str(v), '4'])
    for b, paper in [(274, 2), (18, 1), (35, 1)]:
        v = m.check(f'changed scratch cells at {b}', changed[b], paper, 0)
    a = m.check('AREALS-arm scratch at 274', comp['areals', 274]['scratch_mean'], 0.203, 3)
    # The text quotes the Table M.3 entry (printed to 4 dp) at 3 dp.
    c = m.check('coefficient-arm scratch at 274 (from Table M.3)', as_printed(m3[274]['scratch'], 4), 0.192, 3)
    m.rows.append(['Scratch at 274 cars: AREALS arm → per-seed search', f'{fmt(a, 4)} → {fmt(c, 4)}', '0.203 → 0.192'])

    # pretraining seeds
    pq = load_json(BASE, 'coefficient_replicates', 'patch_query_windsor.json')
    for s_, paper in zip(pq['pretraining_seeds'], [17, 29, 43]):
        m.check(f'coefficient pretraining seed {paper}', s_, paper, 0)
    assert [runs[n]['seed'] for n in ['operator32', 'operator_seed29', 'operator_seed43']] == pq['pretraining_seeds']
    m.rows.append(['Coefficient prior pretraining seeds', '/'.join(map(str, pq['pretraining_seeds'])), '17/29/43'])
    v = m.check('fine-tuning seed of the pretraining-seed replicates', uniform([{'s': s_} for s_ in pq['fine_tuning_seeds']], 's'), 0, 0)
    m.rows.append(['Fine-tuning seed of each pretraining-seed copy', str(v), '0'])
    sd = [m3[b]['std'] for b in BUDGETS]
    lo, hi = m.check('pretraining-seed s.d. min', min(sd), 0.0016, 4), m.check('pretraining-seed s.d. max', max(sd), 0.0068, 4)
    m.rows.append(['Standard deviation over pretraining seeds (nRMSE)', f'{lo:.5f}–{hi:.5f}', '0.0016–0.0068'])

    # scope
    largest = [k for k, w_ in winners.items() if w_['lr'] == grid[-1]]
    v = m.check('cells choosing the largest rate', len(largest), 9, 0)
    w = m.check('... of them from scratch', sum('/scratch/' in k for k in largest), 6, 0)
    x = m.check('cells in total', len(winners), 30, 0)
    m.rows.append(['Largest encoder rate chosen in N of the cells (from scratch)', f'{v} of {x} ({w})', '9 of 30 (6)'])
    v = m.check('patch cells choosing the smallest rate',
                sum(w_['arm'] == 'patch' and w_['lr'] == grid[0] for w_ in winners.values()), 8, 0)
    m.check('patch arm cells', sum(w_['arm'] == 'patch' for w_ in winners.values()), 10, 0)
    m.rows.append(['Patch arm chose the smallest rate in N of its 10 cells', str(v), '8'])
    steps = [load_json(*p.relative_to(RESULTS).parts)['step'] for p in tuning + followup]
    v = m.check('fits best at or after update 7,000 (%)', 100 * np.mean([s_ >= 7000 for s_ in steps]), 73, 0)
    m.rows.append([f'Fits reaching their best validation at or after update 7,000 ({sum(s_ >= 7000 for s_ in steps)}/{len(steps)})',
                   f'{v:.1f}%', '73%'])
    interim = load_json(BASE, 'checkpoint_sensitivity', 'validation_summary.json')
    g = m.check('interim checkpoint updates', interim['geometry_step'], 6500, 0)
    cars = m.check('interim training cars', interim['windsor_training_cars'], 18, 0)
    sd0 = m.check('interim fine-tuning seed', interim['seed'], 0, 0)
    early = m.check('interim checkpoint validation nRMSE', interim['selected']['validation_nrmse'], 0.2522, 4)
    final = m.check('final checkpoint validation nRMSE', winners['areals/pretrained/18']['validation'], 0.2734, 4)
    m.rows.append([f'AREALS checkpoint at {g:,} updates vs final, {cars} cars, seed {sd0} (validation)',
                   f'{fmt(early, 4)} vs {fmt(final, 4)}', '0.2522 vs 0.2734'])
    return [s, m]


def tables() -> list[Table]:
    comp_doc, comp = comparison()
    values, _ = matched_test()
    for arm in ARMS:
        for b in BUDGETS:
            assert abs(values[arm, 'pretrained', b].mean() - comp[arm, b]['pretrained_mean']) < 1e-12
    prior = coefficient_prior()
    reps = replicates(prior)
    m3_table, m3 = table_m3(prior, reps)
    return [table_1(comp), table_m1(), table_m2(comp, validation_means()), m3_table,
            *text_numbers(comp_doc, comp, values, m3)]
