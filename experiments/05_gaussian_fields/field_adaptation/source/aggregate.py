import json

import numpy as np

from study import ROOT, write

GROUPS = ['masked_native_frozen', 'unmasked_native_frozen', 'masked_common_frame_frozen', 'random_frozen',
          'masked_native_reset_decoder_frozen', 'unmasked_native_reset_decoder_frozen',
          'masked_native_full', 'unmasked_native_full', 'masked_common_frame_full', 'random_full']
PAIRS = [('masked_native_frozen', 'unmasked_native_frozen'),
         ('masked_common_frame_frozen', 'unmasked_native_frozen'),
         ('masked_common_frame_frozen', 'masked_native_frozen'),
         ('masked_native_frozen', 'masked_native_reset_decoder_frozen'),
         ('unmasked_native_frozen', 'unmasked_native_reset_decoder_frozen'),
         ('masked_native_frozen', 'random_frozen'), ('unmasked_native_frozen', 'random_frozen'),
         ('masked_native_full', 'unmasked_native_full')]


def efficiency(x):
    return (x[..., 0] - x[..., 2]) / (x[..., 0] - x[..., 1])


def contrast(a, b, setting=None):
    delta = efficiency(a) - efficiency(b)
    risk_reduction = b[..., 2] - a[..., 2]
    if setting is None:
        return delta.mean(-1), risk_reduction.mean(-1)
    return delta[..., setting], risk_reduction[..., setting]


def summarise(root, groups, pairs):
    """Per-setting efficiencies and paired contrasts for both splits, written to root/SUMMARY.json."""
    all_results = {}
    for split in ['val', 'test']:
        rows = json.loads((root / f'bench/scores/grf_{split}_rows.json').read_text())
        shapes, settings = [sorted({r[key] for r in rows}) for key in ['shape', 'setting']]
        assert len(shapes) == 16 and len(settings) == 5
        draws_per_shape = 4 if split == 'val' else 8
        index = {}
        for row in rows:
            index.setdefault((row['method'], row['shape'], row['setting']), []).append(row)
        matrices, summaries = {}, {}
        rng = np.random.default_rng(91318)
        shape_draws = rng.integers(0, 16, (10000, 16))
        seed_draws = rng.integers(0, 3, (10000, 3))
        for group in groups:
            x = np.empty((16, 3, 5, 3))
            for i, shape in enumerate(shapes):
                for seed in range(3):
                    for j, setting in enumerate(settings):
                        selected = index[group + f'_seed{seed}', shape, setting]
                        assert len(selected) == draws_per_shape
                        assert len({r['case'] for r in selected}) == draws_per_shape
                        x[i, seed, j] = [np.mean([r[k] for r in selected]) for k in ['prior', 'bayes', 'risk']]
            matrices[group] = x
            point = x.mean((0, 1))
            summaries[group] = {setting: dict(efficiency=float(efficiency(point[j])), raw_mse=float(point[j, 2]),
                seeds=[dict(efficiency=float(efficiency(x[:, s, j].mean(0))), raw_mse=float(x[:, s, j, 2].mean())) for s in range(3)])
                for j, setting in enumerate(settings)}
        contrasts = []
        for first, second in pairs:
            a, b = matrices[first], matrices[second]
            assert np.array_equal(a[..., :2], b[..., :2])
            ag, bg = a[shape_draws].mean((1, 2)), b[shape_draws].mean((1, 2))
            aj = a[shape_draws[:, :, None], seed_draws[:, None, :]].mean((1, 2))
            bj = b[shape_draws[:, :, None], seed_draws[:, None, :]].mean((1, 2))
            for j, setting in [(None, 'equal_weight_all_settings')] + list(enumerate(settings)):
                point_delta, point_risk = contrast(a.mean((0, 1)), b.mean((0, 1)), j)
                geometry_delta, _ = contrast(ag, bg, j)
                joint_delta, joint_risk = contrast(aj, bj, j)
                assert ((aj[..., 0] - aj[..., 1]) > 0).all()
                seeds = []
                for s in range(3):
                    delta, risk = contrast(a[:, s].mean(0), b[:, s].mean(0), j)
                    seeds.append(dict(field_seed=s, geometry_seed=17 + s,
                        efficiency_difference=float(delta), raw_mse_reduction=float(risk)))
                interval = np.quantile(joint_delta, [.025, .975]).tolist()
                positive = interval[0] > 0 and all(v['efficiency_difference'] > 0 for v in seeds)
                negative = interval[1] < 0 and all(v['efficiency_difference'] < 0 for v in seeds)
                contrasts.append(dict(first=first, second=second, setting=setting,
                    efficiency_difference=float(point_delta),
                    raw_mse_reduction=float(point_risk),
                    geometry_interval=np.quantile(geometry_delta, [.025, .975]).tolist(),
                    seed_and_geometry_interval=interval,
                    raw_mse_seed_and_geometry_interval=np.quantile(joint_risk, [.025, .975]).tolist(),
                    per_seed=seeds, directional_verdict='first_better' if positive else 'second_better' if negative else 'unresolved'))
        all_results[split] = dict(summary=summaries, contrasts=contrasts, geometries=16,
            fields_per_geometry=draws_per_shape, seeds=3, bootstrap_seed=91318, bootstrap_draws=10000)
    write(root / 'SUMMARY.json', all_results)
    return all_results


def main():
    all_results = summarise(ROOT, GROUPS, PAIRS)
    lines = ['# Gaussian-field efficiency on the test fields', '',
        '| Test setting | Masked native, frozen | Unmasked native, frozen | Masked common frame, frozen | Random, frozen |',
        '|---|---:|---:|---:|---:|']
    for setting in all_results['test']['summary']['masked_native_frozen']:
        cells = [f"{all_results['test']['summary'][group][setting]['efficiency']:.3f}"
                 for group in ['masked_native_frozen', 'unmasked_native_frozen', 'masked_common_frame_frozen', 'random_frozen']]
        lines.append('| ' + setting + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Equal-weight contrasts', '', '| First minus second | Efficiency difference | 95% seed-and-geometry interval |', '|---|---:|---|']
    for c in all_results['test']['contrasts']:
        if c['setting'] == 'equal_weight_all_settings':
            lines.append(f"| {c['first']} minus {c['second']} | {c['efficiency_difference']:+.4f} | {c['seed_and_geometry_interval']} |")
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
