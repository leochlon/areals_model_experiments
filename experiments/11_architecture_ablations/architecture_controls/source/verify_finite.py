import json
from pathlib import Path

import numpy as np

ROOT = Path('/workspace/runs/areals_architecture_controls_20261024/results/finite_token_raw')


def main():
    # Hand-check the metric formula before applying it to model outputs.
    assert float(np.mean(((np.array([1., 3.]) - np.array([0., 1.])) / 2.) ** 2)) == .625
    rows = json.loads((ROOT / 'rows.json').read_text())
    files = sorted(ROOT.glob('*.npz'))
    assert len(files) == 64
    worst = dict(distance_nmse=0., output_change=0., learned_head_change=0., normal_degrees=0.)
    checked = 0
    for path in files:
        identity, tokens = path.stem.rsplit('_', 1)
        relevant = [r for r in rows if r['id'] == identity and r['tokens'] == int(tokens)]
        assert len(relevant) == 7
        with np.load(path) as a:
            target, scale = a['truth'].astype(np.float64), float(a['scale'])
            for row in relevant:
                prefix = row['mode'] + '_' + str(row['seed'])
                p = a[prefix + '_distance'].astype(np.float64)
                base = a['reference_101_distance'].astype(np.float64)
                delta = (p - base) / scale
                head = a[prefix + '_head'].astype(np.float64)
                base_head = a['reference_101_head'].astype(np.float64)
                metrics = dict(distance_nmse=float(np.sum((p - target) ** 2) / len(target) / scale ** 2),
                               output_change=float(np.sqrt(np.sum(delta ** 2) / len(delta))),
                               learned_head_change=float(np.sqrt(np.sum(((head - base_head) / scale) ** 2) / len(delta))))
                valid = target / scale <= float(a['normal_band'])
                n = a[prefix + '_normal'].astype(np.float64)
                cosine = np.einsum('ij,ij->i', n, a['query_normals'])
                if valid.any():
                    metrics['normal_degrees'] = float(np.mean(np.arccos(np.minimum(np.abs(cosine[valid]), 1))) * 180 / np.pi)
                for name, value in metrics.items():
                    worst[name] = max(worst[name], abs(value - row[name]))
                checked += 1
    assert worst['distance_nmse'] < 1e-10
    assert worst['output_change'] < 1e-7
    assert worst['learned_head_change'] < 1e-7
    assert worst['normal_degrees'] < 1e-4
    result = dict(passed=True, hand_fixture=True, geometries=32, token_settings=2,
                  raw_prediction_cases=checked, maximum_disagreements=worst)
    (ROOT / 'independent_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
