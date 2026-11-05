import json
from pathlib import Path

import numpy as np

ROOT = Path('/workspace/runs/areals_architecture_controls_20261024')
ARMS = ['reference', 'ordinary_attention', 'unanchored_latents', 'oriented_normal_target',
        'unoriented_vector_target', 'anchored_no_rotary', 'unanchored_no_rotary']


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    assert np.isfinite(values).all() and values.ndim == 1
    draws = np.random.default_rng(20261024).integers(0, len(values), (4000, len(values)))
    low, high = np.quantile(values[draws].mean(1), [.025, .975])
    return dict(mean=float(values.mean()), low=float(low), high=float(high), objects=len(values))


def geometry():
    folder = ROOT / 'results/geometry_audit'
    verification = json.loads((folder / 'verification.json').read_text())
    assert verification['passed'] and verification['models'] == 21
    values, seeds = {}, []
    identities = None
    for arm in ARMS:
        rows = []
        for seed in range(3):
            with np.load(folder / f'{arm}_seed{seed}.npz') as raw:
                ids = sorted(k[:-len('_normalized_prediction')] for k in raw.files if k.endswith('_normalized_prediction'))
                assert len(ids) == 128
                if identities is None:
                    identities = ids
                assert ids == identities
                per_object = []
                for identity in ids:
                    get = lambda key: raw[identity + '_' + key].astype(np.float64)
                    distance = np.mean((get('normalized_prediction') - get('target') / get('scale'))**2)
                    valid = raw[identity + '_valid']
                    cosine = np.sum(get('normals') * get('target_normals'), axis=-1)
                    normal = np.degrees(np.arccos(np.clip(np.abs(cosine[valid]), 0, 1))).mean()
                    assert np.isfinite(distance) and np.isfinite(normal)
                    per_object.append([distance, normal])
                rows.append(per_object)
                seeds.append(dict(arm=arm, seed=seed, normalized_mse=float(np.mean(rows[-1], axis=0)[0]),
                                  normal_degrees=float(np.mean(rows[-1], axis=0)[1])))
        values[arm] = np.asarray(rows)
    summary = {arm: {key: interval(array[:, :, index].mean(0))
                    for index, key in enumerate(['normalized_mse', 'normal_degrees'])}
               for arm, array in values.items()}
    pairs = [('reference', 'ordinary_attention'), ('reference', 'unanchored_latents'),
             ('anchored_no_rotary', 'unanchored_no_rotary'), ('reference', 'oriented_normal_target'),
             ('reference', 'unoriented_vector_target'), ('oriented_normal_target', 'unoriented_vector_target')]
    contrasts = []
    for first, second in pairs:
        contrasts.append(dict(first=first, second=second, direction='first minus second; negative favours first',
              **{key: interval((values[first][:, :, index]-values[second][:, :, index]).mean(0))
                 for index, key in enumerate(['normalized_mse', 'normal_degrees'])}))
    result = dict(passed=True, models=21, test_objects=128, summaries=summary, seeds=seeds, contrasts=contrasts)
    write(folder / 'grouped.json', result)
    print('ARCHITECTURE_GROUPED', json.dumps({k:{m:v['mean'] for m,v in x.items()} for k,x in summary.items()}))


def density():
    folder = ROOT / 'results/density_stress'
    original = json.loads((folder / 'rows.json').read_text())
    assert len(original) == 1344
    rows, maximum = [], 0.
    cache = {}
    for row in original:
        key = (row['id'], row['arm'], row['seed'])
        if key not in cache:
            path = folder / f'{key[0]}_{key[1]}_seed{key[2]}.npz'
            with np.load(path) as z:
                cache[key] = {k:z[k].astype(np.float64) for k in z.files if not k.endswith('_normal')}
        z = cache[key]
        prediction = z[f'{row["mode"]}_{row["draw"]}_distance']
        mse = float(np.mean(((prediction-z['truth'])/z['scale'])**2))
        change = float(np.sqrt(np.mean(((prediction-z['uniform_101_distance'])/z['scale'])**2)))
        maximum = max(maximum, abs(mse-row['normalized_mse']), abs(change-row['prediction_change']))
        rows.append({**row, 'normalized_mse':mse, 'prediction_change':change})
    assert maximum < 1e-12 and len(cache) == 192
    ids = sorted({r['id'] for r in rows})
    assert len(ids) == 32
    matrix = {}
    summaries = {}
    for arm in ['reference', 'ordinary_attention']:
        summaries[arm] = {}
        for mode in ['uniform', 'biased_fixed_anchors_frame', 'biased_reanchor']:
            for metric in ['normalized_mse', 'prediction_change']:
                values = np.asarray([np.mean([r[metric] for r in rows if r['id']==identity
                      and r['arm']==arm and r['mode']==mode]) for identity in ids])
                matrix[arm, mode, metric] = values
                summaries[arm].setdefault(mode, {})[metric] = interval(values)
    contrasts = {}
    for mode in ['biased_fixed_anchors_frame', 'biased_reanchor']:
        contrasts[mode] = {}
        for metric in ['normalized_mse', 'prediction_change']:
            ref = matrix['reference', mode, metric]
            ordinary = matrix['ordinary_attention', mode, metric]
            contrasts[mode][metric+'_reference_minus_ordinary'] = interval(ref-ordinary)
        for arm in ['reference', 'ordinary_attention']:
            contrasts[mode][arm+'_mse_change_from_uniform'] = interval(
                matrix[arm, mode, 'normalized_mse']-matrix[arm, 'uniform', 'normalized_mse'])
    write(folder / 'independent_grouped.json', dict(passed=True, predictions=len(rows), models=6, meshes=32,
          maximum_metric_disagreement=maximum, summaries=summaries, contrasts=contrasts))
    print('DENSITY_VERIFIED', maximum, json.dumps(contrasts))


if __name__ == '__main__':
    geometry()
    density()
