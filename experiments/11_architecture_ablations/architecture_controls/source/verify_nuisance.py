import json
from pathlib import Path

import numpy as np

from areal_tokens.data import stable_seed
from summarize_architecture import interval, write

ROOT = Path('/workspace/runs/areals_architecture_controls_20261024')
OUT = ROOT / 'results/nuisance_raw'


def metrics(distance, normals, baseline):
    d = distance.astype(np.float64)
    n = normals.astype(np.float64)
    truth = baseline['truth'][0].astype(np.float64)
    qn = baseline['query_normals'][0].astype(np.float64)
    angle = np.degrees(np.arccos(np.clip(np.abs(np.sum(n * qn, axis=-1)), 0, 1)))
    valid = baseline['normal_valid'][0]
    hidden = baseline['hidden'][0]
    result = dict(nmse=float(np.mean(((d-truth)/baseline['scale'][0, 0])**2)))
    for key, pick in [('normal_all', valid), ('normal_hidden', valid & hidden)]:
        result[key] = float(angle[pick].mean()) if pick.any() else None
    return result


def read(number):
    with np.load(OUT / f'prediction{number:04d}.npz') as raw:
        return {k: raw[k] for k in raw.files}


def main():
    original = json.loads((OUT / 'rows.json').read_text())
    records = json.loads((OUT / 'manifest.json').read_text())['records']
    assert len(original) == 192 and len(records) == 32
    config = json.loads((ROOT / 'production_config.json').read_text())
    counter, recomputed = 0, []
    maximum = dict(nmse=0., normal_degrees=0.)
    for mask in [0., .6]:
        for record in records:
            counter += 1
            baseline = read(counter)
            with np.load(record['path']) as source:
                # Match the immutable identity to the saved raw-query stream.
                rng = np.random.default_rng(stable_seed(config['train']['seed'], record['id']+':queries', 0))
                chosen = rng.choice(len(source['udf']), config['data']['queries'], replace=False)
                assert np.array_equal(source['queries'][chosen], baseline['queries'][0])
                assert np.array_equal(source['udf'][chosen], baseline['truth'][0])
            base = metrics(baseline['distance'][0], baseline['normals'][0], baseline)
            for nuisance in ['resample', 'rotation', 'sign']:
                counter += 1
                raw = read(counter)
                assert raw['distance'].shape == (4, 1024)
                for view in range(4):
                    assert np.array_equal(raw['truth'][view], baseline['truth'][0])
                    assert np.array_equal(raw['query_normals'][view], baseline['query_normals'][0])
                    r = raw['rotation'][view].astype(np.float64)
                    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-6)
                    np.testing.assert_allclose(raw['queries'][view], baseline['queries'][0] @ r.T, atol=2e-6)
                normals = np.einsum('bqi,bij->bqj', raw['normals'].astype(np.float64), raw['rotation'].astype(np.float64))
                distance = raw['distance'].astype(np.float64)
                projector = np.einsum('bqi,bqj->qij', normals, normals)/4
                average_normal = np.linalg.eigh(projector)[1][..., :, -1]
                view_scores = [metrics(d, n, baseline) for d,n in zip(distance, normals)]
                view = {k: float(np.mean([s[k] for s in view_scores])) if base[k] is not None else None for k in base}
                orbit = metrics(distance.mean(0), average_normal, baseline)
                row = dict(id=record['id'], mask=mask, nuisance=nuisance, base=base, view=view, orbit=orbit)
                saved = next(r for r in original if (r['id'], r['mask'], r['nuisance'])==(record['id'],mask,nuisance))
                for mode in ['base', 'view', 'orbit']:
                    for key in base:
                        if row[mode][key] is None:
                            assert not np.isfinite(saved[mode][key])
                        else:
                            family = 'nmse' if key=='nmse' else 'normal_degrees'
                            maximum[family] = max(maximum[family], abs(row[mode][key]-saved[mode][key]))
                recomputed.append(row)
    assert counter == 256 and maximum['nmse'] < 1e-7 and maximum['normal_degrees'] < .02
    summaries = []
    for mask in [0., .6]:
        for nuisance in ['resample', 'rotation', 'sign']:
            rows = [r for r in recomputed if r['mask']==mask and r['nuisance']==nuisance]
            summary = dict(mask=mask, nuisance=nuisance, metrics={})
            for key in ['nmse', 'normal_all', 'normal_hidden']:
                if rows[0]['base'][key] is None:
                    continue
                values = {mode: np.asarray([r[mode][key] for r in rows]) for mode in ['base','view','orbit']}
                if key=='nmse':
                    values = {mode: np.sqrt(x) for mode,x in values.items()}
                summary['metrics']['normalized_rmse' if key=='nmse' else key] = dict(
                    baseline=interval(values['base']), view_minus_baseline=interval(values['view']-values['base']),
                    orbit_minus_baseline=interval(values['orbit']-values['base']))
            summaries.append(summary)
    write(OUT / 'independent_grouped.json', dict(passed=True, raw_prediction_files=256, meshes=32,
          rows=192, maximum_metric_disagreement=maximum, settings=summaries))
    print('NUISANCE_VERIFIED', json.dumps(maximum))


if __name__ == '__main__':
    main()
