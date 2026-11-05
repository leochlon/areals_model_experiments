import hashlib
import json
import os
from pathlib import Path

import numpy as np

import field_benchmark as fb
from surface_fields import SurfaceGRF
from study import ROOT, write


def fingerprint(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    old = Path(os.environ['TASK_ROOT']) / 'grf'
    records = json.loads((ROOT / 'field_manifest.json').read_text())['records']
    original = json.loads((old / 'manifest.json').read_text())['records']
    assert len({r['id'] for r in records}) == len(records)
    families = {s: {r['family'] for r in records if r['split'] == s} for s in ['train', 'val', 'test']}
    assert [len(families[s]) for s in ['train', 'val', 'test']] == [228, 16, 16]
    assert not families['train'] & families['val']
    assert not families['train'] & families['test']
    assert not families['val'] & families['test']
    physical = {s: set() for s in families}
    for record in records:
        field = ROOT / 'fields' / (record['id'] + '.npz')
        with np.load(field) as packed:
            physical[record['split']].add(fingerprint(packed['points']))
        if record['split'] != 'test':
            before = (old / 'fields' / field.name).stat()
            after = field.stat()
            assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
    assert not physical['train'] & physical['val']
    assert not physical['train'] & physical['test']
    assert not physical['val'] & physical['test']
    old_test_ids = {r['id'] for r in original if r['split'] == 'test'}
    assert not old_test_ids & {r['id'] for r in records}
    old_values = set()
    for identity in old_test_ids:
        with np.load(old / 'fields' / (identity + '.npz')) as packed:
            old_values.add(fingerprint(packed['values']))
    rng = np.random.default_rng(91317)
    confirmed = 0
    for family in sorted(families['test']):
        with np.load(ROOT / 'bases' / (family + '.npz')) as b:
            grf = SurfaceGRF(b['lam'], b['phi'].astype(np.float64), b['mass'],
                ells=tuple(b['ells']), nu=float(b['nu']), sigma=.001, scale=float(b['scale']))
        expected, _ = grf.sample(rng, 8)
        for draw in range(8):
            identity = f'{family}__prodconfirm{draw}'
            with np.load(ROOT / 'fields' / (identity + '.npz')) as packed:
                assert np.array_equal(packed['values'], expected[draw].astype(np.float32))
                assert fingerprint(packed['values']) not in old_values
            confirmed += 1
    meta, arrays = fb.read_observations(ROOT / 'bench', 'test')
    assert len(meta['entries']) == 640
    reference_checks = []
    for entry in [next(e for e in meta['entries'] if e['setting'] == s) for s in ['sensors-5', 'sensors-20']]:
        record = next(r for r in records if r['id'] == entry['car'])
        with np.load(ROOT / 'fields' / (record['id'] + '.npz')) as packed:
            values = packed['values'].astype(np.float64)
        with np.load(ROOT / 'bases' / (record['family'] + '.npz')) as b:
            grf = SurfaceGRF(b['lam'], b['phi'].astype(np.float64), b['mass'],
                ells=tuple(b['ells']), nu=float(b['nu']), sigma=.001, scale=float(b['scale']))
        observed = arrays[entry['key']]
        y = values[observed]
        actual, actual_weights = grf.posterior_mean(observed, y)
        means, likelihoods = [], []
        for spectrum in grf.spectra:
            covariance = (grf.phi[observed] * spectrum) @ grf.phi[observed].T + grf.sigma ** 2 * np.eye(len(y))
            chol = np.linalg.cholesky(covariance)
            alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
            means.append((grf.phi * spectrum) @ (grf.phi[observed].T @ alpha))
            likelihoods.append(-.5 * (y @ alpha + 2 * np.log(np.diag(chol)).sum() + len(y) * np.log(2 * np.pi)))
        weights = np.exp(likelihoods - np.max(likelihoods))
        weights /= weights.sum()
        expected = weights @ np.stack(means)
        error = float(np.max(np.abs(actual - expected)))
        assert error < 1e-4, error
        assert np.allclose(actual_weights, weights, atol=1e-5, rtol=1e-5)
        reference_checks.append(dict(setting=entry['setting'], max_absolute_error=error))
    write(ROOT / 'field_data_audit.json', dict(passed=True, families={s: len(v) for s, v in families.items()},
        fresh_fields_reproduced=confirmed, test_observation_sets=640, reference_checks=reference_checks))
    print('field data checks passed', confirmed, reference_checks, flush=True)


if __name__ == '__main__':
    main()
