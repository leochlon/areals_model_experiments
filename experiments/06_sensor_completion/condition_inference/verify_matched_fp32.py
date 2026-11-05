import argparse
from common import *
from analyse import summary
from verify_predictions import independent


def main(partial=False):
    summaries = []
    completed = []
    for name in ['nine_angle', 'doubledelta', 'blendednet']:
        rows = []
        if not all((ROOT / 'matched_fp32' / name / f'seed{s}/complete.json').exists() for s in range(3)):
            if partial:
                continue
            raise AssertionError(f'Incomplete: {name}')
        lookup = {r['id']: r for r in records(name, 'test')}
        obs, masks = fb.read_observations(ROOT / 'data' / name, 'test')
        entries = {(e['car'], e['setting']): e for e in obs['entries']}
        maximum = dict(nrmse=0., nmse=0., force_rel=0.)
        for seed in range(3):
            folder = ROOT / 'matched_fp32' / name / f'seed{seed}'
            done = read(folder / 'complete.json')
            assert done['status'] == 'PASS'
            groups = {}
            for row in read(folder / 'rows.json'):
                groups.setdefault(row['case'], []).append(row)
            assert set(groups) == set(lookup)
            for case, case_rows in groups.items():
                f = field(name, lookup[case])
                assert {r['setting'] for r in case_rows} == {'sensors-5', 'sensors-20', 'sensors-100', 'sensors-500'}
                with np.load(folder / (case + '.npz')) as predictions:
                    for row in case_rows:
                        e = entries[(case, row['setting'])]
                        actual = independent(f, masks[e['key']], predictions[e['key']].astype(np.float64))
                        for metric, value in actual.items():
                            maximum[metric] = max(maximum[metric], abs(value - row[metric]))
                rows.extend(case_rows)
        assert max(maximum.values()) < 1e-5, maximum
        completed.append(dict(dataset=name, predictions=len(rows), maximum_score_difference=maximum))
        for setting in ['sensors-5', 'sensors-20', 'sensors-100', 'sensors-500']:
            summaries.append(dict(dataset=name, **summary(rows, setting)))
    write(ROOT / 'results/matched_fp32_areals.json', dict(status='PASS' if len(completed) == 3 else 'PARTIAL',
          completed=completed, summaries=summaries, precision='FP32', cudnn_sdp=False))
    print(json.dumps(completed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    main(parser.parse_args().partial)
