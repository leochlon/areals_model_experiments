import argparse
from common import *
from latency import sensor_forward


def score(index):
    setup_torch()
    name = ['nine_angle', 'doubledelta', 'blendednet'][index // 3]
    seed = index % 3
    path = historical_checkpoint(name, 'measurements', seed)
    model, _, state = load_checkpoint(path, name, 'measurements')
    out = ROOT / 'matched_fp32' / name / f'seed{seed}'
    assert not out.exists(), out
    out.mkdir(parents=True)
    obs, masks = fb.read_observations(ROOT / 'data' / name, 'test')
    groups = fb.by_car(obs['entries'])
    rows = []
    for number, r in enumerate(records(name, 'test'), 1):
        f, p = field(name, r), prepared(name, r)
        predictions = {}
        for e in groups[r['id']]:
            if e['setting'] not in {'sensors-5', 'sensors-20', 'sensors-100', 'sensors-500'}:
                continue
            ix = masks[e['key']]
            prediction = sensor_forward(model, p, f, ix, state['benchmark'])
            result = metrics(f, ix, prediction.astype(np.float64))
            rows.append(dict(case=r['id'], family=r['family'], seed=seed, setting=e['setting'], **result))
            predictions[e['key']] = prediction.astype(np.float32)
        assert len(predictions) == 4
        np.savez_compressed(out / (r['id'] + '.npz'), **predictions)
        if number % 32 == 0:
            write(out / 'progress.json', dict(cases=number, total=len(records(name, 'test'))))
    write(out / 'rows.json', rows)
    write(out / 'complete.json', dict(status='PASS', dataset=name, seed=seed, checkpoint=str(path),
          precision='FP32', cudnn_sdp=False, cases=number, predictions=len(rows)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True)
    score(parser.parse_args().index)
