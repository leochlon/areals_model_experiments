import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(os.environ.get('STUDY_ROOT', Path(__file__).resolve().parents[1]))


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-index', type=int, help='Also rerun one selected model on the first test car')
    args = parser.parse_args()
    matrix = read(ROOT / 'matrix.json')
    maximum, count = 0., 0
    norms = {}
    for job in matrix:
        folder = ROOT / 'runs' / job['name']
        complete = read(folder / 'complete.json')
        test = read(folder / 'test.json')
        assert len(test['rows']) == 100
        key = f'{job["seed"]}/{job["budget"]}'
        if key in norms:
            assert complete['normalisation'] == norms[key]
        norms[key] = complete['normalisation']
        for row in test['rows']:
            with np.load(ROOT / 'data/fields' / (row['case'] + '.npz')) as f:
                truth, weights = f['values'].astype(np.float64), f['weights'].astype(np.float64)
            with np.load(folder / 'predictions' / (row['case'] + '.npz')) as pred:
                prediction = pred['prediction'][0 if row['sensors'] == 0 else 1].astype(np.float64)
                observed = pred['observed20'] if row['sensors'] else []
            hidden = np.ones(len(truth), bool); hidden[observed] = False
            spread = np.sqrt(np.average((truth-np.average(truth, weights=weights))**2, weights=weights))
            score = np.sqrt(np.average((prediction[hidden]-truth[hidden])**2, weights=weights[hidden])) / spread
            error = abs(score-row['nrmse'])
            assert error < 1e-6, (job['name'], row['case'], error)
            maximum = max(maximum, float(error)); count += 1
    result = dict(passed=True, scored_rows=count,
                  paired_training_normalisation_identical=True, max_metric_difference=maximum,
                  metric_tolerance=1e-6)
    if args.gpu_index is not None:
        sys.path.insert(0, str(ROOT / 'source'))
        sys.path.insert(0, str(ROOT / 'model_src'))
        import torch
        import study
        torch.set_num_threads(2)
        torch.backends.cuda.enable_cudnn_sdp(False)
        job = matrix[args.gpu_index]
        folder = ROOT / 'runs' / job['name']
        model, _, state = study.fb.load_field_encoder(folder / 'best.pt', 'cuda')
        evaluation = study.problems('test')[:1]
        rows = study.evaluate(model, evaluation, state['benchmark'])
        old = {(r['case'], r['sensors']): r for r in read(folder / 'test.json')['rows']}
        gap = max(abs(r['nrmse']-old[(r['case'],r['sensors'])]['nrmse']) for r in rows)
        assert math.isfinite(gap) and gap < 1e-5, gap
        result.update(gpu_checkpoint_replay=job['name'], gpu_score_difference=gap,
                      gpu_replay_tolerance=1e-5, gpu=torch.cuda.get_device_name())
    (ROOT / 'results/replay.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
