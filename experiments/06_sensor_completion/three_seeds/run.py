import argparse
import numpy as np
import torch

from common import ROOT, BENCH, read, write, model_path, train_args
import study

study.configure('measurements')
fb = study.fb

def pilot():
    config = fb.bench_config(BENCH)
    meta, masks = fb.read_observations(BENCH, 'val')
    cases = sorted({e['car'] for e in meta['entries']})[:2]
    method = fb.Areal.load('measurements', model_path(0) / 'field_best.pt', torch.device('cuda'), 16, 'bf16')
    predictions, _, skipped = fb.predict_split(method, config, meta, masks, config['fields'], cars=set(cases), progress=False)
    assert not skipped and len(predictions) == 16
    with np.load(BENCH / 'predictions/measurements_seed0/val.npz') as stored:
        difference = max(float(np.max(np.abs(v.astype(np.float32) - stored[k]))) for k,v in predictions.items())
    assert difference <= 1e-5, difference
    fb.main(['check', '--bench', str(BENCH), '--split', 'val', '--method', 'areal', '--name', 'measurements',
             '--model', str(model_path(0) / 'field_best.pt'), '--cars', '1', '--per-car', '2'])
    write(ROOT / 'pilot.json', dict(status='PASS', seed0_validation_replay_predictions=16,
          maximum_prediction_difference=difference, gpu=torch.cuda.get_device_name()))
    print('pilot passed', difference, flush=True)

def train(seed):
    assert seed in [1,2]
    assert not model_path(seed).exists(), f'{model_path(seed)} already exists'
    fb.main(train_args(seed))
    history = read(model_path(seed) / 'history.json')
    assert len(history) == 40 and [r['epoch'] for r in history] == list(range(1,41))
    assert all(np.isfinite(r[k]) for r in history for k in ['train_loss','val_nrmse'])
    best = torch.load(model_path(seed) / 'field_best.pt', map_location='cpu', weights_only=False)
    assert best['epoch'] == min(history,key=lambda r:r['val_nrmse'])['epoch']
    seed0, meta = read(ROOT / 'protocol.json')['seed0_metadata'], best['benchmark']
    assert meta['seed'] == seed
    assert all(meta[k] == seed0[k] for k in meta.keys() & seed0.keys() if k != 'seed'), 'settings differ from seed 0'
    write(ROOT / 'runs' / f'seed{seed}/training_complete.json', dict(status='PASS', seed=seed, epochs=40,
          updates=10800, selected_epoch=best['epoch']))

def evaluate(seed):
    model = model_path(seed) / 'field_best.pt'
    name = f'measurements_seed{seed}'
    for split in ['val','test']:
        args = ['predict', '--bench', str(BENCH), '--split', split, '--method', 'areal', '--name', name,
                '--model', str(model), '--batch-size', '16', '--amp', 'bf16']
        if split == 'test':
            args.append('--final')
        fb.main(args)
        folder = BENCH / 'predictions' / name
        info = read(folder / f'{split}.json')
        assert info['problems'] == 1152 and info['method']['source']['model'] == str(model.resolve())
    fb.main(['check', '--bench', str(BENCH), '--split', 'val', '--method', 'areal', '--name', name,
             '--model', str(model)])
    write(ROOT / 'runs' / f'seed{seed}/evaluation_complete.json', dict(status='PASS', seed=seed))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['pilot','train','evaluate'])
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.stage == 'pilot': pilot()
    elif args.stage == 'train': train(args.seed)
    else: evaluate(args.seed)
