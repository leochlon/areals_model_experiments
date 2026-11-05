import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(os.environ.get('STUDY_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'model_src'))
import field_benchmark as fb
from areal_tokens.field import FieldEncoder, collate_field, field_loss, save_field_encoder
from areal_tokens.io import load_model
from areal_tokens.model import GeometryAutoencoder


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def stable(value):
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'little')


def records():
    rows = read(ROOT / 'data/manifest.json')['records']
    return [dict(r, path=str(ROOT / 'data' / r['path'])) for r in rows]


def cache_path(identity):
    return ROOT / 'cache/geometry' / (identity + '.npz')


def prepared(record):
    with np.load(cache_path(record['id']), allow_pickle=False) as a:
        arrays = {k: a['a_' + k].copy() for k in fb.CLOUD_KEYS if 'a_' + k in a}
        return fb.Prepared(arrays, a['sample_to_field'].copy(), a['field_to_sample'].copy(),
                           a['anchor_index'].copy())


def field(record):
    path = ROOT / 'data/fields' / (record['id'] + '.npz')
    return fb.load_field(path.parent, record['id'])


@lru_cache(maxsize=1)
def cache_experiment():
    torch.set_num_threads(1)
    _, experiment, _ = load_model(ROOT / 'checkpoints/geometry.pt', 'cpu')
    return experiment


def cache_one(record):
    experiment = cache_experiment()
    f = fb.load_field(ROOT / 'data/fields', record['id'])
    assert len(f.values) >= 1000 and np.isfinite(f.values).all()
    record_for_geometry = dict(record, id=record.get('geometry_id', record['family']))
    p = fb.prepare_car(record_for_geometry, f.points, experiment)
    path = cache_path(record['id'])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{'a_' + k: v for k, v in p.arrays.items() if k in fb.CLOUD_KEYS},
             sample_to_field=p.sample_to_field, field_to_sample=p.field_to_sample,
             anchor_index=p.anchor_index)
    return path.name


def prepare():
    assert not (ROOT / 'matrix.json').exists(), 'Study is already prepared'
    protocol = read(ROOT / 'protocol.json')
    assert read(ROOT / 'data_ready.json')['passed']
    rows = records()
    by_split = {s: [r for r in rows if r['split'] == s] for s in ('train', 'val', 'test')}
    assert {s: len(v) for s, v in by_split.items()} == protocol['split']
    families = {}
    for r in rows:
        families.setdefault(r['family'], set()).add(r['split'])
    assert all(len(s) == 1 for s in families.values()), 'Physical family crosses splits'
    selected, matrix = {}, []
    for seed in protocol['seeds']:
        order = sorted(by_split['train'], key=lambda r: hashlib.sha256(
            f'areals-efficiency-suite-v1|{seed}|{r["id"]}'.encode()).hexdigest())
        for n in protocol['budgets']:
            selected[f'{seed}/{n}'] = [r['id'] for r in order[:n]]
            assert len(selected[f'{seed}/{n}']) == n
            for arm in protocol['arms']:
                matrix.append(dict(index=len(matrix), seed=seed, budget=n, arm=arm,
                                   name=f'{arm}_n{n}_seed{seed}'))
    write(ROOT / 'subsets.json', selected)
    write(ROOT / 'matrix.json', matrix)
    checkpoint = ROOT / 'checkpoints/geometry.pt'
    _, experiment, state = load_model(checkpoint, 'cpu')
    assert experiment.model.latents == 256 and experiment.data.encoder_points == 4096
    write(ROOT / 'evidence/checkpoint.json', dict(epoch=state.get('epoch'),
          config=state['config'], keys=list(state), parameters=sum(v.numel() for v in state['model'].values())))
    with ProcessPoolExecutor(8, mp_context=multiprocessing.get_context('spawn')) as pool:
        for i, _ in enumerate(pool.map(cache_one, rows, chunksize=4)):
            if (i + 1) % 100 == 0:
                print('CACHED', i + 1, '/', len(rows), flush=True)
    print('PREPARED', {s: len(v) for s, v in by_split.items()}, flush=True)


class Training(Dataset):
    def __init__(self, rows, seed, steps, spec):
        self.cars = [(r, field(r), prepared(r)) for r in rows]
        self.seed, self.steps, self.spec = seed, steps, spec
        self.mean, self.std = fb.training_statistics([f for _, f, _ in self.cars])
        self.meta = dict(normalise='dataset', mean=self.mean, std=self.std)
        self.round_size = len(self.cars)
        rounds = math.ceil(steps * spec['batch_size'] / self.round_size)
        self.order = np.concatenate([np.random.default_rng(stable(f'order|{seed}|{i}')).permutation(self.round_size)
                                     + self.round_size * i for i in range(rounds)])
        self.offset = 0

    def __len__(self):
        return self.steps * self.spec['batch_size'] - self.offset

    def __getitem__(self, index):
        draw = int(self.order[index + self.offset])
        round_id, slot = divmod(draw, self.round_size)
        r, f, p = self.cars[slot]
        rng = np.random.default_rng(stable(f'draw|{self.seed}|{round_id}|{slot}|{r["id"]}'))
        observed = np.empty(0, np.int64)
        if rng.random() >= self.spec['sensor_mix']['unobserved']:
            low, high = self.spec['sensor_range']
            count = int(round(math.exp(rng.uniform(math.log(low), math.log(high)))))
            observed = np.sort(rng.choice(len(f.points), count, replace=False, p=f.weights / f.weights.sum()))
        values = (f.values - self.mean) / self.std
        sample = fb.encoder_inputs(p, len(f.points), observed, values[observed])
        chosen = rng.choice(len(f.points), self.spec['queries'], replace=True, p=f.weights / f.weights.sum())
        sample.update(id=r['id'], family=r['family'], split='train',
                      queries=f.points[chosen].astype(np.float32), target=values[chosen].astype(np.float32),
                      hidden=(~np.isin(chosen, observed)).astype(np.float32),
                      query_weights=np.full(self.spec['queries'], 1 / self.spec['queries'], np.float32))
        return sample


def problems(split):
    out = []
    for r in sorted(records(), key=lambda r: r['id']):
        if r['split'] != split:
            continue
        f, p = field(r), prepared(r)
        observed = []
        for k in (0, 20):
            rng = np.random.default_rng(stable(f'eval-v1|{split}|{r["id"]}|{k}'))
            ix = np.sort(rng.choice(len(f.points), k, replace=False, p=f.weights / f.weights.sum()))
            observed.append((k, ix, f.values[ix].copy()))
        out.append((r, f, p, observed))
    return out


@torch.no_grad()
def evaluate(model, evaluation, meta, predictions=None, save_sensor=None):
    model.eval()
    rows = []
    for r, f, p, observations in evaluation:
        samples = [fb.encoder_inputs(p, len(f.points), ix, (v - meta['mean']) / meta['std'])
                   for _, ix, v in observations]
        batch = fb.to_device(fb.collate_inputs(samples), 'cuda')
        queries = torch.as_tensor(f.points, dtype=torch.float32, device='cuda')[None].expand(2, -1, -1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            geom = model.encode(batch, batch['values'], batch['present'], batch['anchor_values'], batch['anchor_present'])
            pred = model.predict(geom, queries)
        pred = pred.float().cpu().numpy().astype(np.float64) * meta['std'] + meta['mean']
        assert np.isfinite(pred).all()
        for (k, ix, _), values in zip(observations, pred):
            row = dict(case=r['id'], family=r['family'], sensors=k, **fb.problem_metrics(f, ix, values))
            assert all(math.isfinite(row[v]) for v in ('nrmse', 'rel_l2'))
            rows.append(row)
        if predictions is not None:
            assert save_sensor in (0, 20)
            predictions.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(predictions / (r['id'] + '.npz'),
                                prediction=pred[0 if save_sensor == 0 else 1].astype(np.float32),
                                observed20=observations[1][1], sensors=save_sensor)
    return rows


def mean_primary(rows, sensors=0):
    return float(np.mean([r['nrmse'] for r in rows if r['sensors'] == sensors]))


def selected(job):
    ids = read(ROOT / 'subsets.json')[f'{job["seed"]}/{job["budget"]}']
    lookup = {r['id']: r for r in records()}
    return [lookup[i] for i in ids]


def initialise(seed, arm):
    torch.manual_seed(seed)
    base, experiment, state = load_model(ROOT / 'checkpoints/geometry.pt', 'cpu')
    if arm == 'scratch':
        base = GeometryAutoencoder(experiment.model)
    torch.manual_seed(57300 + seed)
    return FieldEncoder(base).cuda(), experiment, state


def check():
    """Paired new heads, hidden values never reach the inputs, resumable draws, a finite GPU step."""
    protocol = read(ROOT / 'protocol.json')
    job = read(ROOT / 'matrix.json')[0]
    data = Training(selected(job)[:8], 0, 20, protocol['training'])
    model, _, _ = initialise(0, 'pretrained')
    scratch, _, _ = initialise(0, 'scratch')
    assert torch.equal(model.head.weight, scratch.head.weight)
    assert torch.equal(model.head.bias, scratch.head.bias)
    del scratch
    r, f, p = data.cars[0]
    ix = np.arange(20)
    a = fb.encoder_inputs(p, len(f.points), ix, f.values[ix])
    corrupted = f.values.copy()
    corrupted[20:] += 1000
    b = fb.encoder_inputs(p, len(f.points), ix, corrupted[ix])
    assert all(np.array_equal(a[k], b[k]) for k in a)
    sample = data[0]
    # A saved-state restart must reproduce every draw without worker RNG state.
    data.offset = 8
    resumed = data[0]
    data.offset = 0
    assert all(np.array_equal(resumed[k], data[8][k]) for k in resumed)
    batch = fb.to_device(collate_field([sample] * 8), 'cuda')
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(12):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = field_loss(model, batch, .25)
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    evaluate(model, problems('val')[:1], data.meta)
    write(ROOT / 'check.json', dict(passed=True, paired_new_head_equal=True, hidden_label_input_invariance=True,
          deterministic_resume_draw=True, finite_gpu_forward_backward=True, losses=losses,
          device=torch.cuda.get_device_name()))
    print('CHECK_PASSED', losses, flush=True)


def train(index):
    protocol = read(ROOT / 'protocol.json')
    spec = protocol['training']
    job = read(ROOT / 'matrix.json')[index]
    out = ROOT / 'runs' / job['name']
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        print('ALREADY_COMPLETE', job['name'], flush=True)
        return
    random.seed(job['seed'])
    np.random.seed(job['seed'])
    started = time.monotonic()
    data = Training(selected(job), job['seed'], spec['updates'], spec)
    exposure_counts = np.bincount(data.order[:spec['updates'] * spec['batch_size']] % data.round_size,
                                 minlength=data.round_size)
    assert (exposure_counts > 0).all(), 'Not every budgeted simulation was used'
    assert exposure_counts.max() - exposure_counts.min() <= 1, 'Unbalanced simulation exposures'
    evaluation = problems('val')
    model, _, state = initialise(job['seed'], job['arm'])
    meta = dict(data.meta, job=job, subset=read(ROOT / 'subsets.json')[f'{job["seed"]}/{job["budget"]}'])
    opt = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    schedule = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=spec['learning_rate'], total_steps=spec['updates'],
                                                 pct_start=.1, anneal_strategy='cos')
    history, best, update, elapsed = [], {0: math.inf, 20: math.inf}, 0, 0.
    if (out / 'last.pt').exists():
        resume = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        assert resume['meta'] == meta
        model.load_state_dict(resume['model'])
        opt.load_state_dict(resume['optimizer'])
        schedule.load_state_dict(resume['scheduler'])
        history, best, update, elapsed = resume['history'], resume['best'], resume['update'], resume['elapsed']
        torch.set_rng_state(resume['torch_rng'])
        torch.cuda.set_rng_state_all(resume['cuda_rng'])
    data.offset = update * spec['batch_size']
    loader = DataLoader(data, batch_size=spec['batch_size'], shuffle=False, num_workers=4,
                        collate_fn=collate_field, persistent_workers=True, pin_memory=True,
                        generator=torch.Generator().manual_seed(15000 + job['seed']))
    torch.cuda.reset_peak_memory_stats()
    losses = []
    print('START', job, 'resume_update', update, flush=True)
    for batch in loader:
        model.train()
        batch = fb.to_device(batch, 'cuda')
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = field_loss(model, batch, spec['visible_loss_weight'])
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at update {update}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), spec['gradient_clip'], error_if_nonfinite=True)
        opt.step()
        schedule.step()
        losses.append(loss.item())
        update += 1
        if update % spec['validation_every'] == 0:
            scores = evaluate(model, evaluation, meta)
            values = {k: mean_primary(scores, k) for k in (0, 20)}
            value = values[0]
            seconds = elapsed + time.monotonic() - started
            history.append(dict(update=update, train_loss=float(np.mean(losses)), val_nrmse=value, val_nrmse20=values[20],
                                seconds=seconds, learning_rate=schedule.get_last_lr()[0]))
            losses = []
            for k in (0, 20):
                if values[k] < best[k]:
                    best[k] = values[k]
                    save_field_encoder(out / f'best_s{k}.pt', model, config=state['config'], benchmark=meta, update=update)
                    write(out / f'best_val_s{k}.json', scores)
            if update in (2000, 4000, 8000):
                save_field_encoder(out / f'update_{update}.pt', model, config=state['config'], benchmark=meta, update=update)
                write(out / f'val_{update}.json', scores)
            payload = dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=schedule.state_dict(),
                           history=history, best=best, update=update, elapsed=seconds, meta=meta,
                           torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
            torch.save(payload, out / 'last.pt.tmp')
            (out / 'last.pt.tmp').replace(out / 'last.pt')
            write(out / 'history.json', history)
            print(job['name'], update, 'val', round(value, 5), 'val20', round(values[20], 5), 'best', best,
                  'seconds', round(seconds), flush=True)
    assert update == spec['updates']
    improvements, selected_updates = {}, {}
    for k in (0, 20):
        column = 'val_nrmse' if k == 0 else 'val_nrmse20'
        last = np.mean([h[column] for h in history[-2:]])
        previous = np.mean([h[column] for h in history[-4:-2]])
        improvements[str(k)] = float((previous - last) / previous)
        selected_updates[str(k)] = min(history, key=lambda h: h[column])['update']
    write(out / 'complete.json', dict(**job, updates=update, unique_training_simulations=job['budget'],
          unique_training_geometries=len({r['family'] for r in selected(job)}),
          actual_simulations_exposed=int((exposure_counts > 0).sum()),
          per_simulation_exposures=[int(exposure_counts.min()), int(exposure_counts.max())],
          field_example_exposures=update * spec['batch_size'], normalisation=data.meta,
          best_val_nrmse=best, selected_update=selected_updates,
          trailing_relative_improvement=improvements,
          budget_limited={k: v > .01 for k, v in improvements.items()},
          wall_seconds=elapsed + time.monotonic() - started,
          peak_gpu_bytes=torch.cuda.max_memory_allocated(), device=torch.cuda.get_device_name(),
          torch=torch.__version__, python=platform.python_version(), completed_utc=fb.now()))
    print('COMPLETE', job['name'], flush=True)


def score(index):
    job = read(ROOT / 'matrix.json')[index]
    out = ROOT / 'runs' / job['name']
    assert (out / 'complete.json').exists(), f'{job["name"]} has not finished training'
    if (out / 'test.json').exists():
        return
    evaluation, scores = problems('test'), []
    for k in (0, 20):
        model, _, state = fb.load_field_encoder(out / f'best_s{k}.pt', 'cuda')
        rows = evaluate(model, evaluation, state['benchmark'], out / f'predictions_s{k}', save_sensor=k)
        scores.extend(r for r in rows if r['sensors'] == k)
        del model
    write(out / 'test.json', dict(job=job, rows=scores, evaluated_utc=fb.now()))
    print('TEST_COMPLETE', job['name'], {k: mean_primary(scores, k) for k in (0, 20)}, flush=True)


def main():
    os.umask(0o077)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.enable_cudnn_sdp(False)
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'check', 'train', 'score'])
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    if args.command in ('train', 'score'):
        globals()[args.command](args.index)
    else:
        globals()[args.command]()


if __name__ == '__main__':
    main()
