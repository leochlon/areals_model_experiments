import dataclasses
from functools import lru_cache
import json
import os
from pathlib import Path
import sys

os.umask(0o077)
ROOT = Path(__file__).resolve().parent
HOME_RUNS = Path('/workspace/runs')
BASE = Path('/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/run/best_normal.pt')
EXPANDED = HOME_RUNS / 'areals_existing_data_expansion_20261025'
NINE = HOME_RUNS / 'areals_multicondition_wings_20261023'
sys.path[:0] = [str(ROOT / 'source'), str(ROOT / 'source/areal_src'), str(ROOT / 'source/geometry_src')]

import numpy as np
import torch
from torch import nn
import field_benchmark as fb
from areal_tokens.config import ExperimentConfig
from areal_tokens.field import FieldEncoder, collate_field, field_loss, save_field_encoder
from areal_tokens.model import GeometryAutoencoder

DATASETS = ['nine_angle', 'doubledelta', 'blendednet', 'drivaerml', 'drivaernet', 'reserved_cars']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


@lru_cache(maxsize=None)
def dataset(name):
    return read(ROOT / 'data' / name / 'dataset.json')


def original_bench(name):
    if name == 'nine_angle':
        return NINE / 'bench'
    if name == 'reserved_cars':
        return HOME_RUNS / 'areals_fresh_car_holdout_20261024/benches/completion'
    return EXPANDED / 'datasets' / name / 'bench'


def records(name, split=None):
    rows = read(ROOT / 'data' / name / 'manifest.json')['records']
    return [r for r in rows if split is None or r['split'] == split]


def field(name, record):
    folder = Path(record['field_path']).parent if 'field_path' in record else dataset(name)['fields']
    return fb.load_field(folder, record['id'])


def prepared(name, record):
    with np.load(ROOT / 'cache' / name / (record['cache_key'] + '.npz')) as a:
        return fb.Prepared({k: a['a_' + k].copy() for k in fb.CLOUD_KEYS if 'a_' + k in a},
                          a['sample_to_field'].copy(), a['field_to_sample'].copy(), a['anchor_index'].copy())


@lru_cache(maxsize=None)
def conditions(name):
    return read(ROOT / 'data' / name / 'conditions.json')


class ConditionField(FieldEncoder):
    def __init__(self, base, dimensions):
        super().__init__(base)
        self.condition_projection = nn.Sequential(nn.Linear(dimensions, base.config.width), nn.SiLU(),
                                                  nn.Linear(base.config.width, base.config.width))
        nn.init.zeros_(self.condition_projection[-1].weight)
        nn.init.zeros_(self.condition_projection[-1].bias)

    def encode(self, batch, *channels):
        geometry = super().encode(batch, *channels)
        delta = self.condition_projection(batch['condition'])[:, None]
        return dataclasses.replace(geometry, tokens=geometry.tokens + delta)


def create_model(base, name, role):
    if role == 'conditioned' and name != 'nine_angle':
        return ConditionField(base, len(conditions(name)['names']))
    return FieldEncoder(base)


def initialise(name, role, seed, arm='scratch'):
    torch.manual_seed(seed)
    base, experiment, state = fb.load_model(BASE, 'cpu')
    if arm == 'scratch':
        base = GeometryAutoencoder(experiment.model)
    torch.manual_seed(57300 + seed)
    model = create_model(base, name, role)
    return model, experiment, state


def load_checkpoint(path, name, role, device='cuda'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    experiment = ExperimentConfig.from_dict(state['config'])
    model = create_model(GeometryAutoencoder(experiment.model), name, role)
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval(), experiment, state


def input_sample(name, role, p, count, index, values, meta, condition=None):
    if role == 'measurements':
        return fb.encoder_inputs(p, count, index, (values - meta['mean']) / meta['std'])
    empty = np.empty(0, np.int64)
    sample = fb.encoder_inputs(p, count, empty, np.empty(0))
    if role == 'conditioned':
        stats = conditions(name)
        c = (np.asarray(condition) - stats['mean']) / stats['std']
        if name == 'nine_angle':
            for key in ['values', 'anchor_values']:
                sample[key] = np.full_like(sample[key], float(c[0]))
            for key in ['present', 'anchor_present']:
                sample[key] = np.ones_like(sample[key])
        else:
            sample['condition'] = c.astype(np.float32)
    return sample


def collate(samples, targets=False):
    batch = collate_field(samples) if targets else fb.collate_inputs(samples)
    if 'condition' in samples[0]:
        batch['condition'] = torch.from_numpy(np.stack([s['condition'] for s in samples]))
    return batch


def predict(model, name, role, f, p, observations, meta, condition=None):
    samples = [input_sample(name, role, p, len(f.points), ix, val, meta, condition) for ix, val in observations]
    if role != 'measurements':
        samples = samples[:1]
    batch = fb.to_device(collate(samples), 'cuda')
    queries = torch.as_tensor(f.points, dtype=torch.float32, device='cuda')[None].expand(len(samples), -1, -1)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        g = model.encode(batch, batch['values'], batch['present'], batch['anchor_values'], batch['anchor_present'])
        out = model.predict(g, queries).float().cpu().numpy().astype(np.float64)
    out = out * meta['std'] + meta['mean']
    if role != 'measurements':
        out = np.repeat(out, len(observations), axis=0)
    assert np.isfinite(out).all()
    return out


def metrics(f, ix, prediction):
    hidden = np.ones(len(f.values), dtype=bool)
    hidden[ix] = False
    w = np.asarray(f.weights, np.float64)
    y = np.asarray(f.values, np.float64)
    mu = np.sum(w * y) / w.sum()
    variance = np.sum(w * (y - mu) ** 2) / w.sum()
    mse = np.sum(w[hidden] * (prediction[hidden] - y[hidden]) ** 2) / w[hidden].sum()
    result = fb.problem_metrics(f, ix, prediction)
    independent = float(np.sqrt(mse / variance))
    assert abs(independent - result['nrmse']) < 1e-10
    if f.normals is not None:
        completed = np.asarray(prediction, np.float64).copy()
        completed[ix] = y[ix]
        force = np.sum((w * y)[:, None] * f.normals, axis=0)
        force_error = np.sum((w * (completed - y))[:, None] * f.normals, axis=0)
        if np.linalg.norm(force) > 0:
            independent_force = float(np.linalg.norm(force_error) / np.linalg.norm(force))
            assert abs(independent_force - result['force_rel']) < 1e-9
    result.update(nrmse=independent, nmse=mse / variance)
    return result


def legacy_checkpoint(name,role,seed=0):
    if name=='nine_angle':
        return NINE/'bench/models'/role/'field_best.pt'
    if name=='reserved_cars':
        if role=='measurements':
            return HOME_RUNS/'areals_architecture_controls_20261024/replication/bench/models'/f'pretrained_seed{seed}'/'field_best.pt'
        return None
    arm=dict(measurements='pretrained',geometry='shape',conditioned='conditioned')[role]
    return EXPANDED/'datasets'/name/'models'/f'{arm}_seed{seed}'/'field_best.pt'


def historical_checkpoint(name,role,seed):
    if name=='nine_angle':
        if role=='measurements':
            return HOME_RUNS/'areals_nine_angle_seeds_20261029/bench/models'/f'measurements_seed{seed}'/'field_best.pt'
        if seed:
            return ROOT/'legacy_references/nine_angle'/role/f'seed{seed}/field_best.pt'
    if name=='reserved_cars' and role=='geometry':
        return None
    return legacy_checkpoint(name,role,seed)


def setup_torch():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.enable_cudnn_sdp(False)
