import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from architecture import install, latent_encode
from areal_tokens.config import ExperimentConfig
from areal_tokens.field import FieldEncoder as NativeField
from areal_tokens.model import GeometryAutoencoder
from position_control import strip_rotary

sys.path.insert(0, '/workspace/runs/field_benchmark_4gpu_20261023')
import field_benchmark as fb

ARM, SEED = None, None
NativeCompletion, NativeAreal = fb.Completion, fb.Areal


def base_model(config, seed):
    original = {'anchored_no_rotary': 'reference', 'unanchored_no_rotary': 'unanchored_latents'}.get(ARM, ARM)
    model = install(GeometryAutoencoder(config.model), original, seed)
    return strip_rotary(model) if ARM in ['anchored_no_rotary', 'unanchored_no_rotary'] else model


def geometry_load(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    assert state['ablation'] == ARM
    config = ExperimentConfig.from_dict(state['config'])
    base = base_model(config, state['seed'])
    base.load_state_dict(state['model'], strict=True)
    return base.to(device).eval(), config, state


class Field(NativeField):
    def __init__(self, base):
        super().__init__(base)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(57300 + SEED)
            self.head.reset_parameters()

    def encode(self, batch, values, present, anchor_values, anchor_present):
        if ARM in ['unanchored_latents', 'unanchored_no_rotary']:
            return latent_encode(self.base.encoder, batch, self._channels(values, present))
        return super().encode(batch, values, present, anchor_values, anchor_present)


def field_load(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    config = ExperimentConfig.from_dict(state['config'])
    model = Field(base_model(config, SEED))
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval(), config, state


class Completion(NativeCompletion):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        for key in ['values', 'present', 'anchor_values', 'anchor_present']:
            sample[key] = np.zeros_like(sample[key])
        sample['hidden'] = np.ones_like(sample['hidden'])
        return sample


class Areal(NativeAreal):
    def predict_car(self, surface, observed):
        empty = fb.Observed('geometry-only', {'name': 'unobserved', 'kind': 'none'},
                            np.empty(0, np.int64), np.empty(0))
        prediction = super().predict_car(surface, [empty])[0]
        return [prediction.copy() for _ in observed]


def selection(config, records, limit):
    settings = dict(name='sensors-5', kind='sensors', count=5)
    result = []
    for record in records[:16]:
        field = fb.load_field(config['fields'], record['id'])
        rng = np.random.default_rng(fb.stable_seed(240024, record['id'], 0))
        index = np.sort(rng.choice(len(field.values), 5, replace=False, p=field.weights / field.weights.sum()))
        observed = fb.Observed(record['id'], settings, index, field.values[index])
        result.append((fb.surface_of(record, field), field, [observed]))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--arm', required=True)
    p.add_argument('--seed', required=True, type=int)
    args, remaining = p.parse_known_args()
    ARM, SEED = args.arm, args.seed
    fb.load_model, fb.load_field_encoder, fb.FieldEncoder = geometry_load, field_load, Field
    fb.Completion, fb.Areal, fb.selection_problems = Completion, Areal, selection
    fb.main(remaining)
