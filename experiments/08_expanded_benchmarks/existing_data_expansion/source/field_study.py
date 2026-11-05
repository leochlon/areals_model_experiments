import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

import field_benchmark as fb
from areal_tokens.config import ExperimentConfig
from areal_tokens.model import GeometryAutoencoder
from util import BASE, ROOT, write

NativeCompletion, NativeAreal, NativeField = fb.Completion, fb.Areal, fb.FieldEncoder
native_prepare, native_rng = fb.prepare_car, fb.observation_rng
native_collate = fb.collate_field
ROLE, DATA, CONDITIONS, STATS, RECORDS = None, None, None, None, None
PREPARED = {}


def condition(identity):
    assert ROLE == 'conditioned'
    return ((np.asarray(CONDITIONS['cases'][identity]) - STATS['mean']) / STATS['std']).astype(np.float32)


class ConditionField(NativeField):
    def __init__(self, base):
        super().__init__(base)
        self.condition_projection = nn.Sequential(nn.Linear(len(STATS['mean']), base.config.width), nn.SiLU(),
                                                  nn.Linear(base.config.width, base.config.width))
        nn.init.zeros_(self.condition_projection[-1].weight)
        nn.init.zeros_(self.condition_projection[-1].bias)

    def encode(self, batch, *channels):
        geometry = super().encode(batch, *channels)
        delta = self.condition_projection(batch['condition'])[:, None]
        return dataclasses.replace(geometry, tokens=geometry.tokens + delta)


def conditioned_collate(samples):
    batch = native_collate(samples)
    batch['condition'] = torch.from_numpy(np.stack([s['condition'] for s in samples]))
    return batch


class Completion(NativeCompletion):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if ROLE != 'measurements':
            for key in ['values', 'present', 'anchor_values', 'anchor_present']:
                sample[key] = np.zeros_like(sample[key])
            sample['hidden'] = np.ones_like(sample['hidden'])
        if ROLE == 'conditioned':
            sample['condition'] = condition(sample['id'])
        return sample


class Areal(NativeAreal):
    def __init__(self, name, model, experiment, meta, device, batch_size, amp, source=None):
        if 'input_role' in meta:
            assert meta['input_role'] == ROLE
            assert meta['dataset'] == DATA.name
        meta.update(input_role=ROLE, dataset=DATA.name)
        if ROLE == 'conditioned':
            meta['condition_statistics'] = STATS
        super().__init__(name, model, experiment, meta, device, batch_size, amp, source)

    @torch.no_grad()
    def predict_car(self, surface, observed):
        if ROLE == 'measurements':
            return super().predict_car(surface, observed)
        empty = fb.Observed('shape', fb.BLIND, np.empty(0, np.int64), np.empty(0))
        if ROLE == 'geometry':
            result = super().predict_car(surface, [empty])[0]
        else:
            self.model.eval()
            prepared = fb.prepare_car(surface.record, surface.points, self.experiment)
            sample = fb.encoder_inputs(prepared, len(surface.points), empty.index, empty.values)
            batch = fb.to_device(fb.collate_inputs([sample]), self.device)
            batch['condition'] = torch.from_numpy(condition(surface.id)[None]).to(self.device)
            queries = torch.as_tensor(surface.points, dtype=torch.float32, device=self.device)[None]
            with self.amp():
                geometry = self.model.encode(batch, batch['values'], batch['present'],
                                             batch['anchor_values'], batch['anchor_present'])
                result = self.model.predict(geometry, queries)[0].float().cpu().numpy().astype(np.float64)
            result = result * self.meta['std'] + self.meta['mean']
        return [result.copy() for _ in observed]


def load_field(path, device):
    state = torch.load(path, map_location='cpu', weights_only=True)
    assert state['kind'] == 'field_encoder'
    config = ExperimentConfig.from_dict(state['config'])
    model = ConditionField(GeometryAutoencoder(config.model))
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval(), config, state


def configure(dataset, role):
    global ROLE, DATA, CONDITIONS, STATS, RECORDS
    ROLE, DATA = role, ROOT / 'datasets' / dataset
    assert json.loads((DATA / 'data_checks.json').read_text())['passed']
    records, _ = fb.read_manifest(DATA / 'manifest.json')
    RECORDS = {r['id']: r for r in records}
    if role == 'conditioned':
        CONDITIONS = json.loads((DATA / 'conditions.json').read_text())
        STATS = json.loads((DATA / 'condition_statistics.json').read_text())
        assert STATS['fitted_on'] == 'training records only'
        fb.FieldEncoder, fb.load_field_encoder, fb.collate_field = ConditionField, load_field, conditioned_collate

    def prepare(record, points, experiment):
        digest = hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest()
        key = (record['geometry_id'], digest, experiment.model.latents, experiment.data.encoder_points)
        if key not in PREPARED:
            PREPARED[key] = native_prepare(record, points, experiment)
        return PREPARED[key]

    def observation_rng(seed, purpose, identity, setting, repeat):
        identity = RECORDS.get(identity, {}).get('geometry_id', identity)
        return native_rng(seed, purpose, identity, setting, repeat)

    class LazyPatches(fb.Patches):
        def __init__(self, points, weights, counts=()):
            super().__init__(points, weights, ())

    fb.prepare_car, fb.observation_rng = prepare, observation_rng
    fb.Completion, fb.Areal, fb.Patches = Completion, Areal, LazyPatches


def setup(dataset):
    configure(dataset, 'measurements')
    for split in ['val', 'test']:
        fb.cmd_observations(argparse.Namespace(manifest=str(DATA / 'manifest.json'), fields=str(DATA / 'fields'),
            bench=str(DATA / 'bench'), split=split, patch_counts=[64], hidden=[], sensors=[5, 20, 100, 500],
            unobserved=True, seed=0, repeats=1))
    write(DATA / 'observations_ready.json', dict(passed=True))


def check_setup(dataset):
    """Hidden targets cannot reach the inputs, wrapping keeps the geometry weights, and a few steps train."""
    configure(dataset, 'measurements')
    torch.manual_seed(0)
    base, experiment, _ = fb.load_model(BASE, 'cpu')
    records = fb.records_of(fb.bench_config(DATA / 'bench'), 'train')[:8]
    config = fb.bench_config(DATA / 'bench')
    data = Completion(records, config['fields'], experiment, config['protocol'], 256, 0)
    mean, std = fb.training_statistics([f for _, f, _, _ in data.cars])
    data.meta = dict(normalise='dataset', mean=mean, std=std, floor=.5, mix=dict(unobserved=.1, sensors=.9))
    data.set_epoch(1)
    original = data[0]
    record, field, prepared, _ = data.cars[0]
    ix = np.sort(np.random.default_rng(0).choice(len(field.points), 20, replace=False))
    inputs = fb.encoder_inputs(prepared, len(field.points), ix, field.values[ix])
    changed = field.values.copy()
    hidden = np.ones(len(changed), bool)
    hidden[ix] = False
    changed[hidden] += 1000
    altered = fb.encoder_inputs(prepared, len(field.points), ix, changed[ix])
    for key in ['values', 'present', 'anchor_values', 'anchor_present']:
        assert np.array_equal(inputs[key], altered[key]), key
    initial_parameters = {k: v.clone() for k, v in base.state_dict().items()}
    torch.manual_seed(57300)
    model = NativeField(base).cuda().train()
    for k, before in initial_parameters.items():
        after = model.base.state_dict()[k].cpu()
        if k == 'encoder.embed.0.weight':
            assert torch.equal(before, after[:, :-2]) and not after[:, -2:].any()
        else:
            assert torch.equal(before, after)
    batch = fb.to_device(native_collate([data[i] for i in range(len(data))]), torch.device('cuda'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        loss = fb.field_loss(model, batch, .25)
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0], losses
    write(DATA / 'setup_checks.json', dict(passed=True, losses=losses, checkpoint=str(BASE), batch_size=len(data)))
    print('setup checks passed', dataset, losses, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--role', choices=['measurements', 'geometry', 'conditioned'], default='measurements')
    args, remainder = parser.parse_known_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.enable_cudnn_sdp(False)
    if remainder == ['setup']:
        return setup(args.dataset)
    if remainder == ['check-setup']:
        return check_setup(args.dataset)
    configure(args.dataset, args.role)
    fb.main(remainder)


if __name__ == '__main__':
    main()
