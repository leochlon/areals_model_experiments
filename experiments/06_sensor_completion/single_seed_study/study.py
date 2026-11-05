import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import field_benchmark as fb

ROOT = Path('/workspace/runs/areals_multicondition_wings_20261023')
BASE = Path('/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/run/best_normal.pt')
NativeCompletion, NativeAreal = fb.Completion, fb.Areal
native_prepare, native_rng, native_seed = fb.prepare_car, fb.observation_rng, fb.stable_seed
NativePatches = fb.Patches
ROLE, REF, RECORDS = None, None, None
PREPARED, PATCHES = {}, {}


def condition_inputs(prepared, value):
    sample = {key: prepared.arrays[key] for key in fb.CLOUD_KEYS if key in prepared.arrays}
    n, k = len(prepared.arrays['points']), len(prepared.arrays['anchors'])
    sample.update(values=np.full((n, 1), value, np.float32), present=np.ones((n, 1), np.float32),
                  anchor_values=np.full((k, 1), value, np.float32), anchor_present=np.ones((k, 1), np.float32))
    return sample


def angle(identity):
    assert ROLE == 'conditioned'
    return (REF['cases'][identity]['aoa'] - REF['mean']) / REF['std']


class Completion(NativeCompletion):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if ROLE == 'measurements':
            return sample
        record, field, prepared, _ = self.cars[index]
        if ROLE == 'conditioned':
            sample.update(condition_inputs(prepared, angle(record['id'])))
        else:
            for key in ['values', 'present', 'anchor_values', 'anchor_present']:
                sample[key] = np.zeros_like(sample[key])
        sample['hidden'] = np.ones_like(sample['hidden'])
        return sample


class Areal(NativeAreal):
    def __init__(self, name, model, experiment, meta, device, batch_size, amp, source=None):
        if meta.get('input_role') is not None:
            assert meta['input_role'] == ROLE, (meta['input_role'], ROLE)
        meta.update(input_role=ROLE)
        if ROLE == 'conditioned':
            meta.update(condition_mean=REF['mean'], condition_std=REF['std'])
        super().__init__(name, model, experiment, meta, device, batch_size, amp, source)

    @torch.no_grad()
    def predict_car(self, surface, observed):
        if ROLE == 'measurements':
            return super().predict_car(surface, observed)
        if ROLE == 'geometry':
            empty = fb.Observed('shape', fb.BLIND, np.empty(0, np.int64), np.empty(0))
            result = super().predict_car(surface, [empty])[0]
        else:
            self.model.eval()
            prepared = fb.prepare_car(surface.record, surface.points, self.experiment)
            sample = condition_inputs(prepared, angle(surface.id))
            batch = fb.to_device(fb.collate_inputs([sample]), self.device)
            queries = torch.as_tensor(surface.points, dtype=torch.float32, device=self.device)
            with self.amp():
                geometry = self.model.encode(batch, batch['values'], batch['present'],
                                             batch['anchor_values'], batch['anchor_present'])
                prediction = self.model.predict(geometry, queries[None])
            result = prediction[0].float().cpu().numpy().astype(np.float64) * self.meta['std'] + self.meta['mean']
        return [result.copy() for _ in observed]


def configure(role):
    global ROLE, REF, RECORDS
    ROLE = role
    records, _ = fb.read_manifest(ROOT / 'manifest.json')
    RECORDS = {r['id']: r for r in records}
    REF = json.loads((ROOT / 'conditions_reference_only.json').read_text()) if role == 'conditioned' else None
    def prepare(record, points, experiment):
        key = (record['grid_id'], experiment.model.latents, experiment.data.encoder_points)
        if key not in PREPARED:
            PREPARED[key] = native_prepare(record, points, experiment)
        return PREPARED[key]
    def seed(number, identity, epoch=0):
        case, separator, suffix = identity.partition(':')
        if case in RECORDS:
            identity = RECORDS[case]['grid_id'] + separator + suffix
        return native_seed(number, identity, epoch)
    def observation_rng(number, purpose, identity, setting, repeat):
        grid = RECORDS[identity]['grid_id'] if identity in RECORDS else identity
        return native_rng(number, purpose, grid, setting, repeat)
    class Patches(NativePatches):
        def assignment(self, count):
            if count not in self.cells:
                fingerprint = hashlib.sha256(self.points.tobytes() + self.weights.tobytes()).hexdigest()
                key = (fingerprint, count)
                if key not in PATCHES:
                    PATCHES[key] = super().assignment(count)
                self.cells[count] = PATCHES[key]
            return self.cells[count]
    fb.prepare_car, fb.stable_seed, fb.observation_rng = prepare, seed, observation_rng
    fb.Completion, fb.Areal, fb.Patches = Completion, Areal, Patches
    fb.selection_problems = selection_problems


def selection_problems(config, records, limit):
    families = sorted({r['family'] for r in records})
    chosen = set(np.random.default_rng(0).choice(families, min(4, len(families)), replace=False))
    selected = [r for r in records if r['family'] in chosen]
    if ROLE == 'measurements':
        settings = [dict(name='sensors-20', kind='sensors', count=20),
                    dict(name='sensors-500', kind='sensors', count=500),
                    dict(name='patch-64-0.60', kind='patch', count=64, fraction=.6)]
    else:
        # The predictor ignores these five values; a nonempty set also keeps the native copy control defined.
        settings = [dict(name='sensors-5', kind='sensors', count=5)]
    problems = []
    for record in selected:
        field = fb.load_field(config['fields'], record['id'])
        patches = fb.Patches(field.points, field.weights, [64])
        observed = []
        for setting in settings:
            rng = fb.observation_rng(config['seed'], 'select', record['id'], setting['name'], 0)
            index = fb.draw_observed(setting, patches, field.weights, rng)
            observed.append(fb.Observed(record['id'] + setting['name'], setting, index, field.values[index].copy()))
        problems.append((fb.surface_of(record, field), field, observed))
    return problems


def check_setup():
    config = fb.bench_config(ROOT / 'bench')
    records = fb.records_of(config)
    assert len(records) == 2448
    families = {}
    grids = {}
    for record in records:
        families.setdefault(record['family'], set()).add(record['split'])
        grids.setdefault(record['grid_id'], []).append(record)
    assert all(len(v) == 1 for v in families.values())
    assert all(len(v) == 9 for v in grids.values())
    for members in grids.values():
        assert len({Path(r['path']).stat().st_ino for r in members}) == 1
        first = fb.load_field(config['fields'], members[0]['id'])
        first_masks = {}
        for record in members:
            field = fb.load_field(config['fields'], record['id'])
            assert np.array_equal(first.points, field.points)
            assert np.array_equal(first.weights, field.weights)
            assert np.array_equal(first.normals, field.normals)
            patches = fb.Patches(field.points, field.weights, [64])
            for setting in [dict(name='sensors-20', kind='sensors', count=20),
                            dict(name='patch-64-0.60', kind='patch', count=64, fraction=.6)]:
                ix = fb.draw_observed(setting, patches, field.weights,
                                      fb.observation_rng(0, 'check', record['id'], setting['name'], 0))
                if setting['name'] in first_masks:
                    assert np.array_equal(ix, first_masks[setting['name']])
                first_masks[setting['name']] = ix
    print('data checks passed: family split, common coordinates/normals/weights, hard links, shared masks', flush=True)
    checks, initial = [], None
    train = [r for r in records if r['split'] == 'train'][:8]
    for role in ['geometry', 'measurements', 'conditioned']:
        configure(role)
        torch.manual_seed(0)
        base, experiment, state = fb.load_model(BASE, 'cpu')
        data = Completion(train, config['fields'], experiment, config['protocol'], 256, 0)
        mean, std = fb.training_statistics([f for _, f, _, _ in data.cars])
        data.meta = dict(normalise='dataset', mean=mean, std=std, floor=.5,
                         mix=dict(unobserved=.1, sensors=.45))
        data.set_epoch(1)
        sample = data[0]
        record, field, _, patches = data.cars[0]
        rng = np.random.default_rng(fb.stable_seed(0, record['id'] + ':train', 1))
        setting = fb.training_setting(rng, config['protocol'], data.meta['mix'])
        observed = fb.draw_observed(setting, patches, field.weights, rng)
        hidden = np.ones(len(field.values), bool)
        hidden[observed] = False
        old = field.values.copy()
        field.values[hidden] += 1000
        altered = data[0]
        field.values[:] = old
        for key in ['values','present','anchor_values','anchor_present']:
            assert np.array_equal(sample[key], altered[key]), (role, key)
        if role == 'geometry':
            assert not sample['present'].any() and not sample['values'].any()
        if role == 'conditioned':
            assert np.all(sample['present'] == 1)
            assert np.all(sample['values'] == np.float32(angle(record['id'])))
        batch = fb.to_device(fb.collate_field([data[i] for i in range(8)]), torch.device('cuda'))
        model = fb.FieldEncoder(base).cuda().train()
        weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        initial = initial or weights
        assert all(torch.equal(initial[k], v) for k, v in weights.items()), 'roles start from different weights'
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = []
        for _ in range(10):
            optimizer.zero_grad(set_to_none=True)
            loss = fb.field_loss(model, batch, .25)
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert torch.isfinite(norm)
            optimizer.step()
            losses.append(float(loss.detach()))
        assert losses[-1] < losses[0], (role, losses)
        checks.append(dict(role=role, loss_initial=losses[0], loss_after_10=losses[-1]))
        print(json.dumps(checks[-1]), flush=True)
        del model, optimizer, batch, data, base
        torch.cuda.empty_cache()
    fb.save_json(ROOT / 'setup_checks.json', dict(passed=True, all_cases_checked=len(records), roles=checks))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=['geometry','measurements','conditioned'], default='measurements')
    args, remainder = parser.parse_known_args()
    configure(args.role)
    if remainder == ['check-setup']:
        check_setup()
    else:
        fb.main(remainder)


if __name__ == '__main__':
    main()
