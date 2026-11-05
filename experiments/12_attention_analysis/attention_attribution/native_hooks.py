"""Native, unobserved DoubleDelta adapters for the supplied theory exporter."""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'native'))
import field_benchmark as fb
from areal_tokens.data import read_manifest, to_device
from areal_tokens.field import FieldEncoder, load_field_encoder
from areal_tokens.io import load_model as load_geometry
from export_attention import softmax_attention

AUDIT = {'models': {}, 'cases': [], 'checks': {}}
STATE = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, a.dtype)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def configure(manifest, fields, out):
    records, meta = read_manifest(manifest)
    records = sorted((r for r in records if r['split'] == 'test'), key=lambda r: r['id'])
    counts = Counter(r['family'] for r in records)
    assert len(records) == 144 and len(counts) == 16 and set(counts.values()) == {9}, counts
    STATE.update(records=records, fields=Path(fields), out=Path(out), prepared={}, group_inputs={})
    STATE['out'].mkdir(parents=True, exist_ok=True)
    AUDIT['manifest'] = meta
    AUDIT['group_counts'] = dict(counts)
    AUDIT['environment'] = {'torch': str(torch.__version__), 'numpy': np.__version__,
                            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name()}
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class AttentionCapture:
    def __init__(self, blocks):
        self.enabled = False
        self.rows = [[] for _ in blocks]
        self.calls = 0
        self.max_output_error = 0.0
        self.max_row_error = 0.0
        self.numpy_error = None
        self.handles = []
        for index, block in enumerate(blocks):
            module = block.attention
            saved = {}

            def before(mod, args, saved=saved):
                if self.enabled:
                    saved.clear()
                    saved['rotary'] = []
                    saved['weights'] = args[4].detach()

            def rotary(mod, args, output, saved=saved):
                if self.enabled:
                    saved['rotary'].append(output.detach())

            def kv(mod, args, output, saved=saved):
                if self.enabled:
                    saved['v'] = output.detach().chunk(2, dim=-1)[1]

            def after(mod, args, output, saved=saved, index=index):
                if not self.enabled:
                    return
                assert len(saved['rotary']) == 2
                q, k = saved['rotary']
                w = saved['weights']
                log_w = torch.where(w > 0, w.float().clamp_min(1e-38).log(), -torch.inf)[:, None, None]
                att = softmax_attention(q, k, log_w.to(q.dtype))
                assert torch.isfinite(att).all() and (att >= 0).all()
                row_error = float((att.sum(-1) - 1).abs().max())
                assert row_error < 2e-6, row_error
                v = saved['v'].unflatten(-1, (mod.heads, mod.head_dim)).transpose(1, 2)
                direct = mod.out((att @ v).transpose(1, 2).flatten(-2))
                error = float((direct - output).abs().max())
                assert error < 2e-5 * max(1.0, float(output.abs().max())), error
                if self.numpy_error is None:
                    qn, kn, wn = (a.float().cpu().numpy() for a in (q[:, :, :8], k, log_w))
                    an = softmax_attention(qn, kn, wn)
                    self.numpy_error = float(np.abs(an - att[:, :, :8].cpu().numpy()).max())
                    assert self.numpy_error < 2e-6, self.numpy_error
                self.max_output_error = max(self.max_output_error, error)
                self.max_row_error = max(self.max_row_error, row_error)
                self.calls += 1
                self.rows[index].append(att[0].float().cpu().numpy())

            self.handles.extend([module.register_forward_pre_hook(before),
                                 module.rotary.register_forward_hook(rotary),
                                 module.kv.register_forward_hook(kv), module.register_forward_hook(after)])

    def begin(self):
        self.rows = [[] for _ in self.rows]
        self.enabled = True

    def end(self):
        self.enabled = False
        return np.stack([np.concatenate(layer, axis=1) for layer in self.rows])

    def summary(self):
        return dict(calls=self.calls, max_output_error=self.max_output_error,
                    max_row_sum_error=self.max_row_error, numpy_torch_max_error=self.numpy_error)


@dataclass
class Model:
    name: str
    path: str
    net: object
    experiment: object
    meta: dict
    device: object
    encoder_capture: object
    decoder_capture: object
    cache: dict
    predictions: dict


@torch.inference_mode()
def load_model(path, device='cuda'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('kind') == 'field_encoder':
        net, experiment, state = load_field_encoder(path, device)
        meta = state['benchmark']
        name = 'scratch' if meta['scratch'] else 'finetuned'
        assert meta['normalise'] == 'dataset' and meta['input_role'] == 'measurements'
        assert meta['epochs'] == 40 and meta['seed'] == 0 and meta['train_cars'] == 2160
    else:
        torch.manual_seed(0)
        base, experiment, state = load_geometry(path, 'cpu')
        net = FieldEncoder(base).to(device).eval()
        meta, name = {}, 'pretrained'
        assert torch.count_nonzero(net.base.encoder.embed[0].weight[:, -2:]) == 0
    assert not net.training
    model = Model(name, str(path), net, experiment, meta, device,
                  AttentionCapture(net.base.encoder.blocks), AttentionCapture(net.base.decoder.blocks), {}, {})
    AUDIT['models'][name] = {'path': str(path), 'epoch': state.get('epoch'), 'benchmark': meta,
                             'config': state['config']}
    STATE.setdefault('models', []).append(model)
    if len(STATE['models']) == 3:
        byname = {m.name: m for m in STATE['models']}
        fm, sm = byname['finetuned'].meta, byname['scratch'].meta
        keys = ('normalise', 'mean', 'std', 'floor', 'mix', 'protocol', 'base_checkpoint', 'train_cars', 'queries',
                'visible_weight', 'epochs', 'seed', 'lr', 'new_lr', 'input_role')
        assert all(fm[k] == sm[k] for k in keys)
        assert Path(fm['base_checkpoint']) == Path(byname['pretrained'].path).resolve()
        assert len({json.dumps(AUDIT['models'][m.name]['config']['model'], sort_keys=True)
                    for m in STATE['models']}) == 1
        assert len({m.experiment.data.encoder_points for m in STATE['models']}) == 1
        AUDIT['checks']['matched_checkpoint_architecture_and_training'] = True
    return model


def iterate_cases(setting):
    assert setting == 'none', 'This frozen experiment evaluates the no-sensor setting only.'
    for record in STATE['records']:
        field = fb.load_field(STATE['fields'], record['id'])
        assert field.normals is not None and len(field.points) == 32000
        # The nine conditions of a wing share one geometry, so its encoding is computed once per wing.
        signature = (digest(record['path']), array_digest(field.points, field.normals, field.weights))
        group = record['family']
        previous = STATE['group_inputs'].setdefault(group, signature)
        assert previous == signature, f'Conditions change geometry: {group}'
        STATE['current'] = record
        AUDIT['cases'].append(dict(id=record['id'], group=group, grid_id=record.get('grid_id'), observations=0))
        yield record['id'], field.points, field.normals, field.weights, field.values, group, np.empty(0, np.int64)
        save_audit()


@torch.inference_mode()
def encode(model, X, N, A, observed, values):
    assert len(observed) == len(values) == 0
    record, group = STATE['current'], STATE['current']['family']
    if group not in STATE['prepared']:
        STATE['prepared'][group] = fb.prepare_car(record, X, model.experiment)
    prepared = STATE['prepared'][group]
    if group not in model.cache:
        sample = fb.encoder_inputs(prepared, len(X), observed, values)
        assert all(np.count_nonzero(sample[k]) == 0 for k in ('values', 'present', 'anchor_values', 'anchor_present'))
        batch = to_device(fb.collate_inputs([sample]), model.device)
        model.encoder_capture.begin()
        packet = model.net.encode(batch, batch['values'], batch['present'], batch['anchor_values'], batch['anchor_present'])
        enc = model.encoder_capture.end()
        assert enc.shape == (4, 4, 256, 256)
        if model.name == 'pretrained' and not model.cache:
            original, _, _ = load_geometry(model.path, model.device)
            original_packet = original.encoder(batch)
            q = torch.as_tensor(X[:257], dtype=torch.float32, device=model.device)[None]
            token_error = float((packet.tokens - original_packet.tokens).abs().max())
            feature_error = float((model.net.features(packet, q) - original.query_features(original_packet, q)).abs().max())
            assert token_error < 2e-5 and feature_error < 2e-5, (token_error, feature_error)
            AUDIT['checks']['pretrained_geometry_equivalence'] = dict(token_max_error=token_error,
                                                                     feature_max_error=feature_error)
            del original
        model.cache[group] = {'packet': packet, 'encoder_attention': enc, 'group': group}
    tokens = model.cache[group]
    packet = tokens['packet']
    return tokens, packet.world_positions[0], packet.weights[0]


@torch.inference_mode()
def decoder_attention(model, tokens, Q):
    q = torch.as_tensor(Q, dtype=torch.float32, device=model.device)[None]
    model.decoder_capture.begin()
    model.net.features(tokens['packet'], q)
    att = model.decoder_capture.end()
    assert att.shape == (2, 4, len(Q), 256)
    return att


def encoder_attention(model, tokens):
    return tokens['encoder_attention']


@torch.inference_mode()
def predict(model, tokens, Q):
    group = tokens['group']
    key = (group, array_digest(Q))  # predictions are cached per wing and query set
    if key not in model.predictions:
        assert not model.encoder_capture.enabled and not model.decoder_capture.enabled
        mean, scale = fb.normalising(model.meta, np.empty(0), np.empty(0))
        q = torch.as_tensor(Q, dtype=torch.float32, device=model.device)[None]
        pred = model.net.predict(tokens['packet'], q)[0].cpu().numpy().astype(np.float64) * scale + mean
        assert np.isfinite(pred).all()
        if not model.predictions:
            native = fb.Areal(model.name, model.net, model.experiment, model.meta, torch.device(model.device), 1, 'none')
            native.prepared[STATE['current']['id']] = STATE['prepared'][group]
            surface = fb.Surface(STATE['current']['id'], STATE['current'], Q, np.ones(len(Q)), np.zeros_like(Q))
            obs = fb.Observed('unobserved', {'kind': 'none'}, np.empty(0, np.int64), np.empty(0))
            reference = native.predict_car(surface, [obs])[0]
            error = float(np.abs(pred - reference).max())
            assert error == 0, error
            AUDIT['checks'][f'{model.name}_native_prediction_max_error'] = error
            # No value can enter the encoder when the observation index is empty.
            hostile = fb.encoder_inputs(STATE['prepared'][group], len(Q), np.empty(0, np.int64),
                                       np.full(len(Q), 1e30))
            assert all(np.count_nonzero(hostile[k]) == 0 for k in ('values', 'present', 'anchor_values', 'anchor_present'))
            AUDIT['checks']['hidden_field_values_excluded'] = True
        model.predictions[key] = pred
    return model.predictions[key]


def save_audit():
    for m in STATE['models']:
        AUDIT['models'][m.name]['capture'] = dict(encoder=m.encoder_capture.summary(), decoder=m.decoder_capture.summary())
    (STATE['out'] / 'export_audit.json').write_text(json.dumps(AUDIT, indent=2, allow_nan=False) + '\n')
