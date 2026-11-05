import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

from areal_tokens.config import ExperimentConfig
from areal_tokens.field import FieldEncoder as NativeField
from areal_tokens.model import GeometryAutoencoder, LatentGeometry, SpatialDecoder

ORIGINAL = Path('/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/field_pointmae_released_420')
sys.path.insert(0, str(ORIGINAL))
import extract_features as pointmae
sys.path.insert(0, '/workspace/runs/field_benchmark_4gpu_20261023')
import field_benchmark as fb

ARM, SEED, OUT = None, None, None
CURRENT_MODEL = None
NativeCompletion, NativeAreal = fb.Completion, fb.Areal
NativeAdamW = torch.optim.AdamW


class EncoderAdamW(NativeAdamW):
    def __init__(self, groups, **kwargs):
        encoder = CURRENT_MODEL.pointmae if ARM == 'pointmae' else CURRENT_MODEL.base.encoder
        reader = list(CURRENT_MODEL.base.decoder.parameters()) + list(CURRENT_MODEL.head.parameters())
        if ARM == 'pointmae':
            reader += list(CURRENT_MODEL.projection.parameters())
        super().__init__([dict(params=[p for p in encoder.parameters() if p.requires_grad], lr=groups[0]['lr']),
                          dict(params=reader, lr=groups[1]['lr'])], **kwargs)


class Field(NativeField):
    def __init__(self, base):
        global CURRENT_MODEL
        super().__init__(base)
        # Both backbones receive the same fresh coordinate-query decoder and scalar head.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(91700 + SEED)
            self.base.decoder = SpatialDecoder(base.config, condition_dim=1 if base.config.input_condition == 'complete' else 0)
            self.head = nn.Linear(base.config.width, 1)
        initialization = dict(arm=ARM, seed=SEED, input_point_cap=4096, tokens=256)
        if ARM == 'pointmae':
            self.pointmae, _, metadata = pointmae.author_encoder()
            self.pointmae.requires_grad_(True)
            # The released model uses coordinate MLP positions and supports variable group counts.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(91700 + SEED)
                self.projection = nn.Linear(384, 192)
            self.base.encoder.requires_grad_(False)  # Unused; it never enters this arm's forward pass.
            # The 256 Point-MAE groups are centred on the same surface anchors as AREALS; no feature interpolation.
            initialization.update(pointmae_source=metadata, coordinates_only=True, downstream_groups=256,
                                  released_pretraining_groups=64, group_size=32)
            active = self.pointmae
        else:
            initialization.update(coordinates_and_normals=True)
            active = self.base.encoder
        initialization.update(encoder_parameter_names=[name for name, _ in active.named_parameters()])
        if OUT is not None and not (OUT / 'initialization.json').exists():
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / 'initialization.json').write_text(json.dumps(initialization, indent=2)+'\n')
            name, parameter = next((n, p) for n, p in active.named_parameters() if p.requires_grad)
            def hook(gradient):
                target = OUT / 'encoder_gradient.json'
                if not target.exists():
                    target.write_text(json.dumps(dict(parameter=name, gradient_norm=float(gradient.norm()),
                          finite=bool(torch.isfinite(gradient).all()), requires_grad=True), indent=2)+'\n')
            parameter.register_hook(hook)
        CURRENT_MODEL = self

    def encode(self, batch, values, present, anchor_values, anchor_present):
        if ARM == 'areals':
            return super().encode(batch, values, present, anchor_values, anchor_present)
        points, anchors = batch['points'].float(), batch['anchors'].float()
        valid = batch['point_weights'] > 0
        center = (points * valid[..., None]).sum(1, keepdim=True) / valid.sum(1)[:, None, None]
        radius = (points-center).square().sum(-1).sqrt().masked_fill(~valid, 0).amax(1, keepdim=True).clamp_min(1e-12)[..., None]
        normalized, centers = (points-center)/radius, (anchors-center)/radius
        with torch.no_grad():
            nearest = torch.cdist(centers, normalized).masked_fill(~valid[:, None], torch.inf).topk(32, largest=False).indices
        rows = torch.arange(len(points), device=points.device)[:, None, None]
        neighborhoods = normalized[rows, nearest] - centers[:, :, None]
        tokens, mask = self.pointmae(neighborhoods.contiguous(), centers.contiguous(), noaug=True)
        assert tokens.shape[1:] == (256, 384) and not mask.any()
        return LatentGeometry(self.projection(tokens), anchors, batch['anchor_normals'], batch['anchor_weights'],
                              batch['center'], batch['scale'], batch['area'])


def load(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    config = ExperimentConfig.from_dict(state['config'])
    model = Field(GeometryAutoencoder(config.model))
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval(), config, state


class Completion(NativeCompletion):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        for key in ['values', 'present', 'anchor_values', 'anchor_present']:
            sample[key].fill(0)
        sample['hidden'].fill(1)
        return sample


class Areal(NativeAreal):
    def __init__(self, name, model, experiment, meta, device, batch_size, amp, source=None):
        meta.update(online_backbone=ARM, encoder_finetuned=True, fresh_common_query_decoder=True,
                    pointmae_native_groups=256 if ARM == 'pointmae' else None)
        super().__init__(name, model, experiment, meta, device, batch_size, amp, source)

    def predict_car(self, surface, observed):
        import numpy as np
        empty = fb.Observed('shape', dict(name='unobserved', kind='none'), np.empty(0, np.int64), np.empty(0))
        result = super().predict_car(surface, [empty])[0]
        return [result.copy() for _ in observed]


if __name__ == '__main__':
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--backbone', choices=['areals', 'pointmae'], required=True)
    p.add_argument('--audit-seed', type=int, required=True)
    args, remaining = p.parse_known_args()
    ARM, SEED = args.backbone, args.audit_seed
    if remaining[0] == 'train-areal':
        OUT = Path(remaining[remaining.index('--out') + 1])
        assert not (OUT / 'field_best.pt').exists()
    fb.FieldEncoder, fb.load_field_encoder = Field, load
    if remaining[0] == 'train-areal':
        torch.optim.AdamW = EncoderAdamW
    fb.Completion, fb.Areal = Completion, Areal
    from transfer_architecture import selection
    fb.selection_problems = selection
    fb.main(remaining)
