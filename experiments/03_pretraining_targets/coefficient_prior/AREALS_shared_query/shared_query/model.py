"""Native AREALS with auxiliary supervision on its existing spatial queries.

All modes instantiate exactly the same modules.  The native encoder and query
decoder are retained verbatim and are the computation exported for pressure.
Only training losses differ between baseline, Hessian and operator runs.
"""
from __future__ import annotations

import math
from pathlib import Path
import torch
from torch import nn

from areal_tokens.config import ModelConfig
from areal_tokens.losses import reconstruction_loss
from areal_tokens.model import GeometryAutoencoder
from .geometry import operator_losses, symmetric_frobenius_squared


class JointGeometry(nn.Module):
    def __init__(self, model_config: ModelConfig | dict, hessian_scales: int = 3):
        super().__init__()
        if isinstance(model_config, dict):
            model_config = ModelConfig(**model_config)
        if isinstance(hessian_scales, bool) or not isinstance(hessian_scales, int) or hessian_scales < 1:
            raise ValueError("hessian_scales must be a positive integer")
        if model_config.normal_head != 'projector':
            raise ValueError("Shared-query comparison requires the native normal-projector head")
        if model_config.hint_noise != 0:
            raise ValueError("Target-derived hint noise must be zero in the shared-query comparison")
        self.base = GeometryAutoencoder(model_config)
        self.hessian_scales = hessian_scales
        width = model_config.width
        self.hessian_head = nn.Linear(width, 6 * hessian_scales)
        # Frame and radius identify the chart in which the requested output is
        # expressed. They are not target coefficients or hidden geometry.
        self.patch_head = nn.Sequential(
            nn.Linear(width + 10, width), nn.GELU(), nn.Linear(width, 10))
        nn.init.normal_(self.patch_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.patch_head[-1].bias)

    def forward(self, batch: dict, aux: dict | None = None):
        geometry = self.base.encoder(batch)
        out = {'native': self.base.decode(geometry, batch['queries'])}
        aux = {} if aux is None else aux
        if 'h_x' in aux:
            hidden = self.base.query_features(geometry, aux['h_x'])
            out['H'] = self.hessian_head(hidden).unflatten(-1, (self.hessian_scales, 6))
        if 'patch_origin' in aux:
            origin, frame, radius = (aux[k] for k in ('patch_origin', 'patch_frame', 'patch_radius'))
            if origin.ndim != 3 or origin.shape[-1] != 3:
                raise ValueError("patch_origin must be [objects, patches, 3] in world coordinates")
            if frame.shape != (*origin.shape[:-1], 3, 3) or radius.shape != origin.shape[:-1]:
                raise ValueError("Patch frame/radius shapes do not match origins")
            if not bool(torch.isfinite(origin).all()) or not bool(torch.isfinite(frame).all()):
                raise ValueError("Patch origins and frames must be finite")
            with torch.no_grad():
                with torch.autocast(device_type=frame.device.type, enabled=False):
                    unit = torch.eye(3, dtype=torch.float32, device=frame.device)
                    frame32 = frame.float()
                    error = (frame32.transpose(-1, -2) @ frame32 - unit).abs().amax()
                    if bool(error > 1e-3) or bool((torch.linalg.det(frame32) < 0.999).any()):
                        raise ValueError("Patch frames must be orthonormal and right-handed")
            if not bool(torch.isfinite(radius).all()) or bool((radius <= 0).any()):
                raise ValueError("Patch radii must be finite positive world lengths")
            hidden = self.base.query_features(geometry, origin)
            relative_radius = radius / geometry.scale.reshape(len(radius), 1)
            metadata = torch.cat((frame.flatten(-2), relative_radius.log().unsqueeze(-1)), -1)
            out['coeff'] = self.patch_head(torch.cat((hidden, metadata.to(hidden.dtype)), -1))
        return out


def loss_for(model, batch, aux, train_config, mode='baseline', aux_weight=0.1):
    """Compute a fixed, explicit objective without switching off geometry terms.

    baseline: L_native(d,P)
    hessian:  L_native(d,P) + aux_weight * L_H
    operator: L_native(d,P) + aux_weight * (L_local_position + L_local_P + L_relative_A)

    L_native is the unchanged native AREALS reconstruction loss.  Its normal
    weight must remain strictly positive in every mode.  Both auxiliary heads
    are always instantiated, preserving the same model and initialization.
    Returned metrics are detached scalar tensors suitable for distributed logs.
    """
    if mode not in ('baseline', 'hessian', 'operator'):
        raise ValueError(f"Unknown supervision mode {mode!r}")
    if not math.isfinite(float(train_config.normal_weight)) or train_config.normal_weight <= 0:
        raise ValueError("Continuous native normal-projector supervision must remain active")
    if not math.isfinite(float(aux_weight)) or aux_weight < 0 or (mode != 'baseline' and aux_weight == 0):
        raise ValueError("Auxiliary weight must be finite, and positive for an auxiliary mode")
    aux = {} if aux is None else aux
    if mode == 'hessian':
        required = {'h_x', 'h_target'}
        forward_aux = {k: aux[k] for k in ['h_x'] if k in aux}
    elif mode == 'operator':
        required = {'patch_origin', 'patch_frame', 'patch_radius', 'patch_coeff', 'patch_uv', 'patch_tau'}
        forward_aux = {k: aux[k] for k in ['patch_origin', 'patch_frame', 'patch_radius'] if k in aux}
    else:
        required, forward_aux = set(), None
    missing = required - aux.keys()
    if missing:
        raise ValueError(f"Missing {mode} supervision arrays: {sorted(missing)}")
    prediction = model(batch, forward_aux)
    native = reconstruction_loss(prediction['native'], batch, train_config)
    total = native['loss']
    metrics = dict(native_loss=native['loss'], distance=native['distance'], normal=native['normal'],
                   normal_fraction=native['normal_fraction'])
    with torch.no_grad():
        native_pred = prediction['native']
        distance_error = native_pred['normalized_udf'].float() - batch['udf'].float() / batch['scale'].float()
        metrics['distance_nrmse'] = distance_error.square().mean().sqrt()
        cosine = (native_pred['normals'].float() * batch['query_normals'].float()).sum(-1).abs().clamp(0, 1)
        valid = batch['normal_valid'].float()
        metrics['normal_degrees'] = (torch.rad2deg(torch.acos(cosine)) * valid).sum() / valid.sum().clamp_min(1)
    if mode == 'hessian':
        target = aux['h_target'].float()
        if target.shape != prediction['H'].shape or not bool(torch.isfinite(target).all()):
            raise ValueError("Hessian target shape or finiteness is invalid")
        error = prediction['H'].float() - target
        h_loss = symmetric_frobenius_squared(error).mean()
        total = total + aux_weight * h_loss
        metrics['H'] = h_loss
    elif mode == 'operator':
        extra, terms = operator_losses(prediction['coeff'], aux['patch_coeff'], aux['patch_uv'],
                                       aux['patch_tau'], aux.get('patch_weights'))
        total = total + aux_weight * extra
        metrics.update(auxiliary_loss=extra, **terms)
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Nonfinite shared-query training loss")
    metrics['loss'] = total
    return total, {k: value.detach() for k, value in metrics.items()}


def export_native(model):
    """Return the complete pressure-reusable native state, including query decoder.

    The caller writes its ExperimentConfig alongside this state in the ordinary
    native checkpoint format.  No new global/query module is created downstream.
    """
    unwrapped = model.module if hasattr(model, 'module') else model
    return {key: value.detach().cpu().clone() for key, value in unwrapped.base.state_dict().items()}


def load_native_initialization(model: JointGeometry, checkpoint: str | Path | dict):
    """Strictly load a trusted native checkpoint into the unchanged base only."""
    state = torch.load(checkpoint, map_location='cpu', weights_only=False) if not isinstance(checkpoint, dict) else checkpoint
    weights = state.get('model', state)
    model.base.load_state_dict(weights, strict=True)
    return state
