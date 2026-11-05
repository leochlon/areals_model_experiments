"""AeroJEPA context encoder backed by the frozen Areal geometry checkpoint."""

import torch
from torch import nn

from areal_tokens.io import load_model
from physicsnemo.experimental.models.aerojepa.encoders.base import BaseContextEncoder
from physicsnemo.experimental.models.aerojepa.layers import EncoderOutput, TokenSet


class ArealContextEncoder(BaseContextEncoder):
    """Consume a prepared Areal surface in its normalized coordinate frame.

    The first K positions are its K anchors, followed by the remaining points.
    context_feat columns: normal xyz, point weight, anchor weight (first K),
    and total physical surface area. Query coordinates use the same frame.
    No flow values or operating conditions enter this encoder.
    """

    supports_batched_forward = True

    def __init__(self, checkpoint: str, token_dim: int = 128,
                 latent_count: int | None = None):
        super().__init__()
        model, experiment, state = load_model(checkpoint, "cpu")
        assert state["epoch"] == 14
        self.encoder = model.encoder.eval().requires_grad_(False)
        self.source_latents = experiment.model.latents
        self.latents = self.source_latents if latent_count is None else int(latent_count)
        self.projection = nn.Linear(experiment.model.width, token_dim)
        self.checkpoint = str(checkpoint)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval().requires_grad_(False)
        return self

    def forward(self, *, context_pos, context_feat):
        feature_count = 6 if self.latents == self.source_latents else 8
        if context_pos.ndim != 2 or context_feat.shape != (len(context_pos), feature_count):
            raise ValueError(f"Expected positions [N,3] and geometry-only features [N,{feature_count}]")
        if len(context_pos) < self.latents:
            raise ValueError("The prepared surface must contain all Areal anchors")
        k = self.latents
        if k == self.source_latents:
            indices = torch.arange(k, device=context_pos.device)
            weights = context_feat[:k, 4]
        else:
            indices = torch.argsort(context_feat[:, 6], descending=True, stable=True)[:k].flip(0)
            weights = context_feat[indices, 7]
        anchors, anchor_normals = context_pos[indices], context_feat[indices, :3]
        batch = dict(points=context_pos[None], normals=context_feat[None, :, :3],
                     point_weights=context_feat[None, :, 3], anchors=anchors[None],
                     anchor_normals=anchor_normals[None], anchor_weights=weights[None],
                     center=context_pos.new_zeros(1, 3), scale=context_pos.new_ones(1, 1),
                     area=context_feat[None, :1, 5], complete=context_pos.new_ones(1, 1))
        with torch.no_grad():
            packet = self.encoder(batch)
        features = self.projection(packet.tokens[0])
        global_token = (features * weights[:, None]).sum(0) / weights.sum().clamp_min(1e-12)
        aux = dict(normals=anchor_normals, area_weights=weights,
                   physical_area_weights=weights * context_feat[0, 5])
        tokens = TokenSet(features=features, coords=anchors,
                          mask=torch.ones(k, dtype=torch.bool, device=context_pos.device),
                          global_token=global_token, aux=aux)
        return EncoderOutput(tokens=tokens, global_token=global_token,
                             aux=dict(point_token_count=k, geometry_prior="areal_epoch14"))

    def forward_batched(self, *, context_pos, context_feat, context_pos_n):
        outputs = [self.forward(context_pos=p[:int(n)], context_feat=f[:int(n)])
                   for p, f, n in zip(context_pos, context_feat, context_pos_n, strict=True)]
        tokens = TokenSet(features=torch.stack([x.tokens.features for x in outputs]),
                          coords=torch.stack([x.tokens.coords for x in outputs]),
                          mask=torch.stack([x.tokens.mask for x in outputs]),
                          global_token=torch.stack([x.global_token for x in outputs]),
                          aux={key: torch.stack([x.tokens.aux[key] for x in outputs])
                               for key in outputs[0].tokens.aux})
        return EncoderOutput(tokens=tokens, global_token=tokens.global_token,
                             aux=dict(point_token_count=context_pos_n.new_full(context_pos_n.shape, self.latents)))
