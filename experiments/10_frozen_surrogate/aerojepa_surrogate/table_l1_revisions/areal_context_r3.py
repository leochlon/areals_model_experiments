"""AeroJEPA context encoder backed by the frozen Areal geometry checkpoint."""
import json
from pathlib import Path

import numpy as np
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
                 latent_count: int | None = None, dense_context: str | None = None,
                 dense_anchors: str | None = None):
        super().__init__()
        model, experiment, state = load_model(checkpoint, "cpu")
        assert state["epoch"] == 14
        self.encoder = model.encoder.eval().requires_grad_(False)
        self.source_latents = experiment.model.latents
        self.latents = self.source_latents if latent_count is None else int(latent_count)
        self.projection = nn.Linear(experiment.model.width, token_dim)
        self.checkpoint = str(checkpoint)
        self.dense_cpu, self.dense_gpu = {}, {}
        self.anchor_cpu, self.anchor_gpu = {}, {}
        if dense_context:
            root = Path(dense_context)
            metadata = json.loads((root / "ready.json").read_text())
            for row in metadata["records"]:
                with np.load(root / (row["id"] + ".npz"), allow_pickle=False) as p:
                    self.dense_cpu[row["code"]] = tuple(torch.from_numpy(p[key]) for key in
                                                       ("points", "normals", "point_weights"))
        if dense_anchors:
            if not dense_context:
                raise ValueError("Dense anchors require a dense geometry context")
            root = Path(dense_anchors)
            metadata = json.loads((root / "ready.json").read_text())
            if metadata["geometry_tokens"] != self.latents:
                raise ValueError("Dense anchor count does not match geometry tokens")
            for row in metadata["records"]:
                with np.load(root / (row["id"] + ".npz"), allow_pickle=False) as p:
                    self.anchor_cpu[row["code"]] = tuple(torch.from_numpy(p[key]) for key in
                                                        ("anchors", "anchor_normals", "anchor_weights"))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval().requires_grad_(False)
        return self

    def forward(self, *, context_pos, context_feat):
        feature_count = 6 if self.anchor_cpu or self.latents == self.source_latents else 8
        if self.dense_cpu:
            feature_count += 1
        if context_pos.ndim != 2 or context_feat.shape != (len(context_pos), feature_count):
            raise ValueError(f"Expected positions [N,3] and geometry-only features [N,{feature_count}]")
        if not self.anchor_cpu and len(context_pos) < self.latents:
            raise ValueError("The prepared surface must contain all Areal anchors")
        k = self.latents
        if self.anchor_cpu:
            code = int(context_feat[0, -1].item())
            key = (code, context_pos.device)
            if key not in self.anchor_gpu:
                self.anchor_gpu[key] = tuple(t.to(context_pos.device) for t in self.anchor_cpu[code])
            anchors, anchor_normals, weights = self.anchor_gpu[key]
        elif k == self.source_latents:
            indices = torch.arange(k, device=context_pos.device)
            weights = context_feat[:k, 4]
        else:
            indices = torch.argsort(context_feat[:, 6], descending=True, stable=True)[:k].flip(0)
            weights = context_feat[indices, 7]
        if not self.anchor_cpu:
            anchors, anchor_normals = context_pos[indices], context_feat[indices, :3]
        points, normals, point_weights = context_pos, context_feat[:, :3], context_feat[:, 3]
        if self.dense_cpu:
            code = int(context_feat[0, -1].item())
            key = (code, context_pos.device)
            if key not in self.dense_gpu:
                self.dense_gpu[key] = tuple(t.to(context_pos.device) for t in self.dense_cpu[code])
            points, normals, point_weights = self.dense_gpu[key]
        batch = dict(points=points[None], normals=normals[None],
                     point_weights=point_weights[None], anchors=anchors[None],
                     anchor_normals=anchor_normals[None], anchor_weights=weights[None],
                     center=context_pos.new_zeros(1, 3), scale=context_pos.new_ones(1, 1),
                     area=context_feat[None, :1, 5], complete=context_pos.new_ones(1, 1))
        with torch.no_grad():
            packet = self.encoder(batch)
        if packet.tokens.shape[1] != k:
            raise ValueError(f"Encoder returned {packet.tokens.shape[1]} tokens, expected {k}")
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
