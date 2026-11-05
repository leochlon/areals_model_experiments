"""Pressure queries over the existing polynomial-patch encoder.

Checkpoint-compatible encoder parameters; no pressure labels enter encode_scene.
Query points do not attend to other query points. Prediction is therefore
independent of the inference query batch and its ordering.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import train_mesh_geometry as geometry


class TokenPatchEncoder(geometry.PatchEncoder):
    """Same state_dict as PatchEncoder, exposing its existing four tokens."""

    def encode_tokens(self, x, weights=None):
        p, n = x[..., :3], x[..., 3:]
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        projector = torch.stack((n[..., 0] ** 2, n[..., 1] ** 2, n[..., 2] ** 2,
                                 math.sqrt(2) * n[..., 0] * n[..., 1],
                                 math.sqrt(2) * n[..., 0] * n[..., 2],
                                 math.sqrt(2) * n[..., 1] * n[..., 2]), -1)
        features = self.embed(torch.cat((p, projector), -1))
        h = self.read(self.tokens.expand(len(x), -1, -1), features, weights)
        for block in self.blocks:
            h = block(h, h)
        return h


class FourierCoordinates(nn.Module):
    def __init__(self, bands=5):
        super().__init__()
        self.register_buffer("frequencies", 2 * math.pi * 2. ** torch.arange(bands))
        self.size = 3 * (1 + 2 * bands)

    def forward(self, xyz):
        phase = xyz[..., :, None] * self.frequencies
        return torch.cat((xyz, phase.sin().flatten(-2), phase.cos().flatten(-2)), -1)


class PressureModel(nn.Module):
    def __init__(self, encoder_config, condition_dim, width=128, heads=4,
                 global_tokens=32, global_depth=2, neighbors=8, bands=5,
                 encoder_chunk=32, gradient_checkpointing=True):
        super().__init__()
        if width <= 0 or width % heads or min(global_tokens, neighbors, encoder_chunk) <= 0:
            raise ValueError("Invalid decoder width/heads/token/chunk configuration")
        self.config = dict(encoder_config=encoder_config, condition_dim=condition_dim,
                           width=width, heads=heads, global_tokens=global_tokens,
                           global_depth=global_depth, neighbors=neighbors, bands=bands,
                           encoder_chunk=encoder_chunk,
                           gradient_checkpointing=gradient_checkpointing)
        self.encoder = TokenPatchEncoder(**encoder_config)
        self.project_tokens = nn.Linear(encoder_config["width"], width)
        # Geometry side information: origin(3), frame(9), log radius, log mesh scale.
        self.patch_metadata = nn.Sequential(nn.Linear(14, width), nn.GELU(), nn.Linear(width, width))
        self.patch_coefficients = nn.Linear(len(geometry.powers(encoder_config["degree"])), width)
        self.condition = nn.Sequential(nn.Linear(max(condition_dim, 1), width), nn.GELU(), nn.Linear(width, width))
        self.global_seeds = nn.Parameter(torch.randn(1, global_tokens, width) * .02)
        self.pool = geometry.AttentionBlock(width, heads)
        self.global_blocks = nn.ModuleList([geometry.AttentionBlock(width, heads) for _ in range(global_depth)])
        self.coordinates = FourierCoordinates(bands)
        self.query_embedding = nn.Sequential(nn.Linear(self.coordinates.size, width), nn.GELU(), nn.Linear(width, width))
        self.relative_embedding = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
        self.local_read = geometry.AttentionBlock(width, heads)
        self.global_read = geometry.AttentionBlock(width, heads)
        self.pressure_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        self.register_buffer("cp_mean", torch.tensor(0.))
        self.register_buffer("cp_std", torch.tensor(1.))
        self.register_buffer("condition_mean", torch.zeros(condition_dim))
        self.register_buffer("condition_std", torch.ones(condition_dim))

    def set_normalization(self, stats):
        self.cp_mean.fill_(stats["cp_mean"])
        self.cp_std.fill_(stats["cp_std"])
        self.condition_mean.copy_(torch.as_tensor(stats["condition_mean"]))
        self.condition_std.copy_(torch.as_tensor(stats["condition_std"]))

    def freeze_encoder(self, frozen):
        self.encoder.requires_grad_(not frozen)

    def encode_scene(self, scene, conditions):
        """scene contains inputs, origins, frames, radii and mesh scale ONLY."""
        chunks = []
        train_encoder = self.training and torch.is_grad_enabled() and any(p.requires_grad for p in self.encoder.parameters())
        for x in scene["inputs"].split(self.config["encoder_chunk"]):
            if train_encoder and self.config["gradient_checkpointing"]:
                h = checkpoint(self.encoder.encode_tokens, x, use_reentrant=False)
            else:
                h = self.encoder.encode_tokens(x)
            chunks.append(h)
        h = torch.cat(chunks)
        coefficients = self.encoder.head(h.mean(1))
        meta = torch.cat((scene["origin"], scene["frame"].flatten(1),
                          scene["radius"].log()[:, None],
                          scene["scale"].log().expand(len(h), 1)), -1)
        tokens = self.project_tokens(h) + (self.patch_metadata(meta) + self.patch_coefficients(coefficients))[:, None]
        if self.config["condition_dim"]:
            condition = (conditions - self.condition_mean) / self.condition_std
        else:
            condition = tokens.new_zeros(1)
        c = self.condition(condition)
        glob = self.pool(self.global_seeds + c[None, None], tokens.flatten(0, 1)[None])
        for block in self.global_blocks:
            glob = block(glob, glob)
        return dict(tokens=tokens, global_tokens=glob, condition=c,
                    origin=scene["origin"], frame=scene["frame"], radius=scene["radius"])

    def decode(self, encoded, xyz, neighbor_ids=None, standardized=False):
        """xyz is in the mesh's bbox-centred, diagonal-normalized coordinates."""
        if neighbor_ids is None:
            k = min(self.config["neighbors"], len(encoded["origin"]))
            neighbor_ids = torch.cdist(xyz, encoded["origin"]).topk(k, largest=False).indices
        if neighbor_ids.ndim != 2 or len(neighbor_ids) != len(xyz):
            raise ValueError("neighbor_ids must have shape [number of queries, K]")
        delta = xyz[:, None] - encoded["origin"][neighbor_ids]
        local = torch.einsum("qki,qkij->qkj", delta, encoded["frame"][neighbor_ids])
        local = local / encoded["radius"][neighbor_ids, None]
        # Bounded growth away from the surface; these are features, not an
        # extrapolated polynomial chart or a claim of nearest-point inversion.
        relative = torch.cat((local.sign() * torch.log1p(local.abs()),
                              torch.log1p(local.norm(dim=-1, keepdim=True))), -1)
        nearby = encoded["tokens"][neighbor_ids] + self.relative_embedding(relative)[:, :, None]
        q = self.query_embedding(self.coordinates(xyz)) + encoded["condition"]
        q = self.local_read(q[:, None], nearby.flatten(1, 2))[:, 0]
        q = self.global_read(q[None], encoded["global_tokens"])[0]
        pred = self.pressure_head(q)[:, 0]
        return pred if standardized else pred * self.cp_std + self.cp_mean

    def forward(self, scene, conditions, xyz, neighbor_ids=None, standardized=False):
        return self.decode(self.encode_scene(scene, conditions), xyz, neighbor_ids, standardized)


def load_geometry_initialization(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("version") != geometry.VERSION or "model_config" not in state:
        raise ValueError("Expected train_mesh_geometry.py checkpoint; original best_normal.pt is a different model")
    # Strict loading catches format/config drift before starting pressure training.
    encoder = TokenPatchEncoder(**state["model_config"])
    encoder.load_state_dict(state["model"], strict=True)
    return state
