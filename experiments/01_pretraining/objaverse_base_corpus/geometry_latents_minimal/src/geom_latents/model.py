"""Spatial vector tokens and coordinate-query decoders; no scalar token bottleneck."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class FourierFeatures(nn.Module):
    def __init__(self, bands: int, low: float, high: float):
        super().__init__()
        self.register_buffer(
            "frequencies",
            2
            * torch.pi
            / torch.logspace(
                torch.log10(torch.tensor(high)).item(), torch.log10(torch.tensor(low)).item(), bands
            ),
        )
        self.output_dim = 3 + 6 * bands

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        angles = positions.float()[..., :, None] * self.frequencies.float()
        return torch.cat([positions.float(), angles.sin().flatten(-2), angles.cos().flatten(-2)], dim=-1)


class Rotary3D(nn.Module):
    def __init__(self, head_dim: int, low: float, high: float):
        super().__init__()
        if head_dim % 6:
            raise ValueError("3D rotary head width must be divisible by six")
        frequencies = (
            2
            * torch.pi
            / torch.logspace(
                torch.log10(torch.tensor(high)).item(), torch.log10(torch.tensor(low)).item(), head_dim // 6
            )
        )
        self.register_buffer("frequencies", frequencies.repeat(3))
        self.register_buffer("axes", torch.arange(3).repeat_interleave(head_dim // 6))

    def forward(self, value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        angles = positions.float()[..., self.axes] * self.frequencies.float()
        cosine, sine = angles.cos()[:, None], angles.sin()[:, None]
        pairs = value.float().unflatten(-1, (-1, 2))
        a, b = pairs.unbind(-1)
        return torch.stack([a * cosine - b * sine, a * sine + b * cosine], dim=-1).flatten(-2).to(value.dtype)


class WeightedAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads, self.head_dim = config.heads, config.width // config.heads
        self.q = nn.Linear(config.width, config.width)
        self.kv = nn.Linear(config.width, 2 * config.width)
        self.out = nn.Linear(config.width, config.width)
        self.rotary = Rotary3D(self.head_dim, config.min_wavelength, config.max_wavelength)
        self.dropout = config.dropout

    def forward(self, query, values, query_positions, value_positions, weights):
        def split(value):
            return value.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)

        q = self.rotary(split(self.q(query)), query_positions)
        k, v = self.kv(values).chunk(2, dim=-1)
        k, v = self.rotary(split(k), value_positions), split(v)
        log_weights = torch.where(weights > 0, weights.float().clamp_min(1e-38).log(), -torch.inf)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=log_weights[:, None, None].to(q.dtype),
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(output.transpose(1, 2).flatten(-2))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.query_norm, self.value_norm = nn.LayerNorm(config.width), nn.LayerNorm(config.width)
        self.attention = WeightedAttention(config)
        self.ff_norm = nn.LayerNorm(config.width)
        self.ff = nn.Sequential(
            nn.Linear(config.width, config.width * config.ff_multiplier),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.width * config.ff_multiplier, config.width),
        )

    def forward(self, query, values, query_positions, value_positions, weights):
        query = query + self.attention(
            self.query_norm(query), self.value_norm(values), query_positions, value_positions, weights
        )
        return query + self.ff(self.ff_norm(query))


@dataclass
class LatentGeometry:
    tokens: torch.Tensor
    positions: torch.Tensor
    normals: torch.Tensor
    weights: torch.Tensor
    center: torch.Tensor
    scale: torch.Tensor
    area: torch.Tensor
    initial_tokens: torch.Tensor | None = None

    @property
    def world_positions(self):
        return self.positions * self.scale[:, None] + self.center[:, None]

    def with_tokens(self, tokens: torch.Tensor) -> "LatentGeometry":
        return replace(self, tokens=tokens)

    def detach(self) -> "LatentGeometry":
        return LatentGeometry(
            **{name: value.detach() if value is not None else None for name, value in vars(self).items()}
        )


class GeometryEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.position = FourierFeatures(config.fourier_bands, config.min_wavelength, config.max_wavelength)
        self.embed = nn.Sequential(
            nn.Linear(self.position.output_dim + 3, config.width),
            nn.GELU(),
            nn.Linear(config.width, config.width),
        )
        self.latent_token = nn.Parameter(torch.randn(1, 1, config.width) * 0.02)
        self.read = Block(config)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.encoder_depth)])

    def forward(self, batch: dict) -> LatentGeometry:
        positions, anchors = batch["points"], batch["anchors"]
        point_features = self.embed(torch.cat([self.position(positions), batch["normals"]], dim=-1))
        initial = self.embed(torch.cat([self.position(anchors), batch["anchor_normals"]], dim=-1))
        initial = initial + self.latent_token
        tokens = self.read(initial, point_features, anchors, positions, batch["point_weights"])
        for block in self.blocks:
            tokens = block(tokens, tokens, anchors, anchors, batch["anchor_weights"])
        return LatentGeometry(
            tokens,
            anchors,
            batch["anchor_normals"],
            batch["anchor_weights"],
            batch["center"],
            batch["scale"],
            batch["area"],
            initial,
        )


class SpatialDecoder(nn.Module):
    def __init__(self, config: ModelConfig, condition_dim: int = 0):
        super().__init__()
        self.config, self.condition_dim = config, condition_dim
        self.position = FourierFeatures(config.fourier_bands, config.min_wavelength, config.max_wavelength)
        self.embed = nn.Sequential(
            nn.Linear(self.position.output_dim, config.width),
            nn.GELU(),
            nn.Linear(config.width, config.width),
        )
        self.condition = nn.Linear(condition_dim, config.width) if condition_dim else None
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.decoder_depth)])
        self.norm = nn.LayerNorm(config.width)

    def forward(self, geometry: LatentGeometry, world_queries: torch.Tensor, conditions=None):
        if world_queries.ndim != 3 or world_queries.shape[-1] != 3:
            raise ValueError("Queries must have shape [batch, queries, 3]")
        if world_queries.shape[0] != geometry.tokens.shape[0]:
            raise ValueError("Query and geometry batch sizes differ")
        if self.condition is not None:
            if conditions is None or conditions.shape != (len(world_queries), self.condition_dim):
                raise ValueError("Missing or invalid operating/boundary conditions")
        elif conditions is not None:
            raise ValueError("This decoder was configured without conditions")
        results = []
        for world in world_queries.split(self.config.query_chunk, dim=1):
            query_positions = (world - geometry.center[:, None]) / geometry.scale[:, None]
            query = self.embed(self.position(query_positions))
            if self.condition is not None:
                query = query + self.condition(conditions)[:, None]
            # Queries never attend to each other: chunking or reordering them
            # cannot change the underlying coordinate-to-output function.
            for block in self.blocks:
                query = block(query, geometry.tokens, query_positions, geometry.positions, geometry.weights)
            results.append(self.norm(query))
        if not results:
            return geometry.tokens.new_empty((len(world_queries), 0, self.config.width))
        return torch.cat(results, dim=1)


class GeometryAutoencoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = GeometryEncoder(config)
        self.decoder = SpatialDecoder(config)
        self.distance_head, self.normal_head = nn.Linear(config.width, 1), nn.Linear(config.width, 3)

    def query_features(self, geometry: LatentGeometry, world_queries: torch.Tensor):
        return self.decoder(geometry, world_queries)

    def decode(self, geometry: LatentGeometry, world_queries: torch.Tensor):
        hidden = self.query_features(geometry, world_queries)
        normalized_distance = F.softplus(self.distance_head(hidden).float().squeeze(-1))
        normals = F.normalize(self.normal_head(hidden).float(), dim=-1, eps=1e-8)
        return dict(
            udf=normalized_distance * geometry.scale, normals=normals, normalized_udf=normalized_distance
        )

    def forward(self, batch: dict):
        geometry = self.encoder(batch)
        return self.decode(geometry, batch["queries"])


class FieldDecoder(nn.Module):
    """Optional downstream head. No query-field statistics or target leakage.

    Freeze the encoder if measuring transfer. Supply known operating conditions.
    Any target scaling must be fitted on the field-training split externally.
    """

    def __init__(self, config: ModelConfig, out_channels: int, condition_dim: int = 0):
        super().__init__()
        if out_channels < 1:
            raise ValueError("out_channels must be positive")
        if condition_dim < 0:
            raise ValueError("condition_dim must be nonnegative")
        self.condition_dim = condition_dim
        # Retain size, position and surface measure for dimensional physics.
        # Geometry reconstruction is similarity-covariant; a physical field need
        # not be. These metadata are inputs, never estimated from target fields.
        self.decoder = SpatialDecoder(config, condition_dim + 5)
        self.head = nn.Linear(config.width, out_channels)

    def forward(self, geometry: LatentGeometry, world_queries, conditions=None):
        if self.condition_dim:
            if conditions is None or conditions.shape != (len(geometry.tokens), self.condition_dim):
                raise ValueError("Missing or invalid operating/boundary conditions")
        elif conditions is not None:
            raise ValueError("This field head was configured without conditions")
        metadata = torch.cat(
            [geometry.center, geometry.scale.log(), (geometry.area / geometry.scale.square()).log()], dim=-1
        )
        if conditions is not None:
            metadata = torch.cat([metadata, conditions], dim=-1)
        return self.head(self.decoder(geometry, world_queries, metadata))
