"""Encoder, query decoder and the disk readout that decodes a packet."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class FourierFeatures(nn.Module):
    def __init__(self, bands: int, low: float, high: float):
        super().__init__()
        self.register_buffer("frequencies", 2 * torch.pi / torch.logspace(
            torch.log10(torch.tensor(high)).item(), torch.log10(torch.tensor(low)).item(), bands))
        self.output_dim = 3 + 6 * bands

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        angles = positions.float()[..., :, None] * self.frequencies.float()
        return torch.cat([positions.float(), angles.sin().flatten(-2), angles.cos().flatten(-2)], dim=-1)


class Rotary3D(nn.Module):
    def __init__(self, head_dim: int, low: float, high: float):
        super().__init__()
        frequencies = 2 * torch.pi / torch.logspace(
            torch.log10(torch.tensor(high)).item(), torch.log10(torch.tensor(low)).item(), head_dim // 6)
        self.register_buffer("frequencies", frequencies.repeat(3))
        self.register_buffer("axes", torch.arange(3).repeat_interleave(head_dim // 6))

    def forward(self, value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        angles = positions.float()[..., self.axes] * self.frequencies.float()
        cosine, sine = angles.cos()[:, None], angles.sin()[:, None]
        a, b = value.float().unflatten(-1, (-1, 2)).unbind(-1)
        return torch.stack([a * cosine - b * sine, a * sine + b * cosine], dim=-1).flatten(-2).to(value.dtype)


class WeightedAttention(nn.Module):
    """Attention whose keys carry area weights: log(weight) is added to the logits."""

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
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=log_weights[:, None, None].to(q.dtype),
                                                dropout_p=self.dropout if self.training else 0.0)
        return self.out(output.transpose(1, 2).flatten(-2))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.query_norm, self.value_norm = nn.LayerNorm(config.width), nn.LayerNorm(config.width)
        self.attention = WeightedAttention(config)
        self.ff_norm = nn.LayerNorm(config.width)
        self.ff = nn.Sequential(nn.Linear(config.width, config.width * config.ff_multiplier), nn.GELU(),
                                nn.Dropout(config.dropout), nn.Linear(config.width * config.ff_multiplier, config.width))

    def forward(self, query, values, query_positions, value_positions, weights):
        query = query + self.attention(self.query_norm(query), self.value_norm(values), query_positions,
                                       value_positions, weights)
        return query + self.ff(self.ff_norm(query))


@dataclass
class LatentGeometry:
    """A packet: tokens at anchor positions (normalized frame) with normals and area weights, the frame, the
    readout's surfels, whether the input was the whole surface, and the input's symmetry maps."""

    tokens: torch.Tensor
    positions: torch.Tensor
    normals: torch.Tensor
    weights: torch.Tensor
    center: torch.Tensor
    scale: torch.Tensor
    area: torch.Tensor
    surfel_positions: torch.Tensor | None = None  # None: the readout uses the anchors
    surfel_normals: torch.Tensor | None = None
    surfel_weights: torch.Tensor | None = None
    complete: torch.Tensor | None = None  # [B, 1]; None counts as complete
    # maps x -> R x + t: R [B, K, 3, 3], t [B, K, 3], support [B, K], valid [B, K], per-surfel hold [B, K, S]
    symmetry_rotation: torch.Tensor | None = None
    symmetry_translation: torch.Tensor | None = None
    symmetry_support: torch.Tensor | None = None
    symmetry_valid: torch.Tensor | None = None
    symmetry_local: torch.Tensor | None = None

    @property
    def world_positions(self):
        return self.positions * self.scale[:, None] + self.center[:, None]

    def with_tokens(self, tokens: torch.Tensor) -> "LatentGeometry":
        return replace(self, tokens=tokens)


PACKET_KEYS = {"surfel_positions": "surfels", "surfel_normals": "surfel_normals", "surfel_weights": "surfel_weights",
               "complete": "complete", "symmetry_rotation": "symmetry_rotation",
               "symmetry_translation": "symmetry_translation", "symmetry_support": "symmetry_support",
               "symmetry_valid": "symmetry_valid", "symmetry_local": "symmetry_local"}


def packet_extras(batch: dict) -> dict:
    """The optional packet fields a batch carries."""
    return {name: batch.get(key) for name, key in PACKET_KEYS.items()}


class GeometryEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.position = FourierFeatures(config.fourier_bands, config.min_wavelength, config.max_wavelength)
        self.embed = nn.Sequential(nn.Linear(self.position.output_dim + 3, config.width), nn.GELU(),
                                   nn.Linear(config.width, config.width))
        self.latent_token = nn.Parameter(torch.randn(1, 1, config.width) * 0.02)
        self.read = Block(config)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.encoder_depth)])

    def tokens(self, point_features, anchor_features, batch: dict) -> torch.Tensor:
        anchors = batch["anchors"]
        tokens = self.read(anchor_features + self.latent_token, point_features, anchors, batch["points"],
                           batch["point_weights"])
        for block in self.blocks:
            tokens = block(tokens, tokens, anchors, anchors, batch["anchor_weights"])
        return tokens

    def forward(self, batch: dict) -> LatentGeometry:
        points = self.embed(torch.cat([self.position(batch["points"]), batch["normals"]], dim=-1))
        anchors = self.embed(torch.cat([self.position(batch["anchors"]), batch["anchor_normals"]], dim=-1))
        return LatentGeometry(self.tokens(points, anchors, batch), batch["anchors"], batch["anchor_normals"],
                              batch["anchor_weights"], batch["center"], batch["scale"], batch["area"],
                              **packet_extras(batch))


class SpatialDecoder(nn.Module):
    """Each query attends to the tokens only, never to other queries, so outputs do not depend on batching."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.position = FourierFeatures(config.fourier_bands, config.min_wavelength, config.max_wavelength)
        self.embed = nn.Sequential(nn.Linear(self.position.output_dim, config.width), nn.GELU(),
                                   nn.Linear(config.width, config.width))
        self.condition = nn.Linear(1, config.width)  # the completeness flag
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.decoder_depth)])
        self.norm = nn.LayerNorm(config.width)

    def forward(self, geometry: LatentGeometry, world_queries: torch.Tensor, complete: torch.Tensor):
        results = []
        for world in world_queries.split(self.config.query_chunk, dim=1):
            positions = (world - geometry.center[:, None]) / geometry.scale[:, None]
            query = self.embed(self.position(positions)) + self.condition(complete)[:, None]
            for block in self.blocks:
                query = block(query, geometry.tokens, positions, geometry.positions, geometry.weights)
            results.append(self.norm(query))
        if not results:
            return geometry.tokens.new_empty((len(world_queries), 0, self.config.width))
        return torch.cat(results, dim=1)


def symmetric_matrix(values: torch.Tensor) -> torch.Tensor:
    """[..., 6] -> symmetric [..., 3, 3]; layout xx, yy, zz, xy, xz, yz."""
    xx, yy, zz, xy, xz, yz = values.unbind(-1)
    return torch.stack([xx, xy, xz, xy, yy, yz, xz, yz, zz], dim=-1).unflatten(-1, (3, 3))


def _surfels(geometry: LatentGeometry):
    if geometry.surfel_positions is not None:
        return geometry.surfel_positions.float(), geometry.surfel_normals.float(), geometry.surfel_weights > 0
    return geometry.positions.float(), geometry.normals.float(), geometry.weights > 0


def _spacing(points: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Median nearest-neighbour distance between the valid points of each shape. [B]"""
    pairwise = torch.cdist(points, points)
    eye = torch.eye(points.shape[1], dtype=torch.bool, device=points.device)
    pairwise = pairwise.masked_fill(~valid[:, None, :] | eye, torch.inf)
    nearest = pairwise.min(dim=-1).values.masked_fill(~valid, torch.nan)
    return torch.nanmedian(nearest, dim=1).values.clamp_min(1e-6)


def surfel_readout(geometry: LatentGeometry, world_queries: torch.Tensor, *, neighbours: int = 16,
                   radius: float = 0.5, temperature: float = 0.3, chunk: int = 256):
    """Decode the packet as disks: each surfel is a disk through its position, perpendicular to its normal, of radius
    `radius` x the median surfel spacing.

    Returns the normalized distance to the nearest disk [B, Q]; a sign-free normal tensor [B, Q, 3, 3], the disks'
    projectors n nᵀ under a softmax of -(disk distance - nearest) / (temperature x spacing); and evidence [B, Q, 2]:
    how far outside its nearest disk the query projects and how high above it, in spacings. Over hidden surface,
    queries project outside every disk."""
    points, normals, valid = _surfels(geometry)
    normals = F.normalize(normals, dim=-1, eps=1e-8)
    spacing = _spacing(points, valid)
    count = min(neighbours, points.shape[1])
    distances, tensors, evidence = [], [], []
    for world in world_queries.split(chunk, dim=1):
        query = ((world - geometry.center[:, None]) / geometry.scale[:, None]).float()
        near, index = torch.cdist(query, points).masked_fill(~valid[:, None, :], torch.inf).topk(
            count, dim=-1, largest=False)
        gather = index[..., None].expand(-1, -1, -1, 3)
        centers = torch.gather(points[:, None].expand(-1, query.shape[1], -1, -1), 2, gather)
        planes = torch.gather(normals[:, None].expand(-1, query.shape[1], -1, -1), 2, gather)
        offset = query[:, :, None, :] - centers
        height = (offset * planes).sum(-1)
        lateral = (offset.square().sum(-1) - height.square()).clamp_min(0).add(1e-12).sqrt()
        rim = F.relu(lateral - radius * spacing[:, None, None])
        disk = (height.square() + rim.square() + 1e-12).sqrt().masked_fill(~torch.isfinite(near), torch.inf)
        nearest, which = disk.min(dim=-1)
        weight = torch.softmax(-(disk - nearest[..., None]) / (temperature * spacing)[:, None, None], dim=-1)
        distances.append(nearest)
        tensors.append(torch.einsum("bqk,bqki,bqkj->bqij", weight, planes, planes))
        pick = which[..., None]
        evidence.append(torch.stack([rim.gather(-1, pick)[..., 0], height.abs().gather(-1, pick)[..., 0]], -1)
                        / spacing[:, None, None])
    return torch.cat(distances, dim=1), torch.cat(tensors, dim=1), torch.cat(evidence, dim=1)


SYMMETRY_FEATURES = 6  # image evidence (2), hold near the query, map support, image distance, direct distance


class GeometryAutoencoder(nn.Module):
    """Predicts, at any point, the distance to the whole surface and its normal as a line (n nᵀ).

    Per query and quantity, a softmax blends the disk readout at the query, the disk readout at the query's image
    under each symmetry map (carried back), and the learned heads. The disk and map logits come from the decoder
    state plus the readouts' own evidence; the heads' logit is 0."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = GeometryEncoder(config)
        self.decoder = SpatialDecoder(config)
        self.distance_head = nn.Linear(config.width, 1)
        self.normal_head = nn.Linear(config.width, 6)
        self.prior_gate = nn.Linear(config.width, 2)  # starts trusting the disks: logit 2
        nn.init.zeros_(self.prior_gate.weight)
        nn.init.constant_(self.prior_gate.bias, 2.0)
        self.prior_evidence = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 2))
        nn.init.zeros_(self.prior_evidence[-1].weight)
        nn.init.zeros_(self.prior_evidence[-1].bias)
        if config.symmetry_maps:
            self.symmetry_gate = nn.Linear(config.width, 2)  # starts nearly closed: logit -4
            nn.init.zeros_(self.symmetry_gate.weight)
            nn.init.constant_(self.symmetry_gate.bias, -4.0)
            self.symmetry_evidence = nn.Sequential(nn.Linear(SYMMETRY_FEATURES, 32), nn.SiLU(), nn.Linear(32, 2))
            nn.init.zeros_(self.symmetry_evidence[-1].weight)
            nn.init.zeros_(self.symmetry_evidence[-1].bias)

    def query_features(self, geometry: LatentGeometry, world_queries: torch.Tensor) -> torch.Tensor:
        complete = geometry.complete
        if complete is None:
            complete = geometry.tokens.new_ones(len(geometry.tokens), 1)
        return self.decoder(geometry, world_queries, complete.to(geometry.tokens.dtype))

    def decode(self, geometry: LatentGeometry, world_queries: torch.Tensor) -> dict:
        disk_distance, disk_tensor, evidence = surfel_readout(geometry, world_queries, chunk=self.config.query_chunk)
        hidden = self.query_features(geometry, world_queries)
        head_distance = F.softplus(self.distance_head(hidden).float().squeeze(-1))
        head_tensor = symmetric_matrix(self.normal_head(hidden).float())
        logits = self.prior_gate(hidden).float() + self.prior_evidence(torch.log1p(evidence.clamp(0, 100)))
        distances = [disk_distance[..., None], head_distance[..., None]]
        tensors = [disk_tensor[..., None, :, :], head_tensor[..., None, :, :]]
        logits = [logits[..., None, :], torch.zeros_like(logits)[..., None, :]]
        extras = {}
        if self.config.symmetry_maps and geometry.symmetry_rotation is not None:
            map_distance, map_tensor, map_logits = self._symmetry(geometry, world_queries, hidden, disk_distance)
            distances.insert(1, map_distance)
            tensors.insert(1, map_tensor)
            logits.insert(1, map_logits)
        weights = torch.softmax(torch.cat(logits, dim=-2), dim=-2)  # [B, Q, candidates, (distance, normal)]
        distance = (weights[..., 0] * torch.cat(distances, dim=-1)).sum(-1)
        tensor = (weights[..., 1, None, None] * torch.cat(tensors, dim=-3)).sum(-3)
        if len(logits) == 3:
            extras["symmetry_weight"] = weights[..., 1:-1, 1].sum(-1)
        # the loss trains the matrix; the eigenvector is a readout, with no gradient through eigh
        normals = torch.linalg.eigh(tensor.detach())[1][..., -1]
        return dict(udf=distance * geometry.scale, normalized_udf=distance, normals=normals, normal_tensor=tensor,
                    **extras)

    def _symmetry(self, geometry: LatentGeometry, world_queries: torch.Tensor, hidden: torch.Tensor,
                  disk_distance: torch.Tensor):
        """Each map's disk readout at the queries' images, carried back, and its gate logits:
        distance [B, Q, K], tensor [B, Q, K, 3, 3], logits [B, Q, K, 2]."""
        rotation, translation = geometry.symmetry_rotation.float(), geometry.symmetry_translation.float()
        batch, maps = rotation.shape[:2]
        count = world_queries.shape[1]
        scale, center = geometry.scale[:, None].float(), geometry.center[:, None].float()
        query = (world_queries.float() - center) / scale
        image = torch.einsum("bkij,bqj->bkqi", rotation, query) + translation[:, :, None]
        world_image = (image * scale[:, None] + center[:, None]).reshape(batch, maps * count, 3)
        distance, tensor, evidence = surfel_readout(geometry, world_image, chunk=self.config.query_chunk)
        distance = distance.view(batch, maps, count)
        evidence = evidence.view(batch, maps, count, 2)
        # a map carries normals by R, so the image's projector comes back as Rᵀ P R
        tensor = torch.einsum("bkji,bkqjl,bklm->bkqim", rotation, tensor.view(batch, maps, count, 3, 3), rotation)
        # does the map hold on the 8 disks nearest the query
        points, _, valid = _surfels(geometry)
        near = torch.cdist(query, points).masked_fill(~valid[:, None, :], torch.inf).topk(
            min(8, points.shape[1]), dim=-1, largest=False).indices
        flags = geometry.symmetry_local.float()[..., : points.shape[1]]
        local = torch.gather(flags, 2, near.reshape(batch, 1, -1).expand(-1, maps, -1)).view(
            batch, maps, count, -1).mean(-1)
        spacing = _spacing(points, valid)[:, None, None]
        features = torch.stack([
            torch.log1p(evidence[..., 0].clamp(0, 100)),
            torch.log1p(evidence[..., 1].clamp(0, 100)),
            local,
            geometry.symmetry_support.float()[:, :, None].expand(-1, -1, count),
            torch.log1p((distance / spacing).clamp(0, 100)),
            torch.log1p((disk_distance.float()[:, None] / spacing).clamp(0, 100)).expand(-1, maps, -1),
        ], -1)
        logits = self.symmetry_gate(hidden).float()[:, None] + self.symmetry_evidence(features)
        logits = logits.masked_fill(geometry.symmetry_valid[:, :, None, None] <= 0, -1e4)
        return distance.transpose(1, 2), tensor.transpose(1, 2), logits.transpose(1, 2)

    def forward(self, batch: dict) -> dict:
        return self.decode(self.encoder(batch), batch["queries"])
