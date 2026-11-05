"""Spatial vector tokens and coordinate-query decoders; no scalar token bottleneck."""

from __future__ import annotations

import math
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
    # the surfel readout's own, denser point set (surfels_per_token > 1); None reads the anchors
    surfel_positions: torch.Tensor | None = None
    surfel_normals: torch.Tensor | None = None
    surfel_weights: torch.Tensor | None = None
    # [B, 1]: 1 where the encoded input is the whole surface; None counts as complete
    complete: torch.Tensor | None = None
    # [B, 5]: the visible input's best mirror plane in normalized coordinates (normal, offset, fit score)
    mirror: torch.Tensor | None = None
    # the visible input's rigid symmetry maps x -> R x + t in normalized coordinates: R [B, K, 3, 3], t [B, K, 3],
    # region support [B, K], valid [B, K] (0 pads), and per map and surfel whether the map holds there [B, K, S]
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
            batch.get("surfels"),
            batch.get("surfel_normals"),
            batch.get("surfel_weights"),
            batch.get("complete"),
            batch.get("mirror"),
            symmetry_rotation=batch.get("symmetry_rotation"),
            symmetry_translation=batch.get("symmetry_translation"),
            symmetry_support=batch.get("symmetry_support"),
            symmetry_valid=batch.get("symmetry_valid"),
            symmetry_local=batch.get("symmetry_local"),
        )


class SpatialDecoder(nn.Module):
    def __init__(self, config: ModelConfig, condition_dim: int = 0, query_condition_dim: int = 0):
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
        # per-query inputs beside the position (the denoiser's hint); zero-initialised, so the
        # decoder starts as the position-only one
        self.query_condition = None
        if query_condition_dim:
            self.query_condition = nn.Linear(query_condition_dim, config.width)
            nn.init.zeros_(self.query_condition.weight)
            nn.init.zeros_(self.query_condition.bias)

    def forward(self, geometry: LatentGeometry, world_queries: torch.Tensor, conditions=None, query_conditions=None):
        if world_queries.ndim != 3 or world_queries.shape[-1] != 3:
            raise ValueError("Queries must have shape [batch, queries, 3]")
        if world_queries.shape[0] != geometry.tokens.shape[0]:
            raise ValueError("Query and geometry batch sizes differ")
        if self.condition is not None:
            if conditions is None or conditions.shape != (len(world_queries), self.condition_dim):
                raise ValueError("Missing or invalid operating/boundary conditions")
        elif conditions is not None:
            raise ValueError("This decoder was configured without conditions")
        if (self.query_condition is None) != (query_conditions is None):
            raise ValueError("Per-query conditions must be given exactly when the decoder was configured for them")
        if query_conditions is not None and query_conditions.shape[:2] != world_queries.shape[:2]:
            raise ValueError("Per-query conditions must match the queries")
        results = []
        extra = (
            query_conditions.split(self.config.query_chunk, dim=1)
            if query_conditions is not None
            else [None] * len(world_queries.split(self.config.query_chunk, dim=1))
        )
        for world, per_query in zip(world_queries.split(self.config.query_chunk, dim=1), extra, strict=True):
            query_positions = (world - geometry.center[:, None]) / geometry.scale[:, None]
            query = self.embed(self.position(query_positions))
            if self.condition is not None:
                query = query + self.condition(conditions)[:, None]
            if per_query is not None:
                query = query + self.query_condition(per_query.to(query.dtype))
            # Queries never attend to each other: chunking or reordering them
            # cannot change the underlying coordinate-to-output function.
            for block in self.blocks:
                query = block(query, geometry.tokens, query_positions, geometry.positions, geometry.weights)
            results.append(self.norm(query))
        if not results:
            return geometry.tokens.new_empty((len(world_queries), 0, self.config.width))
        return torch.cat(results, dim=1)


def symmetric_matrix(values: torch.Tensor) -> torch.Tensor:
    """[..., 6] -> symmetric [..., 3, 3] with the diagonal first, then xy, xz, yz."""
    xx, yy, zz, xy, xz, yz = values.unbind(-1)
    return torch.stack([xx, xy, xz, xy, yy, yz, xz, yz, zz], dim=-1).unflatten(-1, (3, 3))


def anchor_normal_prior(geometry: LatentGeometry, world_queries: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """Sign-free local normal from the packet's own anchors: a softmax over anchors of
    -|q - x_a|^2 / (2 sigma^2) + log(area_a), with sigma half of the shape's median
    anchor spacing, applied to the anchors' projectors n_a n_a^T. [B, Q, 3, 3]."""
    anchors = geometry.positions.float()
    valid = geometry.weights > 0
    count = anchors.shape[1]
    pairwise = torch.cdist(anchors, anchors)
    pairwise = pairwise.masked_fill(~valid[:, None, :] | torch.eye(count, dtype=torch.bool, device=anchors.device), torch.inf)
    nearest = pairwise.min(dim=-1).values.masked_fill(~valid, torch.nan)
    sigma = (0.5 * torch.nanmedian(nearest, dim=1).values).clamp_min(1e-6)  # [B]
    log_area = torch.where(valid, geometry.weights.float().clamp_min(1e-38).log(), -torch.inf)
    normals = geometry.normals.float()
    projectors = normals[..., :, None] * normals[..., None, :]  # [B, A, 3, 3]
    out = []
    for world in world_queries.split(chunk, dim=1):
        query = ((world - geometry.center[:, None]) / geometry.scale[:, None]).float()
        logits = -torch.cdist(query, anchors).square() / (2 * sigma.square())[:, None, None] + log_area[:, None]
        out.append(torch.einsum("bqa,baij->bqij", torch.softmax(logits, dim=-1), projectors))
    return torch.cat(out, dim=1)


UNION_TEMPERATURE = 0.01  # soft minimum width, in RMS radii
UNION_MARGIN = 0.01  # the disks overestimate by more than this: the nearer surface is unseen

# Denoising the surfel normal (normal_prior="surfels_denoise"). The surfel normal is read as the
# true normal plus noise of this level; training hints drawn from the true normal are rotated by an
# angle in HINT_ANGLES, which is also their noise level. The eps loss weights a query by
# (HINT_SIGMA / sigma)^2, capped, as the epsilon loss of a diffusion model weights its noise levels.
HINT_SIGMA = math.radians(10.0)
HINT_ANGLES = (math.radians(2.0), math.radians(60.0))
EPS_WEIGHT_CAP = 5.0
HINT_FEATURES = 10  # hint projector (6), noise level, surfel tensor's top eigenvalue, evidence (2)


def outer6(normals: torch.Tensor) -> torch.Tensor:
    """n n^T as [..., 6] in the symmetric_matrix layout: xx, yy, zz, xy, xz, yz."""
    x, y, z = normals.unbind(-1)
    return torch.stack([x * x, y * y, z * z, x * y, x * z, y * z], dim=-1)


def rotate_by(normals: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate unit vectors [..., 3] by angles [...] about random axes perpendicular to them."""
    axis = torch.randn_like(normals)
    axis = F.normalize(axis - (axis * normals).sum(-1, keepdim=True) * normals, dim=-1, eps=1e-8)
    return normals * torch.cos(angles)[..., None] + axis * torch.sin(angles)[..., None]


def _surfels(geometry: LatentGeometry):
    """The readout's disks: the packet's own surfels when it carries them, else its anchors."""
    if geometry.surfel_positions is not None:
        return geometry.surfel_positions.float(), geometry.surfel_normals.float(), geometry.surfel_weights > 0
    return geometry.positions.float(), geometry.normals.float(), geometry.weights > 0


def _anchor_spacing(geometry: LatentGeometry, anchors=None, valid=None) -> torch.Tensor:
    """Median nearest-neighbour distance between the valid anchors (or given points) of each shape. [B]"""
    if anchors is None:
        anchors, valid = geometry.positions.float(), geometry.weights > 0
    count = anchors.shape[1]
    pairwise = torch.cdist(anchors, anchors)
    eye = torch.eye(count, dtype=torch.bool, device=anchors.device)
    pairwise = pairwise.masked_fill(~valid[:, None, :] | eye, torch.inf)
    nearest = pairwise.min(dim=-1).values.masked_fill(~valid, torch.nan)
    return torch.nanmedian(nearest, dim=1).values.clamp_min(1e-6)


def surfel_readout(geometry: LatentGeometry, world_queries: torch.Tensor, *, neighbours: int = 16,
                   radius: float = 0.5, temperature: float = 0.3, chunk: int = 256):
    """Analytic decode of the packet as surfels: each anchor (or, when the packet carries them, each
    surfel) is a disk through its position, perpendicular to its normal, of radius `radius` x the
    median spacing.

    Returns the normalised distance to the nearest disk [B, Q], a sign-free normal tensor
    [B, Q, 3, 3] (the disks' projectors n n^T under a softmax of -(disk distance - nearest) /
    (temperature x spacing), so the nearest disk dominates and ties blend smoothly), and the
    evidence [B, Q, 2] for how far to trust them: how far outside its nearest disk the query
    projects, and how high above it, both in anchor spacings. Where the relevant surface is
    hidden, queries project outside the visible disks."""
    anchors, normals, valid = _surfels(geometry)
    normals = F.normalize(normals, dim=-1, eps=1e-8)
    spacing = _anchor_spacing(geometry, anchors, valid)
    count = min(neighbours, anchors.shape[1])
    distances, tensors, evidence = [], [], []
    for world in world_queries.split(chunk, dim=1):
        query = ((world - geometry.center[:, None]) / geometry.scale[:, None]).float()
        gap = torch.cdist(query, anchors).masked_fill(~valid[:, None, :], torch.inf)
        near, index = gap.topk(count, dim=-1, largest=False)
        gather = index[..., None].expand(-1, -1, -1, 3)
        points = torch.gather(anchors[:, None].expand(-1, query.shape[1], -1, -1), 2, gather)
        planes = torch.gather(normals[:, None].expand(-1, query.shape[1], -1, -1), 2, gather)
        offset = query[:, :, None, :] - points
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


SYMMETRY_FEATURES = 6  # image evidence (2), local hold, map support, image distance, direct distance


class GeometryAutoencoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = GeometryEncoder(config)
        self.denoise = config.normal_prior == "surfels_denoise"
        self.decoder = SpatialDecoder(
            config,
            condition_dim=1 if config.input_condition == "complete" else 0,
            query_condition_dim=HINT_FEATURES if self.denoise else 0,
        )
        self.distance_head, self.normal_head = (
            nn.Linear(config.width, 1),
            nn.Linear(config.width, 6 if config.normal_head == "projector" else 3),
        )
        if config.normal_prior in ("anchors", "surfels_residual", "surfels_denoise"):
            # the head learns a correction to the prior, so it starts at zero
            nn.init.zeros_(self.normal_head.weight)
            nn.init.zeros_(self.normal_head.bias)
        if config.distance_prior != "union" and (
            config.normal_prior in ("surfels", "surfels_residual") or config.distance_prior == "surfels"
        ):
            # per-query trust in the surfel readout [distance, normal]; starts at sigmoid(2) = 0.88
            self.prior_gate = nn.Linear(config.width, 2)
            nn.init.zeros_(self.prior_gate.weight)
            nn.init.constant_(self.prior_gate.bias, 2.0)
            # the trunk alone cannot tell visible from hidden surface; the readout's own evidence can
            self.prior_evidence = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 2))
            nn.init.zeros_(self.prior_evidence[-1].weight)
            nn.init.zeros_(self.prior_evidence[-1].bias)
        if config.mirror_prior != "none":
            # trust in the mirrored readout [distance, normal]; starts nearly off (logit -4)
            self.mirror_gate = nn.Linear(config.width, 2)
            nn.init.zeros_(self.mirror_gate.weight)
            nn.init.constant_(self.mirror_gate.bias, -4.0)
            self.mirror_evidence = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, 2))
            nn.init.zeros_(self.mirror_evidence[-1].weight)
            nn.init.zeros_(self.mirror_evidence[-1].bias)
        if config.symmetry_prior != "none":
            # trust in each symmetry map's carried-back readout [distance, normal]: a trunk term shared by the maps
            # plus a per-map term from the map's own evidence; starts nearly off (logit -4)
            self.symmetry_gate = nn.Linear(config.width, 2)
            nn.init.zeros_(self.symmetry_gate.weight)
            nn.init.constant_(self.symmetry_gate.bias, -4.0)
            self.symmetry_evidence = nn.Sequential(nn.Linear(SYMMETRY_FEATURES, 32), nn.SiLU(), nn.Linear(32, 2))
            nn.init.zeros_(self.symmetry_evidence[-1].weight)
            nn.init.zeros_(self.symmetry_evidence[-1].bias)

    def _hint(self, readout, clean_normals=None, clean_valid=None):
        """The denoiser's per-query input: the surfel normal as a noisy observation of the true one.
        In training, a hint_noise fraction of the queries with a target instead get the true normal
        rotated by a random angle, which is then their noise level."""
        _, prior_tensor, evidence = readout
        with torch.no_grad():
            values, vectors = torch.linalg.eigh(prior_tensor.float())
        hint, top = vectors[..., -1], values[..., -1]
        sigma = torch.full_like(top, HINT_SIGMA)
        if self.training and self.config.hint_noise > 0 and clean_normals is not None:
            low, high = HINT_ANGLES
            angle = low + (high - low) * torch.rand_like(top)
            noisy = rotate_by(F.normalize(clean_normals.float(), dim=-1, eps=1e-8), angle)
            use = (torch.rand_like(top) < self.config.hint_noise) & clean_valid.bool()
            hint = torch.where(use[..., None], noisy, hint)
            sigma = torch.where(use, angle, sigma)
            top = torch.where(use, torch.ones_like(top), top)
        features = torch.cat(
            [outer6(hint), sigma[..., None], top[..., None], torch.log1p(evidence.float().clamp(0, 100))], dim=-1
        )
        return hint, sigma, features

    def _conditions(self, geometry: LatentGeometry):
        if self.config.input_condition != "complete":
            return None
        if geometry.complete is None:
            return geometry.tokens.new_ones(len(geometry.tokens), 1)
        return geometry.complete.to(geometry.tokens.dtype)

    def query_features(self, geometry: LatentGeometry, world_queries: torch.Tensor):
        conditions = self._conditions(geometry)
        if self.denoise:
            readout = surfel_readout(geometry, world_queries, chunk=self.config.query_chunk)
            return self.decoder(geometry, world_queries, conditions, query_conditions=self._hint(readout)[2])
        return self.decoder(geometry, world_queries, conditions)

    def decode(self, geometry: LatentGeometry, world_queries: torch.Tensor, clean_normals=None, clean_valid=None):
        extras = {}
        surfels = (
            self.config.normal_prior in ("surfels", "surfels_residual", "surfels_denoise")
            or self.config.distance_prior != "none"
        )
        if surfels:
            readout = surfel_readout(geometry, world_queries, chunk=self.config.query_chunk)
            prior_distance, prior_tensor, evidence = readout
        if self.denoise:
            hint, sigma, features = self._hint(readout, clean_normals, clean_valid)
            hidden = self.decoder(geometry, world_queries, self._conditions(geometry), query_conditions=features)
        else:
            hidden = self.query_features(geometry, world_queries)
        normalized_distance = F.softplus(self.distance_head(hidden).float().squeeze(-1))
        if surfels:
            if self.config.distance_prior == "union":
                unseen = normalized_distance  # the head's distance to the part the disks do not cover
                branches = -torch.stack([prior_distance, unseen], dim=-1) / UNION_TEMPERATURE
                normalized_distance = -UNION_TEMPERATURE * torch.logsumexp(branches, dim=-1)
                normal_gate = torch.softmax(branches, dim=-1)[..., 0]
                extras = dict(prior_distance=prior_distance, unseen_distance=unseen)
            elif getattr(self, "prior_gate", None) is not None:
                logits = self.prior_gate(hidden).float() + self.prior_evidence(
                    torch.log1p(evidence.clamp(0, 100))
                )
                if self.config.symmetry_prior != "none" and geometry.symmetry_rotation is not None:
                    sym_distance, sym_tensor, sym_logits = self._symmetry(geometry, world_queries, hidden,
                                                                          prior_distance)
                    # (K + 2)-way blend per quantity: direct surfels, each map's carried readout, heads (logit 0)
                    stacked = torch.cat([logits[..., None, :], sym_logits, torch.zeros_like(logits)[..., None, :]], -2)
                    weights = torch.softmax(stacked, dim=-2)  # [B, Q, K + 2, 2]
                    candidates = torch.cat([prior_distance[..., None], sym_distance, normalized_distance[..., None]], -1)
                    normalized_distance = (weights[..., 0] * candidates).sum(-1)
                    symmetry_weights = weights[..., 1]  # normal weights [B, Q, K + 2]
                    extras["symmetry_weight"] = symmetry_weights[..., 1:-1].sum(-1)
                elif self.config.mirror_prior != "none" and geometry.mirror is not None:
                    mirror_distance, mirror_tensor, mirror_logits = self._mirror(geometry, world_queries, hidden)
                    # three-way blend per quantity: direct surfels, mirrored surfels, heads (logit 0)
                    weights = torch.softmax(torch.stack([logits, mirror_logits, torch.zeros_like(logits)], -1), -1)
                    normal_weights = weights[..., 1, :]
                    normalized_distance = (weights[..., 0, 0] * prior_distance + weights[..., 0, 1] * mirror_distance
                                           + weights[..., 0, 2] * normalized_distance)
                    extras["mirror_weight"] = weights[..., 1, 1]
                else:
                    gate = torch.sigmoid(logits)
                    normal_gate = gate[..., 1]
                    if self.config.distance_prior == "surfels":
                        normalized_distance = gate[..., 0] * prior_distance + (1 - gate[..., 0]) * normalized_distance
        if self.config.distance_prior != "none":
            # Complete inputs use the measured disks; completion is only needed for partial inputs.
            complete = geometry.complete
            if complete is None:
                complete = torch.ones_like(geometry.scale)
            normalized_distance = torch.where(complete == 1, prior_distance, normalized_distance)
        if self.config.normal_head == "projector":
            tensor = symmetric_matrix(self.normal_head(hidden).float())
            if self.config.normal_prior == "anchors":
                tensor = tensor + anchor_normal_prior(geometry, world_queries, self.config.query_chunk)
            elif self.config.normal_prior == "surfels" and "symmetry_weight" in extras:
                w = symmetry_weights[..., None, None]
                tensor = (w[..., 0, :, :] * prior_tensor + (w[..., 1:-1, :, :] * sym_tensor).sum(-3)
                          + w[..., -1, :, :] * tensor)
            elif self.config.normal_prior == "surfels" and "mirror_weight" in extras:
                w = normal_weights[..., None, None, :]
                tensor = w[..., 0] * prior_tensor + w[..., 1] * mirror_tensor + w[..., 2] * tensor
            elif self.config.normal_prior == "surfels":
                tensor = normal_gate[..., None, None] * prior_tensor + (1 - normal_gate[..., None, None]) * tensor
            elif self.config.normal_prior == "surfels_residual":
                tensor = normal_gate[..., None, None] * prior_tensor + tensor
            elif self.denoise:
                projector = symmetric_matrix(outer6(hint))
                if self.config.normal_target == "eps":
                    # the head predicts the noise (hint - truth) / sigma; the loss on the noise is the
                    # loss on the clean tensor over sigma^2, rescaled to 1 at the surfel noise level
                    tensor = projector - sigma[..., None, None] * tensor
                    weight = (HINT_SIGMA / sigma).square().clamp(max=EPS_WEIGHT_CAP)
                else:
                    tensor = projector + tensor
                    weight = torch.ones_like(sigma)
                extras["normal_loss_weight"] = weight
            # The loss trains the matrix; the eigenvector is a readout, so no
            # gradient is taken through the eigendecomposition.
            normals = torch.linalg.eigh(tensor.detach())[1][..., -1]
            return dict(
                udf=normalized_distance * geometry.scale,
                normals=normals,
                normalized_udf=normalized_distance,
                normal_tensor=tensor,
                **extras,
            )
        normals = F.normalize(self.normal_head(hidden).float(), dim=-1, eps=1e-8)
        return dict(
            udf=normalized_distance * geometry.scale, normals=normals, normalized_udf=normalized_distance, **extras
        )

    def _mirror(self, geometry: LatentGeometry, world_queries: torch.Tensor, hidden: torch.Tensor):
        """The surfel readout at each query's mirror image across the packet's mirror plane, reflected back, and the
        gate logits for trusting it."""
        plane = geometry.mirror.float()
        normal, offset, fit = plane[:, :3], plane[:, 3], plane[:, 4]
        scale, center = geometry.scale[:, None].float(), geometry.center[:, None].float()
        query = (world_queries.float() - center) / scale
        signed = (query * normal[:, None]).sum(-1) - offset[:, None]
        image = (query - 2 * signed[..., None] * normal[:, None]) * scale + center
        distance, tensor, evidence = surfel_readout(geometry, image, chunk=self.config.query_chunk)
        reflect = torch.eye(3, device=plane.device) - 2 * normal[:, :, None] * normal[:, None, :]
        tensor = reflect[:, None] @ tensor @ reflect[:, None]
        features = torch.cat([torch.log1p(evidence.clamp(0, 100)), fit[:, None, None].expand(-1, evidence.shape[1], 1)], -1)
        logits = self.mirror_gate(hidden).float() + self.mirror_evidence(features)
        return distance, tensor, logits

    def _symmetry(self, geometry: LatentGeometry, world_queries: torch.Tensor, hidden: torch.Tensor,
                  prior_distance: torch.Tensor):
        """Each symmetry map's surfel readout at the queries' images, carried back, and the gate logits for trusting
        it: distance [B, Q, K], tensor [B, Q, K, 3, 3], logits [B, Q, K, 2]."""
        rotation = geometry.symmetry_rotation.float()
        translation = geometry.symmetry_translation.float()
        batch, maps = rotation.shape[:2]
        count = world_queries.shape[1]
        scale, center = geometry.scale[:, None].float(), geometry.center[:, None].float()
        query = (world_queries.float() - center) / scale  # [B, Q, 3]
        image = torch.einsum("bkij,bqj->bkqi", rotation, query) + translation[:, :, None]
        world_image = (image * scale[:, None] + center[:, None]).reshape(batch, maps * count, 3)
        distance, tensor, evidence = surfel_readout(geometry, world_image, chunk=self.config.query_chunk)
        distance = distance.view(batch, maps, count)
        evidence = evidence.view(batch, maps, count, 2)
        # a map carries normals by R, so the image's projector comes back as R^T P R
        tensor = torch.einsum("bkji,bkqjl,bklm->bkqim", rotation, tensor.view(batch, maps, count, 3, 3), rotation)
        # does each map hold on the disks around the query: its per-surfel flags over the 8 nearest disks
        anchors, _, valid = _surfels(geometry)
        near = torch.cdist(query, anchors).masked_fill(~valid[:, None, :], torch.inf).topk(
            min(8, anchors.shape[1]), dim=-1, largest=False).indices  # [B, Q, 8]
        flags = geometry.symmetry_local.float()[..., : anchors.shape[1]]  # [B, K, S]
        local = torch.gather(flags, 2, near.reshape(batch, 1, -1).expand(-1, maps, -1)).view(
            batch, maps, count, -1).mean(-1)
        spacing = _anchor_spacing(geometry, anchors, valid)[:, None, None]
        features = torch.stack([
            torch.log1p(evidence[..., 0].clamp(0, 100)),
            torch.log1p(evidence[..., 1].clamp(0, 100)),
            local,
            geometry.symmetry_support.float()[:, :, None].expand(-1, -1, count),
            torch.log1p((distance / spacing).clamp(0, 100)),
            torch.log1p((prior_distance.float()[:, None] / spacing).clamp(0, 100)).expand(-1, maps, -1),
        ], -1)  # [B, K, Q, SYMMETRY_FEATURES]
        logits = self.symmetry_gate(hidden).float()[:, None] + self.symmetry_evidence(features)
        logits = logits.masked_fill(geometry.symmetry_valid[:, :, None, None] <= 0, -1e4)
        return distance.transpose(1, 2), tensor.transpose(1, 2), logits.transpose(1, 2)

    def forward(self, batch: dict):
        geometry = self.encoder(batch)
        if self.denoise and self.training and self.config.hint_noise > 0:
            # noised true normals are training inputs to the denoiser, never evaluation inputs
            return self.decode(geometry, batch["queries"], batch["query_normals"], batch["normal_valid"])
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
