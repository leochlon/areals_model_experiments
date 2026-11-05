"""Validated, serializable configuration with no silently ignored keys."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _finite_config(value):
    for name, item in vars(value).items():
        if isinstance(item, (float, int)) and (isinstance(item, bool) or not math.isfinite(item)):
            raise ValueError(f"{name} must be finite and numeric, not boolean")
    integer_names = {
        "width",
        "heads",
        "latents",
        "encoder_depth",
        "decoder_depth",
        "ff_multiplier",
        "fourier_bands",
        "query_chunk",
        "surfels_per_token",
        "symmetry_maps",
        "encoder_points",
        "queries",
        "mask_patches",
        "epochs",
        "batch_size",
        "accumulation",
        "warmup_epochs",
        "warmup_steps",
        "workers",
        "seed",
    }
    for name, item in vars(value).items():
        if name in integer_names and not isinstance(item, int):
            raise ValueError(f"{name} must be an integer")


@dataclass(frozen=True)
class ModelConfig:
    width: int = 192
    heads: int = 4
    latents: int = 256
    encoder_depth: int = 4
    decoder_depth: int = 2
    ff_multiplier: int = 4
    fourier_bands: int = 8
    min_wavelength: float = 0.05
    max_wavelength: float = 4.0
    dropout: float = 0.0
    query_chunk: int = 256
    # "vector" predicts a unit normal. "projector" predicts the symmetric matrix
    # n n^T and reads the normal as its top eigenvector: a normal as a line, not an
    # arrow. Around an open boundary edge, or wherever the sign convention changes,
    # a unit-vector field must flip sign across a wall of near-90-degree errors;
    # the projector is sign-free and has no such walls. Unoriented only.
    normal_head: str = "vector"
    # "anchors" (projector head only): start every query from the kernel-weighted
    # average of its nearby anchors' normal projectors, which the packet already
    # carries, and let the head learn the correction. The anchors' own normals are
    # a better guess than a decoder rebuilding orientation from the tokens alone.
    normal_prior: str = "none"
    # "surfels" reads the packet as a surfel cloud: every anchor is a disk with its
    # normal, radius half the median anchor spacing. The distance to the nearest
    # disk and a soft-min-weighted blend of the nearby disks' normal projectors
    # are exact on visible geometry; a learned per-query gate mixes them with the
    # heads, which must take over where the relevant surface is hidden.
    # "union": the surface is the visible surfels plus whatever is hidden, and the distance to a
    # union is the minimum of the distances. The distance head learns the distance to the unseen
    # part only; a soft minimum with the surfel distance gives the prediction and says, per query,
    # which part is nearer, which also decides whose normal is used. No learned gate.
    # normal_prior "surfels_residual" adds the head to the (weighted) surfel projector instead of
    # blending, so on visible surface it learns corrections to the flat disks.
    distance_prior: str = "none"
    # normal_prior "surfels_denoise" treats the surfel normal as a noisy observation, a hint: the
    # decoder reads it (with its noise level and the readout's evidence) as a per-query input and
    # the head removes the noise. normal_target "residual" predicts the correction to the hint;
    # "eps" predicts the noise itself, scaled by the noise level, and subtracts it, diffusion style.
    # hint_noise is the fraction of training queries whose hint is the true normal rotated by a
    # random angle instead of the surfel normal, so the head learns a general denoiser.
    normal_target: str = "residual"
    hint_noise: float = 0.0
    # The surfel readout's resolution. Each token keeps this many surfels (position and normal),
    # the farthest-point samples of its input that extend the anchor set: 1 means the anchors
    # themselves. The readout's accuracy on visible geometry is set by surfel spacing, and a
    # surfel costs six numbers against a token's `width`.
    surfels_per_token: int = 1
    # "complete" tells the decoder, per shape, whether its input is the whole surface. Masked
    # pretraining otherwise teaches the surfel gate a caution that is wrong for complete inputs,
    # which is how downstream geometry arrives. Packets saved without the flag count as complete.
    input_condition: str = "none"
    # "surfels": a third prior beside the surfels and the heads. The visible input's best mirror plane is found by
    # reflecting it onto itself; the surfel readout at each query's mirror image, reflected back, is blended in by
    # a learned gate fed with how well the plane fits and how far the mirror image lies from any disk. Needs the
    # gated surfel priors (normal_prior and distance_prior "surfels").
    mirror_prior: str = "none"
    # "maps": candidates beside the surfels and the heads from the visible input's own rigid symmetries: reflections,
    # rotations and translations between repeated parts, found by transformation-space voting on curvature frames.
    # The surfel readout at each query's image under each of the best `symmetry_maps` maps, carried back, is blended
    # in by a learned gate fed with whether the map holds on the disks around the query, how far the image lies from
    # any disk, and the map's support. Needs the gated surfel priors; not combined with mirror_prior.
    symmetry_prior: str = "none"
    symmetry_maps: int = 8

    def __post_init__(self):
        _finite_config(self)
        if self.normal_head not in ("vector", "projector"):
            raise ValueError("normal_head must be vector or projector")
        if self.normal_prior not in ("none", "anchors", "surfels", "surfels_residual", "surfels_denoise"):
            raise ValueError("normal_prior must be none, anchors, surfels, surfels_residual or surfels_denoise")
        if self.normal_prior != "none" and self.normal_head != "projector":
            raise ValueError("A normal prior needs normal_head=projector")
        if self.normal_target not in ("residual", "eps"):
            raise ValueError("normal_target must be residual or eps")
        if not 0 <= self.hint_noise <= 1:
            raise ValueError("hint_noise must be between 0 and 1")
        if (self.normal_target != "residual" or self.hint_noise > 0) and self.normal_prior != "surfels_denoise":
            raise ValueError("normal_target and hint_noise need normal_prior=surfels_denoise")
        if self.mirror_prior not in ("none", "surfels"):
            raise ValueError("mirror_prior must be none or surfels")
        if self.mirror_prior != "none" and (self.normal_prior != "surfels" or self.distance_prior != "surfels"):
            raise ValueError("mirror_prior needs normal_prior and distance_prior surfels")
        if self.symmetry_prior not in ("none", "maps"):
            raise ValueError("symmetry_prior must be none or maps")
        if self.symmetry_prior != "none" and (self.normal_prior != "surfels" or self.distance_prior != "surfels"):
            raise ValueError("symmetry_prior needs normal_prior and distance_prior surfels")
        if self.symmetry_prior != "none" and self.mirror_prior != "none":
            raise ValueError("symmetry_prior and mirror_prior are alternatives")
        if self.symmetry_maps < 1:
            raise ValueError("symmetry_maps must be positive")
        if self.input_condition not in ("none", "complete"):
            raise ValueError("input_condition must be none or complete")
        if self.surfels_per_token < 1:
            raise ValueError("surfels_per_token must be positive")
        uses_surfels = self.normal_prior.startswith("surfels") or self.distance_prior != "none"
        if self.surfels_per_token > 1 and not uses_surfels:
            raise ValueError("surfels_per_token needs a surfel normal_prior or distance_prior")
        if self.distance_prior not in ("none", "surfels", "union"):
            raise ValueError("distance_prior must be none, surfels or union")
        if self.width <= 0 or self.heads <= 0 or self.width % self.heads:
            raise ValueError("width must be positive and divisible by heads")
        if (self.width // self.heads) % 6:
            raise ValueError("Each attention head needs a multiple of six dimensions for balanced 3D RoPE")
        if (
            min(self.latents, self.decoder_depth, self.ff_multiplier, self.fourier_bands, self.query_chunk)
            < 1
        ):
            raise ValueError("Model sizes must be positive")
        if self.encoder_depth < 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid depth/dropout")
        if not 0 < self.min_wavelength <= self.max_wavelength:
            raise ValueError("Invalid positional wavelength range")


ROTATION_MODES = ("none", "so3", "octahedral")


@dataclass(frozen=True)
class DataConfig:
    encoder_points: int = 4096
    queries: int = 1024
    mask_fraction: float = 0.6
    full_input_probability: float = 0.1
    mask_patches: int = 32
    near_fraction: float = 0.7
    near_sigma: float = 0.03
    normal_band: float = 0.1
    box_padding: float = 0.15
    target_mode: str = "provided"
    # Training-time augmentation for data without a canonical frame or reliable
    # winding (Objaverse). Defaults reproduce the unaugmented ShapeNet-Car setup.
    augment_rotation: str = "none"
    normal_sign_flip_probability: float = 0.0
    # "raise" stops training on the first unreadable record; "skip" logs it once
    # and substitutes the next record, for corpora too large to hand-clean.
    bad_records: str = "raise"
    # "mixed": each training sample draws its patch count from mask_patches x {1/4, 1/2, 1, 2, 4}, at the same hidden
    # fraction, so holes range from a few large ones to many small ones
    mask_patch_mix: str = "fixed"

    def __post_init__(self):
        _finite_config(self)
        if min(self.encoder_points, self.queries, self.mask_patches) < 1:
            raise ValueError("Data sizes must be positive")
        if self.encoder_points < 2:
            raise ValueError("encoder_points must be at least two")
        if (
            not 0 <= self.mask_fraction < 1
            or not 0 <= self.near_fraction <= 1
            or not 0 <= self.full_input_probability <= 1
            or not 0 <= self.normal_sign_flip_probability <= 1
        ):
            raise ValueError("Invalid sampling fractions")
        if min(self.near_sigma, self.normal_band, self.box_padding) <= 0:
            raise ValueError("Geometric sampling scales must be positive")
        if self.target_mode not in ("provided", "surface_samples"):
            raise ValueError("target_mode must be provided or surface_samples")
        if self.augment_rotation not in ROTATION_MODES:
            raise ValueError(f"augment_rotation must be one of {ROTATION_MODES}")
        if self.mask_patch_mix not in ("fixed", "mixed"):
            raise ValueError("mask_patch_mix must be fixed or mixed")
        if self.bad_records not in ("raise", "skip"):
            raise ValueError("bad_records must be raise or skip")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 4
    accumulation: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    # A positive step count overrides warmup_epochs; large corpora train for
    # one or two epochs, where whole-epoch warmup is far too coarse.
    warmup_steps: int = 0
    min_lr_ratio: float = 0.05
    gradient_clip: float = 1.0
    normal_weight: float = 0.1
    # "oriented" is the cosine loss; "unoriented" ignores the sign, for meshes
    # whose winding is inconsistent or whose surfaces are open.
    normal_loss: str = "oriented"
    # "cosine" penalises 1 - cos(angle), which is quadratic in small angles: a
    # 10 degree error costs 0.015. "angle" penalises the angle itself, in
    # radians, so moderate errors keep a gradient; the reported metric is the
    # mean angle, which "angle" optimises directly.
    normal_penalty: str = "cosine"
    huber_delta: float = 0.05
    workers: int = 4
    seed: int = 17
    amp: str = "bf16"

    def __post_init__(self):
        _finite_config(self)
        if min(self.epochs, self.batch_size, self.accumulation) < 1:
            raise ValueError("epochs, batch_size and accumulation must be positive")
        if self.workers < 0 or self.warmup_epochs < 0 or self.warmup_steps < 0:
            raise ValueError("workers, warmup_epochs and warmup_steps must be nonnegative")
        if min(self.learning_rate, self.gradient_clip, self.huber_delta) <= 0:
            raise ValueError("Invalid optimization scales")
        if self.weight_decay < 0 or self.normal_weight < 0 or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("Invalid loss/optimizer weights")
        if self.amp not in ("none", "bf16", "fp16"):
            raise ValueError("amp must be none, bf16, or fp16")
        if self.normal_loss not in ("oriented", "unoriented"):
            raise ValueError("normal_loss must be oriented or unoriented")
        if self.normal_penalty not in ("cosine", "angle"):
            raise ValueError("normal_penalty must be cosine or angle")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        if self.model.normal_head == "projector" and self.train.normal_loss != "unoriented":
            raise ValueError("The projector normal head is sign-free; set train.normal_loss to unoriented")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "ExperimentConfig":
        if not isinstance(value, dict):
            raise ValueError("Configuration must be a JSON object")
        unknown = set(value) - {"model", "data", "train"}
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        return cls(
            ModelConfig(**value.get("model", {})),
            DataConfig(**value.get("data", {})),
            TrainConfig(**value.get("train", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
