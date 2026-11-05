"""Configuration: three frozen dataclasses, loaded from JSON with no unknown keys."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


def _check_types(config) -> None:
    for item in fields(config):
        value = getattr(config, item.name)
        if item.type == "int" and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError(f"{item.name} must be an integer")
        if item.type == "float" and (isinstance(value, bool) or not isinstance(value, (int, float))
                                     or not math.isfinite(value)):
            raise ValueError(f"{item.name} must be a finite number")


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
    # surfels kept per token for the disk readout (farthest-point samples that extend the anchors)
    surfels_per_token: int = 8
    # symmetry maps of the visible input offered to the decoder; 0 turns them off
    symmetry_maps: int = 16

    def __post_init__(self):
        _check_types(self)
        if self.width <= 0 or self.heads <= 0 or self.width % self.heads:
            raise ValueError("width must be positive and divisible by heads")
        if (self.width // self.heads) % 6:
            raise ValueError("each head needs a multiple of six dimensions for 3D rotary embeddings")
        if min(self.latents, self.decoder_depth, self.ff_multiplier, self.fourier_bands, self.query_chunk,
               self.surfels_per_token) < 1 or self.encoder_depth < 0 or self.symmetry_maps < 0:
            raise ValueError("model sizes must be positive")
        if not 0 <= self.dropout < 1 or not 0 < self.min_wavelength <= self.max_wavelength:
            raise ValueError("invalid dropout or wavelength range")


ROTATIONS = ("none", "so3", "octahedral")


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
    # "provided": distance targets from prepare-mesh; "surface_samples": from the point cloud itself
    target_mode: str = "provided"
    augment_rotation: str = "none"
    normal_sign_flip_probability: float = 0.0
    # "skip" reports an unreadable record once and substitutes the next one
    bad_records: str = "raise"

    def __post_init__(self):
        _check_types(self)
        if min(self.queries, self.mask_patches) < 1 or self.encoder_points < 2:
            raise ValueError("data sizes must be positive")
        if not (0 <= self.mask_fraction < 1 and 0 <= self.near_fraction <= 1 and 0 <= self.full_input_probability <= 1
                and 0 <= self.normal_sign_flip_probability <= 1):
            raise ValueError("fractions must lie in [0, 1]")
        if min(self.near_sigma, self.normal_band, self.box_padding) <= 0:
            raise ValueError("sampling scales must be positive")
        if self.target_mode not in ("provided", "surface_samples"):
            raise ValueError("target_mode must be provided or surface_samples")
        if self.augment_rotation not in ROTATIONS:
            raise ValueError(f"augment_rotation must be one of {ROTATIONS}")
        if self.bad_records not in ("raise", "skip"):
            raise ValueError("bad_records must be raise or skip")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 4
    accumulation: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 100
    min_lr_ratio: float = 0.05
    gradient_clip: float = 1.0
    normal_weight: float = 1.0
    huber_delta: float = 0.05
    workers: int = 4
    seed: int = 17
    amp: str = "bf16"

    def __post_init__(self):
        _check_types(self)
        if min(self.epochs, self.batch_size, self.accumulation) < 1 or min(self.workers, self.warmup_steps) < 0:
            raise ValueError("epochs, batch_size and accumulation must be positive")
        if min(self.learning_rate, self.gradient_clip, self.huber_delta) <= 0:
            raise ValueError("learning_rate, gradient_clip and huber_delta must be positive")
        if self.weight_decay < 0 or self.normal_weight < 0 or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("invalid loss or optimizer weights")
        if self.amp not in ("none", "bf16", "fp16"):
            raise ValueError("amp must be none, bf16 or fp16")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "ExperimentConfig":
        unknown = set(value) - {"model", "data", "train"}
        if unknown:
            raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
        return cls(ModelConfig(**value.get("model", {})), DataConfig(**value.get("data", {})),
                   TrainConfig(**value.get("train", {})))

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
