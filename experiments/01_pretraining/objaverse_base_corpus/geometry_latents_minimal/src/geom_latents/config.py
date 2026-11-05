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

    def __post_init__(self):
        _finite_config(self)
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


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

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
