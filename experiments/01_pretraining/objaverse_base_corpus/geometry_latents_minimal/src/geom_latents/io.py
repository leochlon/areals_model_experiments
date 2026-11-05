"""Atomic artifacts and tensor-only checkpoint loading."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .model import GeometryAutoencoder, LatentGeometry

CHECKPOINT_VERSION = 1


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_checkpoint(path: str | Path, state: dict) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_model(path: str | Path, device: torch.device | str = "cpu"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported geometry checkpoint version")
    config = ExperimentConfig.from_dict(state["config"])
    model = GeometryAutoencoder(config.model)
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    return model, config, state


def environment() -> dict:
    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        torch=str(torch.__version__),
        numpy=str(np.__version__),
        cuda=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
    )


def load_latents(path: str | Path, device: torch.device | str = "cpu") -> LatentGeometry:
    """Load one exported shape as a batched latent packet for downstream heads.

    Exports omit initializer features; ``initial_tokens`` is therefore None.
    The learned token vectors and their complete spatial frame are preserved.
    """
    required = {
        "tokens",
        "positions",
        "normalized_positions",
        "normals",
        "weights",
        "center",
        "scale",
        "area",
    }
    with np.load(path, allow_pickle=False) as packed:
        if not required <= set(packed.files):
            raise ValueError(f"Missing latent arrays: {sorted(required - set(packed.files))}")
        arrays = {name: np.asarray(packed[name], dtype=np.float32) for name in required}
    tokens = arrays["tokens"]
    if tokens.ndim != 2 or min(tokens.shape) < 1:
        raise ValueError("Latent tokens must be a nonempty [anchors, width] array")
    count = len(tokens)
    expected = {
        "positions": (count, 3),
        "normalized_positions": (count, 3),
        "normals": (count, 3),
        "weights": (count,),
        "center": (3,),
        "scale": (1,),
        "area": (1,),
    }
    if any(arrays[key].shape != shape for key, shape in expected.items()):
        raise ValueError("Invalid latent position/weight/frame shapes")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Nonfinite latent data")
    if any((arrays[key] <= 0).any() for key in ("weights", "scale", "area")):
        raise ValueError("Exported latent weights, scale and area must be positive")
    reconstructed = arrays["normalized_positions"] * arrays["scale"] + arrays["center"]
    if not np.allclose(arrays["positions"], reconstructed, atol=1e-6, rtol=1e-5):
        raise ValueError("Inconsistent retained coordinate frame")
    tensors = {name: torch.from_numpy(value)[None].to(device) for name, value in arrays.items()}
    return LatentGeometry(
        tokens=tensors["tokens"],
        positions=tensors["normalized_positions"],
        normals=tensors["normals"],
        weights=tensors["weights"],
        center=tensors["center"],
        scale=tensors["scale"],
        area=tensors["area"],
    )
