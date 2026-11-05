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


SYMMETRY_ARRAYS = ("symmetry_rotation", "symmetry_translation", "symmetry_support", "symmetry_valid", "symmetry_local")


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
    optional = ("surfels", "surfel_normals", "surfel_weights")  # a denser readout set, when exported
    with np.load(path, allow_pickle=False) as packed:
        if not required <= set(packed.files):
            raise ValueError(f"Missing latent arrays: {sorted(required - set(packed.files))}")
        arrays = {name: np.asarray(packed[name], dtype=np.float32) for name in required}
        present = [name for name in optional if name in packed.files]
        if present and len(present) != len(optional):
            raise ValueError(f"Surfel arrays must come together: {optional}")
        surfels = {name: np.asarray(packed[name], dtype=np.float32) for name in present}
        complete = np.asarray(packed["complete"], dtype=np.float32) if "complete" in packed.files else None
        mirror = np.asarray(packed["mirror"], dtype=np.float32) if "mirror" in packed.files else None
        present = [name for name in SYMMETRY_ARRAYS if name in packed.files]
        if present and len(present) != len(SYMMETRY_ARRAYS):
            raise ValueError(f"Symmetry arrays must come together: {SYMMETRY_ARRAYS}")
        symmetry = {name: np.asarray(packed[name], dtype=np.float32) for name in present}
    if complete is not None and (complete.shape != (1,) or complete[0] not in (0.0, 1.0)):
        raise ValueError("complete must be a single 0 or 1")
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
    if surfels:
        count_s = len(surfels["surfels"])
        shapes = {"surfels": (count_s, 3), "surfel_normals": (count_s, 3), "surfel_weights": (count_s,)}
        if count_s < 1 or any(surfels[key].shape != shape for key, shape in shapes.items()):
            raise ValueError("Invalid surfel shapes")
        if not all(np.isfinite(value).all() for value in surfels.values()) or (surfels["surfel_weights"] <= 0).any():
            raise ValueError("Surfels must be finite with positive weights")
    if mirror is not None and (mirror.shape != (5,) or not np.isfinite(mirror).all()):
        raise ValueError("mirror must be 5 finite numbers (normal, offset, fit score)")
    if symmetry:
        maps = len(symmetry["symmetry_rotation"])
        disks = len(surfels["surfels"]) if surfels else count
        shapes = {"symmetry_rotation": (maps, 3, 3), "symmetry_translation": (maps, 3), "symmetry_support": (maps,),
                  "symmetry_valid": (maps,), "symmetry_local": (maps, disks)}
        if maps < 1 or any(symmetry[key].shape != shape for key, shape in shapes.items()):
            raise ValueError("Invalid symmetry shapes (symmetry_local has one flag per map and readout disk)")
        if not all(np.isfinite(value).all() for value in symmetry.values()):
            raise ValueError("Nonfinite symmetry arrays")
    extra = {name: torch.from_numpy(value)[None].to(device) for name, value in {**surfels, **symmetry}.items()}
    tensors = {name: torch.from_numpy(value)[None].to(device) for name, value in arrays.items()}
    return LatentGeometry(
        tokens=tensors["tokens"],
        positions=tensors["normalized_positions"],
        normals=tensors["normals"],
        weights=tensors["weights"],
        center=tensors["center"],
        scale=tensors["scale"],
        area=tensors["area"],
        surfel_positions=extra.get("surfels"),
        surfel_normals=extra.get("surfel_normals"),
        surfel_weights=extra.get("surfel_weights"),
        complete=None if complete is None else torch.from_numpy(complete)[None].to(device),
        mirror=None if mirror is None else torch.from_numpy(mirror)[None].to(device),
        symmetry_rotation=extra.get("symmetry_rotation"),
        symmetry_translation=extra.get("symmetry_translation"),
        symmetry_support=extra.get("symmetry_support"),
        symmetry_valid=extra.get("symmetry_valid"),
        symmetry_local=extra.get("symmetry_local"),
    )
