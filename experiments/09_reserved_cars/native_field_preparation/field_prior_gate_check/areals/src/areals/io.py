"""Files: atomic JSON, NPZ and checkpoint writes; checkpoint and packet loading."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .model import GeometryAutoencoder, LatentGeometry

CHECKPOINT_FORMAT = "areals"


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_npz(path: str | Path, **arrays) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def save_checkpoint(path: str | Path, state: dict) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(state, format=CHECKPOINT_FORMAT), temporary)
    os.replace(temporary, path)


def load_model(path: str | Path, device: torch.device | str = "cpu"):
    """The model (in eval mode), its configuration and the full checkpoint state."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not an areals checkpoint")
    config = ExperimentConfig.from_dict(state["config"])
    model = GeometryAutoencoder(config.model)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), config, state


def environment() -> dict:
    return dict(python=platform.python_version(), platform=platform.platform(), torch=str(torch.__version__),
                numpy=str(np.__version__), cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())


REQUIRED = ("tokens", "positions", "normals", "weights", "center", "scale", "area")
OPTIONAL = ("surfels", "surfel_normals", "surfel_weights", "complete", "symmetry_rotation", "symmetry_translation",
            "symmetry_support", "symmetry_valid", "symmetry_local")


def load_latents(path: str | Path, device: torch.device | str = "cpu") -> LatentGeometry:
    """One exported packet as a batch of one."""
    with np.load(path, allow_pickle=False) as packed:
        missing = set(REQUIRED) - set(packed.files)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}")
        arrays = {key: np.asarray(packed[key], dtype=np.float32) for key in REQUIRED + OPTIONAL if key in packed.files}
    tokens = arrays["tokens"]
    if tokens.ndim != 2 or arrays["positions"].shape != (len(tokens), 3) or arrays["weights"].shape != (len(tokens),):
        raise ValueError(f"{path}: tokens, positions and weights disagree")
    if "symmetry_local" in arrays:
        disks = len(arrays["surfels"]) if "surfels" in arrays else len(tokens)
        if arrays["symmetry_local"].shape != (len(arrays["symmetry_rotation"]), disks):
            raise ValueError(f"{path}: symmetry_local needs one flag per map and disk")
    t = {key: torch.from_numpy(value)[None].to(device) for key, value in arrays.items()}
    return LatentGeometry(
        t["tokens"], t["positions"], t["normals"], t["weights"], t["center"], t["scale"], t["area"],
        surfel_positions=t.get("surfels"), surfel_normals=t.get("surfel_normals"),
        surfel_weights=t.get("surfel_weights"), complete=t.get("complete"),
        symmetry_rotation=t.get("symmetry_rotation"), symmetry_translation=t.get("symmetry_translation"),
        symmetry_support=t.get("symmetry_support"), symmetry_valid=t.get("symmetry_valid"),
        symmetry_local=t.get("symmetry_local"))
