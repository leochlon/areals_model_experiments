"""Manifests, masked training examples and batching."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from .config import DataConfig, ModelConfig
from .geometry import (canonical_cloud, flip_normal_signs, prepare_cloud, random_rotation, rotate_cloud,
                       sampled_surface_queries, visible_mask)
from .symmetry import symmetry_candidates

SCHEMA = "areals"
SPLITS = ("train", "val", "test")
TARGETS = ("queries", "udf", "query_normals")


def stable_seed(seed: int, identity: str, epoch: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{identity}:{epoch}".encode()).digest()[:8], "little")


def read_manifest(path: str | Path) -> tuple[list[dict], dict]:
    """Records with absolute paths, and the manifest's metadata. Families never cross splits."""
    path = Path(path).resolve()
    document = json.loads(path.read_text())
    if document.get("schema") != SCHEMA or not document.get("units"):
        raise ValueError(f'a manifest needs "schema": "{SCHEMA}" and "units"')
    records = document.get("records") or []
    if not records:
        raise ValueError("the manifest has no records")
    ids, family_split, paths = set(), {}, set()
    for record in records:
        if not all(isinstance(record.get(k), str) and record[k] for k in ("id", "family", "split", "path")):
            raise ValueError("every record needs id, family, split and path")
        if record["split"] not in SPLITS or record["id"] in ids:
            raise ValueError(f"{record['id']}: unknown split or repeated id")
        if family_split.setdefault(record["family"], record["split"]) != record["split"]:
            raise ValueError(f"family {record['family']} crosses splits")
        ids.add(record["id"])
        record["path"] = str((path.parent / record["path"]).resolve())
        if record["path"] in paths:
            raise ValueError(f"{record['id']}: file used by two records")
        paths.add(record["path"])
    metadata = dict(units=document["units"], manifest=str(path))
    return records, metadata


def load_cloud_file(path: str | Path):
    """A prepared NPZ: the canonical surface cloud and, when present, the distance targets."""
    with np.load(path, allow_pickle=False) as packed:
        missing = {"points", "normals", "weights"} - set(packed.files)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}")
        cloud = canonical_cloud(packed["points"], packed["normals"], packed["weights"])
        targets = {key: np.asarray(packed[key], dtype=np.float64) for key in TARGETS if key in packed.files}
    if targets:
        if set(targets) != set(TARGETS):
            raise ValueError(f"{path}: targets need queries, udf and query_normals together")
        targets["udf"] = targets["udf"].reshape(-1)
        q, d, n = targets["queries"], targets["udf"], targets["query_normals"]
        if q.ndim != 2 or q.shape[1] != 3 or q.shape != n.shape or len(d) != len(q) or len(d) == 0:
            raise ValueError(f"{path}: target shapes disagree")
        if not all(np.isfinite(a).all() for a in targets.values()) or (d < 0).any():
            raise ValueError(f"{path}: invalid distances")
    return cloud, targets


class GeometryDataset(Dataset):
    """One masked example per record and epoch: the visible cloud prepared for the encoder, its symmetry maps, and
    queries with distance and normal targets from the whole surface. Every random draw is seeded by record and
    epoch."""

    def __init__(self, records: list[dict], model: ModelConfig, data: DataConfig, seed: int,
                 mask_fraction: float | None = None, supervision: bool = True, fixed_queries: bool = False,
                 training: bool = False):
        if not records:
            raise ValueError("the split is empty")
        self.records, self.model, self.data, self.seed = records, model, data, seed
        self.mask_fraction = data.mask_fraction if mask_fraction is None else mask_fraction
        self.supervision, self.fixed_queries, self.training = supervision, fixed_queries, training
        self.epoch = 0
        self.failed: set[str] = set()

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict:
        if self.data.bad_records == "raise":
            return self._sample(self.records[index])
        for step in range(len(self.records)):
            record = self.records[(index + step) % len(self.records)]
            try:
                return self._sample(record)
            except Exception as error:  # noqa: BLE001 - any failure makes the record unusable
                if record["id"] not in self.failed:
                    self.failed.add(record["id"])
                    print(f"areals: skipping {record['id']} ({type(error).__name__}: {error})", file=sys.stderr,
                          flush=True)
        raise RuntimeError("every record in this split failed to load")

    def _rng(self, record: dict, stream: str):
        return np.random.default_rng(stable_seed(self.seed, f"{record['id']}:{stream}", self.epoch))

    def _augment(self, record: dict, cloud, targets: dict):
        if self.data.augment_rotation == "none" and self.data.normal_sign_flip_probability <= 0:
            return cloud, targets
        rng = self._rng(record, "augment")
        if self.data.augment_rotation != "none":
            rotation = random_rotation(self.data.augment_rotation, rng)
            cloud = rotate_cloud(cloud, rotation)
            if targets:
                targets = dict(queries=targets["queries"] @ rotation.T, udf=targets["udf"],
                               query_normals=targets["query_normals"] @ rotation.T)
        return flip_normal_signs(cloud, self.data.normal_sign_flip_probability, rng), targets

    def _sample(self, record: dict) -> dict:
        mask_rng, point_rng, query_rng = (self._rng(record, s) for s in ("mask", "points", "queries"))
        cloud, targets = load_cloud_file(record["path"])
        fraction = self.mask_fraction
        if self.training:
            cloud, targets = self._augment(record, cloud, targets)
            if self._rng(record, "corruption").random() < self.data.full_input_probability:
                fraction = 0.0
        visible = visible_mask(cloud, fraction, self.data.mask_patches, mask_rng)
        per = self.model.surfels_per_token
        prepared = prepare_cloud(cloud.subset(visible), self.model.latents, self.data.encoder_points, point_rng,
                                 self.model.latents * per if per > 1 else 0)
        sample = dict(prepared, id=record["id"], family=record["family"], split=record["split"],
                      complete=np.array([float(visible.all())], dtype=np.float32))
        if self.model.symmetry_maps:
            disks = ("surfels", "surfel_normals") if "surfels" in prepared else ("anchors", "anchor_normals")
            maps = symmetry_candidates(prepared["points"], prepared["normals"], prepared[disks[0]],
                                       prepared[disks[1]], self.model.symmetry_maps, self._rng(record, "symmetry"))
            sample.update(zip(("symmetry_rotation", "symmetry_translation", "symmetry_support", "symmetry_valid",
                               "symmetry_local"), maps, strict=True))
        if not self.supervision:
            return sample
        if self.data.target_mode == "provided":
            if not targets:
                raise ValueError(f"{record['id']}: no distance targets; run prepare-mesh or use surface_samples")
            indices = query_rng.choice(len(targets["udf"]), self.data.queries,
                                       replace=len(targets["udf"]) < self.data.queries)
            queries, udf, normals = (targets[k][indices] for k in TARGETS)
        else:
            queries, udf, normals = sampled_surface_queries(
                cloud, np.zeros(len(visible), dtype=bool) if self.fixed_queries else ~visible, self.data.queries,
                self.data.near_fraction, self.data.near_sigma, self.data.box_padding, query_rng)
        length = np.linalg.norm(normals, axis=-1)
        sample.update(
            queries=queries.astype(np.float32),
            udf=udf.astype(np.float32),
            query_normals=(normals / np.maximum(length[:, None], 1e-8)).astype(np.float32),
            normal_valid=(length > 1e-8) & (udf / float(prepared["scale"][0]) <= self.data.normal_band),
            query_hidden=~visible[cKDTree(cloud.points).query(queries, workers=1)[1]],
        )
        return sample


def _pad_stack(values: list[np.ndarray], axis: int = 0) -> torch.Tensor:
    width = max(value.shape[axis] for value in values)
    padded = []
    for value in values:
        padding = [(0, 0)] * value.ndim
        padding[axis] = (0, width - value.shape[axis])
        padded.append(np.pad(value, padding))
    return torch.from_numpy(np.stack(padded))


def collate_clouds(samples: list[dict]) -> dict:
    """Stack samples, padding point, anchor and surfel sets with zero weights."""
    batch = {key: [sample[key] for sample in samples] for key in ("id", "family", "split")}
    for key, value in samples[0].items():
        if key in batch or not all(key in sample for sample in samples):
            continue
        values = [sample[key] for sample in samples]
        if key == "symmetry_local":
            batch[key] = _pad_stack(values, axis=1)
        elif key in ("points", "normals", "point_weights", "anchors", "anchor_normals", "anchor_weights", "surfels",
                     "surfel_normals", "surfel_weights"):
            batch[key] = _pad_stack(values)
        elif isinstance(value, np.ndarray):
            batch[key] = torch.from_numpy(np.stack(values))
    return batch


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}
