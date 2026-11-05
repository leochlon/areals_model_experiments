"""Point-cloud manifests, leakage checks, masked examples, and native exports."""

from __future__ import annotations

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from .config import DataConfig, ModelConfig
from .geometry import (
    canonical_cloud,
    flip_normal_signs,
    mirror_plane,
    prepare_cloud,
    random_rotation,
    rotate_cloud,
    sampled_surface_queries,
    visible_mask,
)
from .symmetry import symmetry_candidates

# The manifest schema keeps its original identifier so manifests, prepared datasets
# and exports written before the rename to Areal stay valid.
SCHEMA = "geometry_latents_v1"
SPLITS = ("train", "val", "test")


def stable_seed(seed: int, identity: str, epoch: int) -> int:
    value = f"{seed}:{identity}:{epoch}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def read_manifest(path: str | Path) -> tuple[list[dict], dict]:
    path = Path(path).resolve()
    document = json.loads(path.read_text())
    if (
        document.get("schema") != SCHEMA
        or not isinstance(document.get("units"), str)
        or not document["units"]
    ):
        raise ValueError(f"Manifest requires schema={SCHEMA} and explicit coordinate units")
    records = document.get("records", [])
    if not records:
        raise ValueError("Empty geometry manifest")
    identities, family_splits, paths = set(), {}, {}
    for record in records:
        if not all(isinstance(record.get(k), str) and record[k] for k in ("id", "family", "split", "path")):
            raise ValueError("Every record needs nonempty id, family, split and path strings")
        if record["split"] not in SPLITS or record["id"] in identities:
            raise ValueError("Invalid split or repeated geometry id")
        identities.add(record["id"])
        family = record["family"]
        if family in family_splits and family_splits[family] != record["split"]:
            raise ValueError(f"Family {family} crosses splits")
        family_splits[family] = record["split"]
        record["path"] = str((path.parent / record["path"]).resolve())
        if record["path"] in paths:
            raise ValueError(f"Same source file reused by {record['id']} and {paths[record['path']]}")
        paths[record["path"]] = record["id"]
    metadata = dict(units=document["units"], manifest=str(path))
    return records, metadata


def load_cloud_file(path: str):
    path = str(Path(path).resolve())
    stat = Path(path).stat()
    return _load_cloud_file(path, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)


@lru_cache(maxsize=4)
def _load_cloud_file(path: str, modified_ns: int, changed_ns: int, size: int):
    # File identity participates in the cache key so an in-process rewrite
    # cannot silently reuse previously loaded arrays.
    with np.load(path, allow_pickle=False) as packed:
        missing = {"points", "normals", "weights"} - set(packed.files)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}; normals/weights are never silently dropped")
        cloud = canonical_cloud(packed["points"], packed["normals"], packed["weights"])
        targets = {
            key: np.asarray(packed[key], dtype=np.float64)
            for key in ("queries", "udf", "query_normals")
            if key in packed.files
        }
    if targets:
        if set(targets) != {"queries", "udf", "query_normals"}:
            raise ValueError(f"{path}: supplied targets require queries, udf and query_normals together")
        targets["udf"] = targets["udf"].reshape(-1)
        q, d, n = targets["queries"], targets["udf"], targets["query_normals"]
        if q.ndim != 2 or q.shape[1] != 3 or q.shape != n.shape or len(d) != len(q) or len(d) == 0:
            raise ValueError(f"{path}: invalid supplied-query shapes")
        if not all(np.isfinite(a).all() for a in targets.values()) or (d < 0).any():
            raise ValueError(f"{path}: invalid supplied unsigned distances")
    return cloud, targets


class GeometryDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        model: ModelConfig,
        data: DataConfig,
        seed: int,
        mask_fraction: float | None = None,
        supervision: bool = True,
        fixed_queries: bool = False,
        training: bool = False,
    ):
        if not records:
            raise ValueError("Dataset split is empty")
        self.records, self.model, self.data, self.seed = records, model, data, seed
        self.mask_fraction = data.mask_fraction if mask_fraction is None else mask_fraction
        if not 0 <= self.mask_fraction < 1:
            raise ValueError("mask_fraction must be in [0,1)")
        self.supervision, self.epoch = supervision, 0
        self.fixed_queries = fixed_queries
        self.training = training
        self.failed_ids: set[str] = set()

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict:
        if self.data.bad_records != "skip":
            return self._sample(self.records[index])
        # Tolerant mode: an unreadable record is reported once per process and
        # replaced by the next readable one, so one bad file cannot end a run.
        for step in range(len(self.records)):
            record = self.records[(index + step) % len(self.records)]
            try:
                return self._sample(record)
            except Exception as error:  # noqa: BLE001 - any loader failure is a bad record
                if record["id"] not in self.failed_ids:
                    self.failed_ids.add(record["id"])
                    print(
                        f"areal: skipping {record['id']} ({type(error).__name__}: {error})",
                        file=sys.stderr,
                        flush=True,
                    )
        raise RuntimeError("Every record in this split failed to load")

    def _augment(self, record: dict, cloud, targets: dict):
        """Training-only corruptions drawn from their own stream, so the mask,
        point and query streams are unchanged when augmentation is off."""
        if self.data.augment_rotation == "none" and self.data.normal_sign_flip_probability <= 0:
            return cloud, targets
        rng = np.random.default_rng(stable_seed(self.seed, record["id"] + ":augment", self.epoch))
        if self.data.augment_rotation != "none":
            rotation = random_rotation(self.data.augment_rotation, rng)
            cloud = rotate_cloud(cloud, rotation)
            if targets:
                targets = dict(
                    queries=targets["queries"] @ rotation.T,
                    udf=targets["udf"],
                    query_normals=targets["query_normals"] @ rotation.T,
                )
        cloud = flip_normal_signs(cloud, self.data.normal_sign_flip_probability, rng)
        return cloud, targets

    def _sample(self, record: dict) -> dict:
        mask_rng = np.random.default_rng(stable_seed(self.seed, record["id"] + ":mask", self.epoch))
        point_rng = np.random.default_rng(stable_seed(self.seed, record["id"] + ":points", self.epoch))
        query_rng = np.random.default_rng(stable_seed(self.seed, record["id"] + ":queries", self.epoch))
        cloud, targets = load_cloud_file(record["path"])
        if self.training:
            cloud, targets = self._augment(record, cloud, targets)
        fraction = self.mask_fraction
        if self.training:
            corruption_rng = np.random.default_rng(
                stable_seed(self.seed, record["id"] + ":corruption", self.epoch)
            )
            if corruption_rng.random() < self.data.full_input_probability:
                fraction = 0.0
        patches = self.data.mask_patches
        if self.data.mask_patch_mix == "mixed" and self.training:
            patches = max(2, int(round(patches * float(mask_rng.choice([0.25, 0.5, 1.0, 2.0, 4.0])))))
        visible = visible_mask(cloud, fraction, patches, mask_rng)
        surfel_count = self.model.latents * self.model.surfels_per_token if self.model.surfels_per_token > 1 else 0
        prepared = prepare_cloud(
            cloud.subset(visible), self.model.latents, self.data.encoder_points, point_rng, surfel_count
        )
        sample = dict(prepared, id=record["id"], family=record["family"], split=record["split"])
        sample["complete"] = np.array([float(visible.all())], dtype=np.float32)
        if self.model.mirror_prior != "none":
            sample["mirror"] = mirror_plane(prepared["points"])
        if self.model.symmetry_prior != "none":
            symmetry_rng = np.random.default_rng(stable_seed(self.seed, record["id"] + ":symmetry", self.epoch))
            surfels = ("surfels", "surfel_normals") if "surfels" in prepared else ("anchors", "anchor_normals")
            rotation, translation, support, valid, local = symmetry_candidates(
                prepared["points"], prepared["normals"], prepared[surfels[0]], prepared[surfels[1]],
                self.model.symmetry_maps, symmetry_rng,
            )
            sample.update(symmetry_rotation=rotation, symmetry_translation=translation, symmetry_support=support,
                          symmetry_valid=valid, symmetry_local=local)
        if self.supervision:
            if self.data.target_mode == "provided":
                if not targets:
                    raise ValueError(
                        f"{record['id']}: no supplied distance targets; use mesh preparation or "
                        "explicitly select target_mode=surface_samples"
                    )
                indices = query_rng.choice(
                    len(targets["udf"]), self.data.queries, replace=len(targets["udf"]) < self.data.queries
                )
                queries, udf, normals = (targets[k][indices] for k in ("queries", "udf", "query_normals"))
            else:
                queries, udf, normals = sampled_surface_queries(
                    cloud,
                    np.zeros(len(visible), dtype=bool) if self.fixed_queries else ~visible,
                    self.data.queries,
                    self.data.near_fraction,
                    self.data.near_sigma,
                    self.data.box_padding,
                    query_rng,
                )
            length = np.linalg.norm(normals, axis=-1)
            valid = (length > 1e-8) & (udf / float(prepared["scale"][0]) <= self.data.normal_band)
            normals = normals / np.maximum(length[:, None], 1e-8)
            sample.update(
                queries=queries.astype(np.float32),
                udf=udf.astype(np.float32),
                query_normals=normals.astype(np.float32),
                normal_valid=valid,
                query_hidden=~visible[cKDTree(cloud.points).query(queries, workers=1)[1]],
            )
        return sample


def collate_clouds(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("Empty batch")
    batch = {key: [sample[key] for sample in samples] for key in ("id", "family", "split")}
    point_keys = ("points", "normals", "point_weights")
    anchor_keys = ("anchors", "anchor_normals", "anchor_weights")
    groups = [point_keys, anchor_keys]
    if "surfels" in samples[0]:
        groups.append(("surfels", "surfel_normals", "surfel_weights"))
    for keys in groups:
        maximum = max(len(sample[keys[0]]) for sample in samples)
        for key in keys:
            values = []
            for sample in samples:
                value = sample[key]
                padding = [(0, maximum - len(value))] + [(0, 0)] * (value.ndim - 1)
                values.append(np.pad(value, padding))
            batch[key] = torch.from_numpy(np.stack(values))
    for key in ("center", "scale", "area"):
        batch[key] = torch.from_numpy(np.stack([sample[key] for sample in samples]))
    if all("mirror" in sample for sample in samples):
        batch["mirror"] = torch.from_numpy(np.stack([sample["mirror"] for sample in samples]))
    if all("symmetry_rotation" in sample for sample in samples):
        for key in ("symmetry_rotation", "symmetry_translation", "symmetry_support", "symmetry_valid"):
            batch[key] = torch.from_numpy(np.stack([sample[key] for sample in samples]))
        width = max(sample["symmetry_local"].shape[1] for sample in samples)  # per surfel: pad like the surfels
        batch["symmetry_local"] = torch.from_numpy(np.stack(
            [np.pad(sample["symmetry_local"], [(0, 0), (0, width - sample["symmetry_local"].shape[1])])
             for sample in samples]))
    if all("complete" in sample for sample in samples):
        batch["complete"] = torch.from_numpy(np.stack([sample["complete"] for sample in samples]))
    if "queries" in samples[0]:
        for key in ("queries", "udf", "query_normals", "normal_valid", "query_hidden"):
            batch[key] = torch.from_numpy(np.stack([sample[key] for sample in samples]))
    return batch


def to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def export_nanofield_geometry(domain, row: int, output: str | Path, *, batch_index: int = 0) -> None:
    """Lossless native geometry export; no target fields or normalization changes.

    Domain.point_features must be exactly the three normal components. If the
    native feature vector contains more fields, explicitly select its normal
    components in the original loader before exporting.
    """
    if domain.point_features is None:
        raise ValueError("Native domain has no normals")
    points = domain.points[row][batch_index].detach().cpu().numpy()
    normals = domain.point_features[row][batch_index].detach().cpu().numpy()
    weights = domain.weights[row][batch_index].detach().cpu().numpy().reshape(-1)
    canonical_cloud(points, normals, weights)  # validate without rewriting the original arrays
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    from .prepare import save_npz

    save_npz(output, points=points, normals=normals, weights=weights)
