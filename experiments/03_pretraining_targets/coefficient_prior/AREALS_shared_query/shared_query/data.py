"""One frozen object cohort and native AREALS inputs for every objective arm.

The prepared HDF5 cache contributes *supervision only*. Patch-local input points,
fitted coefficients and Hessian labels never enter the native geometry encoder.
Patch charts are queried in their original world coordinates. This initial
controlled recipe deliberately disables rotation in every arm: the native
dataset does not expose its augmentation transform to auxiliary labels.
"""
from __future__ import annotations

from collections import OrderedDict
import gzip
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from areal_tokens.data import GeometryDataset, collate_clouds


MODES = ("baseline", "hessian", "operator")


def read_json(path):
    """Read JSON or the archive's losslessly compressed JSON representation."""
    path = Path(path).expanduser().resolve()
    if not path.exists() and path.suffix != ".gz" and Path(str(path) + ".gz").exists():
        path = Path(str(path) + ".gz")
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle), path
    return json.loads(path.read_text()), path


def parse_path_maps(path_maps=None):
    """Accept a mapping, (old,new) pairs, or repeatable OLD=NEW CLI strings."""
    if path_maps is None:
        return []
    items = path_maps.items() if isinstance(path_maps, dict) else path_maps
    result = []
    for item in items:
        if isinstance(item, str):
            if "=" not in item:
                raise ValueError("Path remapping must have the form OLD=NEW")
            old, new = item.split("=", 1)
        else:
            old, new = item
        old, new = str(old).rstrip("/"), str(new).rstrip("/")
        if not old or not new:
            raise ValueError("Path remapping prefixes cannot be empty")
        result.append((old, new))
    # Longest-prefix matching prevents a broad mapping shadowing a precise one.
    return sorted(result, key=lambda pair: len(pair[0]), reverse=True)


def resolve_path(value, parent, maps):
    value = str(value)
    for old, new in maps:
        if value == old or value.startswith(old + "/"):
            value = new + value[len(old):]
            break
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else Path(parent) / path).resolve())


def _positive(value, label):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return value


def _center(metadata, label):
    value = np.asarray(metadata.get("center"), dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"{label} requires a finite three-vector center")
    return value


def projected_queries(triangles, count, rng):
    """Area-uniform projected chart samples, restricted to the unit disk.

    The loss subsequently multiplies by the reference graph's area factor.
    This reproduces the archived chart sampling measure without importing its
    training script or any mesh-processing dependency.
    """
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("Patch triangles must have shape [faces,3,3]")
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    mass = np.abs(cross[:, 2])
    if not np.isfinite(triangles).all() or not np.isfinite(mass).all() or mass.sum() <= 1e-14:
        raise ValueError("Empty or nonfinite projected chart")
    pieces, total = [], 0
    for _ in range(200):
        size = max(64, 2 * (count - total))
        ids = rng.choice(len(triangles), size, p=mass / mass.sum())
        bary = rng.random((size, 2))
        reflected = bary.sum(1) > 1
        bary[reflected] = 1 - bary[reflected]
        points = (triangles[ids, 0] + bary[:, :1] * (triangles[ids, 1] - triangles[ids, 0])
                  + bary[:, 1:] * (triangles[ids, 2] - triangles[ids, 0]))
        kept = points[np.sum(points[:, :2] ** 2, axis=1) < 1, :2]
        pieces.append(kept)
        total += len(kept)
        if total >= count:
            return np.concatenate(pieces)[:count]
    raise ValueError("Insufficient chart support in unit disk")


class JointDataset(Dataset):
    """Deterministic global-example indexing for single GPU or distributed runs.

    `index_path` must be the frozen successful intersection, not an unfinished
    preparation status file. Every arm uses the same IDs and native point/query
    streams. Auxiliary draws use independent RNG streams and cannot change the
    native example. `length` counts object presentations, not optimizer steps.
    Evaluation walks sorted IDs, retaining fixed samples for each object.
    """
    def __init__(self, objects_path, index_path, config, split, seed, length, mode,
                 patches=8, patch_queries=64, hessian_queries=256,
                 evaluation=False, path_maps=None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if split not in ("train", "val", "test"):
            raise ValueError("Invalid split")
        for key, value in (("length", length), ("patches", patches),
                           ("patch_queries", patch_queries), ("hessian_queries", hessian_queries)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if config.data.augment_rotation != "none":
            raise ValueError("Shared-query recipe requires augment_rotation='none' for aligned auxiliary coordinates")
        if config.data.mask_fraction != 0 or config.data.full_input_probability != 1:
            raise ValueError("Shared-query recipe requires complete geometry in every arm")
        if config.data.target_mode != "provided":
            raise ValueError("Native d/P labels must be supplied mesh-query targets")
        if config.data.bad_records != "raise":
            raise ValueError("bad_records must be 'raise'; substituting objects breaks cohort matching")
        self.mode, self.split, self.seed, self.length = mode, split, int(seed), length
        self.patches, self.patch_queries, self.hessian_queries = patches, patch_queries, hessian_queries
        self.evaluation = bool(evaluation)
        self.maps = parse_path_maps(path_maps)
        objects, self.objects_path = read_json(objects_path)
        index, self.index_path = read_json(index_path)
        objects = objects.get("records", objects.get("objects")) if isinstance(objects, dict) else objects
        if not isinstance(objects, list) or not isinstance(index.get("objects"), list):
            raise ValueError("Expected objects list and prepared index with an objects list")
        by_id = self._validated_rows(objects, "native manifest")
        prepared = self._validated_rows(index["objects"], "prepared index")
        if not prepared:
            raise ValueError("Prepared intersection is empty")
        if index.get("complete") is False:
            raise ValueError("Prepared intersection is not marked complete")
        all_rows = []
        scales = None
        for identity, row in sorted(prepared.items()):
            if identity not in by_id:
                raise ValueError(f"Prepared object absent from native manifest: {identity}")
            native = by_id[identity]
            for key in ("family", "split"):
                if native[key] != row[key]:
                    raise ValueError(f"{identity}: native/prepared {key} differs")
            for branch in ("patch", "query"):
                entry = row.get(branch, {})
                if entry.get("status") != "ok" or not entry.get("group"):
                    raise ValueError(f"{identity}: index must be a frozen patch/query success intersection")
                if not isinstance(entry.get("metadata"), dict):
                    raise ValueError(f"{identity}: missing {branch} coordinate metadata")
            pm, hm = row["patch"]["metadata"], row["query"]["metadata"]
            _center(pm, identity + " patch")
            _positive(pm.get("scale"), identity + " patch scale")
            _center(hm, identity + " Hessian")
            _positive(hm.get("rms_radius"), identity + " Hessian scale")
            local_scales = tuple(float(x) for x in hm.get("scales", ()))
            if not local_scales or not all(np.isfinite(x) and x > 0 for x in local_scales):
                raise ValueError(f"{identity}: missing positive Hessian smoothing scales")
            if scales is not None and local_scales != scales:
                raise ValueError("Hessian smoothing scales differ between objects")
            scales = local_scales
            record = dict(native)
            record["path"] = resolve_path(native["path"], self.objects_path.parent, self.maps)
            record["_prepared"] = dict(row, pack=resolve_path(row["pack"], self.index_path.parent, self.maps))
            all_rows.append(record)
        self.hessian_scales = scales
        self.rows = [row for row in all_rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"Empty {split} split in frozen intersection")
        # Missing files fail before expensive model construction/training.
        for row in self.rows:
            if not Path(row["path"]).is_file():
                raise FileNotFoundError(row["path"])
            if mode != "baseline" and not Path(row["_prepared"]["pack"]).is_file():
                raise FileNotFoundError(row["_prepared"]["pack"])
        self.native = GeometryDataset(self.rows, config.model, config.data, self.seed,
                                      mask_fraction=0., supervision=True,
                                      fixed_queries=self.evaluation, training=not self.evaluation)
        self._orders = OrderedDict()
        self._handles = OrderedDict()
        self._verified_groups = set()
        if mode != "baseline":
            self._validate_schemas()
        self.provenance = dict(
            cohort_counts={s: sum(r["split"] == s for r in all_rows) for s in ("train", "val", "test")},
            native_manifest_objects=len(objects), intersection_objects=len(all_rows),
            split=split, mode=mode, seed=self.seed, complete_geometry=True, rotation="none",
            hessian_scales_in_source_rms_units=list(scales), path_maps=self.maps,
            patch_sampling="reference area-mass patches; projected triangle-area queries",
        )

    @staticmethod
    def _validated_rows(rows, label):
        output, families, paths = {}, {}, {}
        for row in rows:
            if not isinstance(row, dict) or not all(isinstance(row.get(k), str) and row[k]
                                                   for k in ("id", "family", "split")):
                raise ValueError(f"{label}: each row needs id/family/split strings")
            if row["split"] not in ("train", "val", "test") or row["id"] in output:
                raise ValueError(f"{label}: duplicate ID or invalid split")
            if families.setdefault(row["family"], row["split"]) != row["split"]:
                raise ValueError(f"{label}: family crosses splits")
            if row.get("path"):
                key = row["path"]
                if key in paths:
                    raise ValueError(f"{label}: same native cache reused by multiple identities")
                paths[key] = row["id"]
            output[row["id"]] = row
        return output

    def __len__(self):
        return self.length

    def _validate_schemas(self):
        """Fail on absent targets before optimization; do not load array payloads."""
        branch = "query" if self.mode == "hessian" else "patch"
        packed = {}
        for row in self.rows:
            packed.setdefault(row["_prepared"]["pack"], []).append(row)
        for path, rows in packed.items():
            with h5py.File(path, "r") as handle:
                for row in rows:
                    expected = row["_prepared"][branch]
                    if expected["group"] not in handle:
                        raise ValueError(f"{row['id']}: prepared {branch} group is missing")
                    group = handle[expected["group"]]
                    if "metadata" not in group.attrs:
                        raise ValueError(f"{row['id']}: prepared group metadata is missing")
                    if json.loads(group.attrs["metadata"]) != expected["metadata"]:
                        raise ValueError(f"{row['id']}: group metadata differs from index")
                    if branch == "query":
                        if "x" not in group or "H" not in group:
                            raise ValueError(f"{row['id']}: Hessian queries/targets are missing")
                        n = len(group["x"])
                        if n < 1 or group["x"].shape != (n, 3) or group["H"].shape != (n, len(self.hessian_scales), 6):
                            raise ValueError(f"{row['id']}: Hessian target shapes are incompatible")
                    else:
                        required = ("origin", "frame", "radius", "coeff", "offset", "area_mass", "triangles", "triangle_offsets")
                        if any(key not in group for key in required):
                            raise ValueError(f"{row['id']}: patch supervision arrays are missing")
                        n = len(group["origin"])
                        shapes = dict(origin=(n, 3), frame=(n, 3, 3), radius=(n,), coeff=(n, 10),
                                      offset=(n,), area_mass=(n,), triangle_offsets=(n + 1,))
                        if n < 1 or any(group[key].shape != shape for key, shape in shapes.items()):
                            raise ValueError(f"{row['id']}: degree-three patch shapes are incompatible")
                        if group["triangles"].ndim != 3 or group["triangles"].shape[1:] != (3, 3):
                            raise ValueError(f"{row['id']}: patch triangle shape is incompatible")

    def _row_index(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)
        epoch, offset = divmod(index, len(self.rows))
        if self.evaluation:
            return offset, 0
        if epoch not in self._orders:
            self._orders[epoch] = np.random.default_rng([self.seed, epoch]).permutation(len(self.rows))
            if len(self._orders) > 4:
                self._orders.popitem(last=False)
        return int(self._orders[epoch][offset]), epoch

    def _group(self, row, branch):
        entry = row["_prepared"]
        path, name = entry["pack"], entry[branch]["group"]
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
            if len(self._handles) > 8:
                self._handles.popitem(last=False)[1].close()
        self._handles.move_to_end(path)
        group = self._handles[path][name]
        key = (path, name)
        if key not in self._verified_groups:
            if "metadata" not in group.attrs:
                raise ValueError(f"{row['id']}: cache group missing coordinate metadata")
            text = group.attrs["metadata"]
            if isinstance(text, bytes):
                text = text.decode()
            if json.loads(text) != entry[branch]["metadata"]:
                raise ValueError(f"{row['id']}: packed metadata differs from frozen index")
            self._verified_groups.add(key)
        return group

    def __getitem__(self, index):
        index = int(index)
        row_index, epoch = self._row_index(index)
        row = self.rows[row_index]
        self.native.set_epoch(epoch)
        sample = self.native[row_index]
        # Evaluation repeat indices keep exactly the same samples per object.
        auxiliary_index = row_index if self.evaluation else index
        rng = np.random.default_rng([self.seed, auxiliary_index, 127983])
        if self.mode == "baseline":
            return sample, {}
        if self.mode == "hessian":
            group = self._group(row, "query")
            n = len(group["x"])
            if n < 1 or group["H"].shape != (n, len(self.hessian_scales), 6):
                raise ValueError(f"{row['id']}: malformed Hessian arrays")
            ids = rng.choice(n, self.hessian_queries, replace=n < self.hessian_queries)
            # h5py requires sorted, unique fancy indices. Two batched reads,
            # then restore the sampled order and any repeated query identities.
            unique, inverse = np.unique(ids, return_inverse=True)
            x = group["x"][unique][inverse].astype(np.float64)
            target = group["H"][unique][inverse].astype(np.float32)
            meta = row["_prepared"]["query"]["metadata"]
            x = x * float(meta["rms_radius"]) + np.asarray(meta["center"])
            if not np.isfinite(x).all() or not np.isfinite(target).all():
                raise ValueError(f"{row['id']}: nonfinite Hessian supervision")
            return sample, dict(h_x=x.astype(np.float32), h_target=target)
        group = self._group(row, "patch")
        mass = group["area_mass"][:].astype(np.float64)
        if not len(mass) or not np.isfinite(mass).all() or (mass < 0).any() or mass.sum() <= 0:
            raise ValueError(f"{row['id']}: invalid patch area masses")
        ids = rng.choice(len(mass), self.patches, replace=True, p=mass / mass.sum())
        arrays = {key: np.stack([group[key][int(i)] for i in ids]).astype(np.float64)
                  for key in ("origin", "frame", "radius", "coeff", "offset")}
        if arrays["coeff"].shape != (self.patches, 10):
            raise ValueError("Operator branch requires the existing degree-three patch cache")
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError(f"{row['id']}: nonfinite patch supervision")
        if np.any(arrays["radius"] <= 0) or np.any(arrays["offset"] <= 0):
            raise ValueError(f"{row['id']}: invalid chart radius/offset")
        frames = arrays["frame"]
        if (not np.allclose(frames.transpose(0, 2, 1) @ frames, np.eye(3), atol=1e-5)
                or np.any(np.linalg.det(frames) <= 0)):
            raise ValueError(f"{row['id']}: nonorthonormal chart frames")
        uv = []
        for i in ids:
            start, end = map(int, group["triangle_offsets"][int(i):int(i) + 2])
            uv.append(projected_queries(group["triangles"][start:end], self.patch_queries, rng))
        tau = rng.uniform(.1, 1., (self.patches, self.patch_queries)) * arrays["offset"][:, None]
        tau[:, ::3] = 0.
        tau[:, 1::3] *= -1
        meta = row["_prepared"]["patch"]["metadata"]
        scale, center = float(meta["scale"]), np.asarray(meta["center"])
        # No fitted coefficient or hidden local geometry enters sample/native.
        auxiliary = dict(
            patch_origin=(arrays["origin"] * scale + center).astype(np.float32),
            patch_frame=frames.astype(np.float32),
            patch_radius=(arrays["radius"] * scale).astype(np.float32),
            patch_coeff=arrays["coeff"], patch_uv=np.stack(uv), patch_tau=tau,
        )
        return sample, auxiliary

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        return state

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


def collate_joint(rows):
    if not rows:
        raise ValueError("Cannot collate an empty batch")
    samples, auxiliary = zip(*rows)
    keys = set(auxiliary[0])
    if any(set(item) != keys for item in auxiliary):
        raise ValueError("Mixed supervision modes in one batch")
    return collate_clouds(list(samples)), {
        key: torch.from_numpy(np.stack([item[key] for item in auxiliary])) for key in sorted(keys)
    }
