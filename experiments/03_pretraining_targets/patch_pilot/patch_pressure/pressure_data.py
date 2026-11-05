"""Explicit CFD/geometry association, split auditing and train-only scaling."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree

from train_mesh_geometry import sha256, VERSION as GEOMETRY_VERSION


def resolve(base, path):
    return str((Path(base) / path).resolve())


def load_geometry(path, mesh_index=0):
    """Read only inference inputs. Never read fitted reference coefficients."""
    path = Path(path).resolve()
    override = None
    if path.suffix == ".json":
        index = json.loads(path.read_text())
        if not 0 <= mesh_index < len(index["shards"]):
            raise ValueError("mesh_index out of range")
        shard = index["shards"][mesh_index]
        path = path.parent / shard["path"]
        if sha256(path) != shard["sha256"]:
            raise ValueError("Geometry shard hash mismatch")
        override = shard["mesh"]
        mesh_index = 0
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"]))
        if meta["version"] != GEOMETRY_VERSION:
            raise ValueError("Unsupported geometry cache format")
        if not 0 <= mesh_index < len(meta["meshes"]):
            raise ValueError("mesh_index out of range")
        mesh = override or meta["meshes"][mesh_index]
        ids = np.flatnonzero(z["mesh_index"] == mesh_index)
        data = {k: np.array(z[k][ids], dtype=np.float32) for k in ("inputs", "origin", "frame", "radius")}
    if not len(ids):
        raise ValueError("Selected geometry contains no patches")
    p = len(ids)
    if data["inputs"].ndim != 3 or data["inputs"].shape[2] != 6:
        raise ValueError("Geometry inputs must have shape [patches, points, 6]")
    if data["origin"].shape != (p, 3) or data["frame"].shape != (p, 3, 3) or data["radius"].shape != (p,):
        raise ValueError("Invalid geometry side information shapes")
    if any(not np.isfinite(x).all() for x in data.values()) or np.any(data["radius"] <= 0):
        raise ValueError("Invalid geometry values or radius")
    if not np.allclose(data["frame"].transpose(0, 2, 1) @ data["frame"], np.eye(3), atol=2e-5):
        raise ValueError("Patch frames must be orthonormal")
    center, scale = np.asarray(mesh["center"], dtype=np.float64), float(mesh["scale"])
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid mesh normalization")
    return dict(**data, center=center, scale=scale, degree=meta["degree"],
                source_sha256=mesh["sha256"], cache_sha256=sha256(path),
                sampled_coverage=mesh.get("sampled_coverage"),
                tree=cKDTree(data["origin"].astype(np.float64)))


def tensor_scene(g, device):
    return {k: torch.as_tensor(g[k], device=device, dtype=torch.float32)
            for k in ("inputs", "origin", "frame", "radius", "scale")}


def normalized_queries(g, xyz):
    return np.asarray((np.asarray(xyz, dtype=np.float64) - g["center"]) / g["scale"], dtype=np.float32)


def neighbors(g, xyz_normalized, count):
    k = min(count, len(g["origin"]))
    indices = g["tree"].query(xyz_normalized, k=k)[1]
    return np.asarray(indices, dtype=np.int64).reshape(len(xyz_normalized), k)


def load_field(path):
    with np.load(path, allow_pickle=False) as z:
        xyz = np.asarray(z["xyz"], dtype=np.float64)
        cp = np.asarray(z["cp"], dtype=np.float64).reshape(-1)
        weights = np.asarray(z["weights"], dtype=np.float64).reshape(-1) if "weights" in z else np.ones(len(cp))
        meta = json.loads(str(z["metadata"])) if "metadata" in z else {}
    if xyz.shape != (len(cp), 3) or weights.shape != cp.shape or len(cp) == 0:
        raise ValueError("CFD field must contain xyz[N,3], cp[N], optional weights[N]")
    if not all(np.isfinite(a).all() for a in (xyz, cp, weights)) or np.any(weights <= 0):
        raise ValueError("CFD values must be finite and weights strictly positive")
    return dict(xyz=xyz, cp=cp, weights=weights, metadata=meta)


class Cases:
    def __init__(self, path, require_train=True, cache_size=2, weighting="provided"):
        self.path = Path(path).resolve()
        self.document = json.loads(self.path.read_text())
        if self.document.get("version") != "patch-pressure-data-v1":
            raise ValueError("Expected patch-pressure-data-v1 manifest")
        self.scope = self.document["evaluation_scope"]
        if self.scope not in ("held_out_cars", "held_out_queries_same_car", "held_out_regions_same_car"):
            raise ValueError("Choose explicit evaluation_scope")
        if weighting not in ("area", "provided"):
            raise ValueError("weighting must be area or provided")
        self.weighting = weighting
        self.condition_names = self.document["condition_names"]
        if len(set(self.condition_names)) != len(self.condition_names):
            raise ValueError("Condition names must be unique")
        self.cache_size = cache_size
        self.cache, self.geometry_cache = OrderedDict(), OrderedDict()
        self.entries = []
        ids = set()
        for entry in self.document["cases"]:
            e = dict(entry)
            if e["id"] in ids or e["split"] not in ("train", "val", "test"):
                raise ValueError("Case IDs must be unique; splits must be train/val/test")
            ids.add(e["id"])
            if not e.get("car_id"):
                raise ValueError("Every case needs a car_id for leakage checks")
            e["mesh_index"] = int(e.get("mesh_index", 0))
            for key in ("geometry", "field", "indices"):
                if key in e:
                    e[key] = resolve(self.path.parent, e[key])
            e["conditions"] = np.asarray(e["conditions"], dtype=np.float32).tolist()
            if len(e["conditions"]) != len(self.condition_names) or not np.isfinite(e["conditions"]).all():
                raise ValueError("conditions must match condition_names, with finite values")
            self.entries.append(e)
        if not self.entries:
            raise ValueError("No CFD cases")
        self.by_split = {s: [i for i, e in enumerate(self.entries) if e["split"] == s] for s in ("train", "val", "test")}
        if require_train and (not self.by_split["train"] or not self.by_split["val"]):
            raise ValueError("Training requires explicit nonempty train AND val splits; no fallback")
        self.audit = self._audit()

    def geometry(self, e):
        key = (e["geometry"], e["mesh_index"])
        if key not in self.geometry_cache:
            self.geometry_cache[key] = load_geometry(*key)
            if len(self.geometry_cache) > self.cache_size:
                self.geometry_cache.popitem(last=False)
        self.geometry_cache.move_to_end(key)
        return self.geometry_cache[key]

    def get(self, i):
        if i not in self.cache:
            e = self.entries[i]
            f = load_field(e["field"])
            if "indices" in e:
                with np.load(e["indices"], allow_pickle=False) as z:
                    indices = np.asarray(z[e.get("indices_key", "indices")])
                if indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) == 0:
                    raise ValueError("Indices must be a nonempty integer vector")
                if indices.min() < 0 or indices.max() >= len(f["cp"]) or len(np.unique(indices)) != len(indices):
                    raise ValueError("Duplicate or out-of-bounds CFD indices")
            else:
                indices = np.arange(len(f["cp"]))
            self.cache[i] = {**{k: f[k][indices] for k in ("xyz", "cp", "weights")},
                             "indices": indices, "metadata": f["metadata"]}
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(i)
        return self.cache[i]

    def _audit(self):
        cars, geometries, fields, query_groups, records = {}, {}, {}, {}, []
        for i, e in enumerate(self.entries):
            g, f = self.geometry(e), self.get(i)
            measure = f["metadata"].get("weight_measure", "unspecified")
            if self.weighting == "area" and measure != "area":
                raise ValueError(f"{e['id']}: area weighting required; convert with --measure area or explicitly use --weighting provided for a different experiment")
            digest = sha256(e["field"])
            idx_digest = sha256(e["indices"]) if "indices" in e else None
            if self.scope == "held_out_cars":
                for table, key in ((cars, e["car_id"]), (geometries, g["source_sha256"]), (fields, digest)):
                    if key in table and table[key] != e["split"]:
                        raise ValueError("A car, source mesh, or CFD file crosses held-out-car splits")
                    table[key] = e["split"]
            else:
                # For each car and operating condition, use one authoritative
                # field file and disjoint coordinate groups, even if the file
                # contains repeated coordinates (common in VTK boundary data).
                key = (e["car_id"], tuple(e["conditions"]))
                if "indices" not in e:
                    raise ValueError("Same-car query evaluation requires explicit index files")
                bucket = query_groups.setdefault(key, dict(field=digest, mesh=g["source_sha256"], splits={}, xyz={}, scale=g["scale"]))
                if bucket["field"] != digest or bucket["mesh"] != g["source_sha256"]:
                    raise ValueError("Same-car query splits must reference the same CFD file and source mesh")
                xyz = np.ascontiguousarray(f["xyz"])
                xyz[xyz == 0] = 0.  # canonicalize signed zero
                row_keys = set(xyz.view(np.dtype((np.void, xyz.dtype.itemsize * 3))).reshape(-1).tolist())
                for split, previous in bucket["splits"].items():
                    if split != e["split"] and row_keys.intersection(previous):
                        raise ValueError("CFD query coordinates leak across splits")
                bucket["splits"].setdefault(e["split"], set()).update(row_keys)
                bucket["xyz"].setdefault(e["split"], []).append(xyz)
            records.append(dict(id=e["id"], car_id=e["car_id"], split=e["split"], queries=len(f["cp"]),
                                conditions=e["conditions"], mesh_index=e["mesh_index"],
                                geometry_sha256=g["cache_sha256"], source_mesh_sha256=g["source_sha256"],
                                field_sha256=digest, indices_sha256=idx_digest,
                                indices_key=e.get("indices_key", "indices"),
                                degree=g["degree"], patches=len(g["origin"]),
                                geometry_sampled_coverage=g["sampled_coverage"], weight_measure=measure))
        separation = []
        for key, bucket in query_groups.items():
            if "train" not in bucket["xyz"]:
                continue
            tree = cKDTree(np.concatenate(bucket["xyz"]["train"]))
            for split in ("val", "test"):
                if split not in bucket["xyz"]:
                    continue
                distances = tree.query(np.concatenate(bucket["xyz"][split]))[0] / bucket["scale"]
                if self.scope == "held_out_regions_same_car":
                    definition = self.document.get("split_definition", {})
                    gap = definition.get("buffer_fraction_diagonal", 0.)
                    if gap <= 0 or distances.min() < gap - 1e-10:
                        raise ValueError("Blocked split violates its positive train/evaluation separation buffer")
                separation.append(dict(car_id=key[0], conditions=list(key[1]), split=split,
                                       min_distance_diagonal=float(distances.min()),
                                       median_distance_diagonal=float(np.median(distances))))
        payload = dict(evaluation_scope=self.scope, condition_names=self.condition_names, cases=records,
                       split_definition=self.document.get("split_definition"), query_separation=separation)
        payload["fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return payload

    def normalization(self):
        means, seconds, conditions = [], [], []
        for i in self.by_split["train"]:
            f = self.get(i)
            means.append(float(np.average(f["cp"], weights=f["weights"])))
            seconds.append(float(np.average(f["cp"] ** 2, weights=f["weights"])))
            conditions.append(self.entries[i]["conditions"])
        mean = float(np.mean(means))
        std = float(np.sqrt(max(float(np.mean(seconds)) - mean ** 2, 0.)))
        c = np.asarray(conditions, dtype=np.float64)
        cs = c.std(0)
        return dict(cp_mean=mean, cp_std=std if std > 1e-10 else 1.,
                    constant_training_pressure=std <= 1e-10,
                    condition_mean=c.mean(0).tolist(),
                    condition_std=np.where(cs > 1e-10, cs, 1.).tolist(),
                    source="training cases only; equal case weight; supplied query weights within case")


def case_metrics(pred, target, weights, train_std):
    pred, target, weights = (np.asarray(a, dtype=np.float64) for a in (pred, target, weights))
    mse = float(np.average((pred - target) ** 2, weights=weights))
    reference_power = float(np.average(target ** 2, weights=weights))
    reference_mean = float(np.average(target, weights=weights))
    reference_variance = float(np.average((target - reference_mean) ** 2, weights=weights))
    return dict(rmse_cp=float(np.sqrt(mse)), mae_cp=float(np.average(np.abs(pred - target), weights=weights)),
                bias_cp=float(np.average(pred - target, weights=weights)),
                nrmse_reference_rms=float(np.sqrt(mse / reference_power)) if reference_power > 1e-20 else None,
                nrmse_reference_std=float(np.sqrt(mse / reference_variance)) if reference_variance > 1e-20 else None,
                nrmse_train_std=float(np.sqrt(mse) / train_std),
                mse_cp=mse, reference_mean_square_cp=reference_power,
                reference_mean_cp=reference_mean, reference_variance_cp=reference_variance,
                max_abs_error_cp=float(np.max(np.abs(pred - target))), queries=len(target))


def summarize_metrics(records, scope, fingerprint, selection_metric="nrmse_std"):
    def average(key):
        values = [r[key] for r in records]
        return float(np.mean(values)) if all(v is not None for v in values) else None
    centered = average("nrmse_reference_std")
    mse = average("mse_cp")
    return dict(split=None, scope=scope, data_fingerprint=fingerprint,
                weighting="equal case weight; supplied query weights within each case",
                area_weighted=all(r.get("weight_measure") == "area" for r in records),
                metric_definitions=dict(
                    nrmse_reference_std="sqrt(sum(w*(cp_pred-cp_true)^2)/sum(w*(cp_true-mean_w(cp_true))^2)); prediction bias retained",
                    nrmse_reference_rms="sqrt(sum(w*(cp_pred-cp_true)^2)/sum(w*cp_true^2))",
                    nrmse_train_std="RMSE(cp) / training-only Cp standard deviation"),
                case_mean_nrmse_reference_std=centered,
                case_mean_nrmse_reference_rms=average("nrmse_reference_rms"),
                case_mean_rmse_cp=average("rmse_cp"), case_balanced_rmse_cp=float(np.sqrt(mse)),
                selection_metric=selection_metric, selection_score=centered if selection_metric == "nrmse_std" else mse,
                evaluated_queries=sum(r["queries"] for r in records), per_case=records)
