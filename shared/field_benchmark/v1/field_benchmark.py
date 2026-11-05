#!/usr/bin/env python3
"""Field-completion benchmark for Areal.

Predict the hidden part of a surface field from the visible part and the shape. Every method is given exactly the
same observed points, and is scored only where the field was hidden.

Methods
  copy     the value of the nearest observed point
  gp       Gaussian-process regression: Matern-5/2 kernel on position, constant mean by generalised least squares,
           hyperparameters tuned per setting on training cars (tune-gp)
  gp_geo   the same with the surface normal in the kernel (sign-free): a geometry-aware classical baseline
  areal    a field model trained by `train-areal`. With --name, any train-areal model, e.g. one trained with --scratch
           (same architecture, random initialisation): the control for what geometry pretraining adds

Observation settings, fixed once per split by `observations`
  patch-C-F    the field points are split into C Voronoi patches around farthest-point centres; patches are hidden in
               random order until the hidden share of surface area is as close to F as whole patches allow
  sensors-K    K scattered points drawn uniformly over the surface area
  unobserved   nothing visible: the surrogate case (with --unobserved; copy and the GPs skip it)

Stages, in order
  observations   fix the observed points of every car in a split (uses field point positions and areas only)
  train-areal    fine-tune a geometry checkpoint into a field model with no access to hidden values
  tune-gp        choose GP hyperparameters per setting on training cars
  predict        run one method on every observation set of a split
  check          rerun a method with the hidden values replaced by noise; its predictions must not change
  score          hidden-region error per setting, bootstrap intervals over cars, paired differences, curves

Data: the manifest and field directory that `areal train-field` uses. fields/<id>.npz holds points [N,3] in the
mesh's coordinates, values [N] and weights [N] (area per point); optional outward normals [N,3] add a force metric.
The validation split is the default. The test split needs --final on predict, check and score.

Example (one GPU):
  B=bench; M=prepared/manifest.json; F=fields; G=runs/base/best.pt
  python field_benchmark.py observations --manifest $M --fields $F --bench $B --split val
  python field_benchmark.py train-areal --bench $B --checkpoint $G --out $B/models/areal
  python field_benchmark.py train-areal --bench $B --checkpoint $G --out $B/models/areal_scratch --scratch
  python field_benchmark.py tune-gp --bench $B
  python field_benchmark.py predict --bench $B --method copy
  python field_benchmark.py predict --bench $B --method gp
  python field_benchmark.py predict --bench $B --method gp_geo
  python field_benchmark.py predict --bench $B --method areal --model $B/models/areal/field_best.pt
  python field_benchmark.py predict --bench $B --method areal --name areal_scratch \\
      --model $B/models/areal_scratch/field_best.pt
  python field_benchmark.py check --bench $B --method areal --model $B/models/areal/field_best.pt
  python field_benchmark.py score --bench $B
Final run: `observations --split test`, then predict, check and score with --split test --final.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

from areal_tokens.data import collate_clouds, load_cloud_file, read_manifest, stable_seed, to_device
from areal_tokens.field import (
    FieldEncoder,
    collate_field,
    field_loss,
    load_field_encoder,
    load_field_file,
    save_field_encoder,
)
from areal_tokens.geometry import farthest_indices, prepare_cloud
from areal_tokens.io import load_model, write_json
from areal_tokens.model import GeometryAutoencoder

CLOUD_KEYS = ("points", "normals", "point_weights", "anchors", "anchor_normals", "anchor_weights", "center",
              "scale", "area", "surfels", "surfel_normals", "surfel_weights")
METRICS = ("nrmse", "mae", "rel_l2", "force_rel")
# Fixed colour and marker per method, so a method keeps its colour whichever others are present.
METHOD_ORDER = ("areal", "areal_scratch", "gp_geo", "gp", "copy")
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


# ---------------------------------------------------------------------------------------------------------------
# Files, splits and devices


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def clean(value):
    """A JSON-safe copy: NaN and infinities become null, numpy scalars plain numbers."""
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def save_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, clean(value))


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def autocast(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        return nullcontext
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return lambda: torch.autocast("cuda", dtype=dtype)


def bench_config(bench: Path) -> dict:
    path = Path(bench) / "config.json"
    if not path.exists():
        raise SystemExit(f"{path} not found: run `observations` first")
    return json.loads(path.read_text())


def records_of(config: dict, split: str | None = None) -> list[dict]:
    records, _ = read_manifest(config["manifest"])
    return [record for record in records if split is None or record["split"] == split]


def require_final(split: str, final: bool) -> None:
    if split == "test" and not final:
        raise SystemExit("the test split is for the final run only: add --final")


# ---------------------------------------------------------------------------------------------------------------
# Fields and what a method may see


@dataclass
class Field:
    points: np.ndarray  # [N, 3] float64, mesh coordinates
    weights: np.ndarray  # [N] float64, area per point, positive
    values: np.ndarray  # [N] float64, the truth
    normals: np.ndarray | None  # [N, 3] unit, outward, when the field file has them


def load_field(fields, identity: str) -> Field:
    path = Path(fields) / f"{identity}.npz"
    arrays = load_field_file(path)
    normals = None
    with np.load(path, allow_pickle=False) as packed:
        if "normals" in packed.files:
            normals = np.asarray(packed["normals"], np.float64)
    if normals is not None:
        if normals.shape != arrays["points"].shape or not np.isfinite(normals).all():
            raise ValueError(f"{path}: normals must be finite and [N,3] like points")
        length = np.linalg.norm(normals, axis=1, keepdims=True)
        if (length < 1e-12).any():
            raise ValueError(f"{path}: zero normal")
        normals = normals / length
    weights = arrays["weights"].astype(np.float64)
    if (weights <= 0).any():
        raise ValueError(f"{path}: every field point needs a positive area weight")
    return Field(arrays["points"].astype(np.float64), weights, arrays["values"].astype(np.float64), normals)


def write_field(path, field: Field, values: np.ndarray) -> None:
    arrays = dict(points=field.points.astype(np.float32), values=values.astype(np.float32),
                  weights=field.weights.astype(np.float32))
    if field.normals is not None:
        arrays["normals"] = field.normals.astype(np.float32)
    np.savez(path, **arrays)


@dataclass
class Surface:
    """A car's geometry: all a method knows besides the observed values."""
    id: str
    record: dict
    points: np.ndarray
    weights: np.ndarray
    normals: np.ndarray  # unit; sign not guaranteed (from the field file, else the nearest prepared mesh sample)


@dataclass
class Observed:
    key: str
    setting: dict
    index: np.ndarray  # observed field points, sorted
    values: np.ndarray  # the field at those points, and nothing else


def surface_of(record: dict, field: Field) -> Surface:
    normals = field.normals
    if normals is None:
        cloud, _ = load_cloud_file(record["path"])
        normals = cloud.normals[cKDTree(cloud.points).query(field.points, workers=-1)[1]]
    return Surface(record["id"], record, field.points, field.weights, normals)


def moments(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Area-weighted mean and standard deviation."""
    w = weights / weights.sum()
    mean = float(w @ values)
    return mean, math.sqrt(max(float(w @ (values - mean) ** 2), 0.0))


# ---------------------------------------------------------------------------------------------------------------
# Observation sets


def settings_of(protocol: dict) -> list[dict]:
    settings = [dict(name=f"patch-{count}-{fraction:.2f}", kind="patch", count=int(count), fraction=float(fraction))
                for count in protocol["patch_counts"] for fraction in protocol["hidden"]]
    settings += [dict(name=f"sensors-{count}", kind="sensors", count=int(count)) for count in protocol["sensors"]]
    if protocol.get("unobserved"):
        settings.append(dict(name="unobserved", kind="none"))
    return settings


class Patches:
    """Voronoi cells of farthest-point centres on a car's field points, one partition per patch count."""

    def __init__(self, points: np.ndarray, weights: np.ndarray, counts=()):
        self.points, self.weights, self.cells = points, weights, {}
        for count in counts:
            self.assignment(count)

    def assignment(self, count: int) -> np.ndarray:
        count = max(2, min(int(count), len(self.points)))
        if count not in self.cells:
            centres = farthest_indices(self.points, self.weights, count)
            self.cells[count] = cKDTree(self.points[centres]).query(self.points, workers=-1)[1]
        return self.cells[count]


def draw_observed(setting: dict, patches: Patches, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sorted indices of the observed field points for one setting."""
    n = len(weights)
    if setting["kind"] == "none":
        return np.zeros(0, np.int64)
    if setting["kind"] == "sensors":
        count = min(int(setting["count"]), n - 1)
        return np.sort(rng.choice(n, count, replace=False, p=weights / weights.sum()))
    cells = patches.assignment(setting["count"])
    count = int(cells.max()) + 1
    area = np.bincount(cells, weights=weights, minlength=count)
    order = rng.permutation(count)
    cumulative = np.cumsum(area[order]) / area.sum()
    hidden = int(np.argmin(np.abs(cumulative[:-1] - setting["fraction"]))) + 1  # at least one patch each way
    return np.flatnonzero(~np.isin(cells, order[:hidden]))


def observation_rng(seed: int, purpose: str, identity: str, setting: str, repeat: int) -> np.random.Generator:
    return np.random.default_rng(stable_seed(seed, f"{purpose}|{identity}|{setting}|{repeat}", 0))


def hidden_share(weights: np.ndarray, observed: np.ndarray) -> float:
    return float(1.0 - weights[observed].sum() / weights.sum())


def read_observations(bench: Path, split: str):
    folder = Path(bench) / "observations"
    if not (folder / f"{split}.json").exists():
        raise SystemExit(f"no observations for the {split} split: run `observations --split {split}`")
    meta = json.loads((folder / f"{split}.json").read_text())
    with np.load(folder / f"{split}.npz") as packed:
        arrays = {key: packed[key].astype(np.int64) for key in packed.files}
    return meta, arrays


def by_car(entries: list[dict]) -> dict:
    groups = {}
    for entry in entries:
        groups.setdefault(entry["car"], []).append(entry)
    return groups


def cmd_observations(args) -> None:
    protocol = dict(patch_counts=sorted(set(args.patch_counts)), hidden=sorted(set(args.hidden)),
                    sensors=sorted(set(args.sensors)), repeats=args.repeats, unobserved=args.unobserved)
    if min(protocol["patch_counts"]) < 2 or min(protocol["sensors"]) < 1 or args.repeats < 1:
        raise SystemExit("patch counts must be at least 2, sensor counts and repeats at least 1")
    if not all(0 < fraction < 1 for fraction in protocol["hidden"]):
        raise SystemExit("hidden fractions must lie strictly between 0 and 1")
    bench = Path(args.bench)
    config = dict(manifest=str(Path(args.manifest).resolve()), fields=str(Path(args.fields).resolve()),
                  seed=args.seed, protocol=protocol)
    if (bench / "config.json").exists():
        if bench_config(bench) != json.loads(json.dumps(config)):
            raise SystemExit(f"{bench / 'config.json'} has a different manifest, field directory, seed or protocol; "
                             "use a new --bench")
    else:
        save_json(bench / "config.json", config)
    records = records_of(config, args.split)
    if not records:
        raise SystemExit(f"no {args.split} records in {config['manifest']}")
    settings = settings_of(protocol)
    arrays, entries, cars = {}, [], {}
    for record in records:
        field = load_field(config["fields"], record["id"])  # positions and areas are all that is used here
        patches = Patches(field.points, field.weights, protocol["patch_counts"])
        cars[record["id"]] = dict(points=len(field.points))
        for setting in settings:
            for repeat in range(protocol["repeats"]):
                rng = observation_rng(args.seed, "benchmark", record["id"], setting["name"], repeat)
                observed = draw_observed(setting, patches, field.weights, rng)
                key = f"o{len(entries)}"
                arrays[key] = observed.astype(np.int32)
                entries.append(dict(key=key, car=record["id"], setting=setting["name"], repeat=repeat,
                                    observed=int(len(observed)), hidden_fraction=hidden_share(field.weights, observed)))
    folder = bench / "observations"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"{args.split}.npz", **arrays)
    save_json(folder / f"{args.split}.json", dict(split=args.split, settings=settings, cars=cars, entries=entries,
                                                  created=now()))
    print(f"{args.split}: {len(cars)} cars x {len(settings)} settings x {args.repeats} repeats = {len(entries)} sets")
    for setting in settings:
        shares = [e["hidden_fraction"] for e in entries if e["setting"] == setting["name"]]
        print(f"  {setting['name']:>18}: hidden share {np.mean(shares):.3f} (min {min(shares):.3f}, "
              f"max {max(shares):.3f})")


# ---------------------------------------------------------------------------------------------------------------
# Scoring one prediction


def problem_metrics(field: Field, observed: np.ndarray, prediction: np.ndarray) -> dict:
    """Errors on the hidden points only, each weighted by its area.
    nrmse: RMSE divided by the car's field spread (1.0 is about the error of predicting the car's own mean).
    mae and rel_l2 (relative to the hidden truth): physical units, for comparison with other papers.
    force_rel: relative error of the integrated pressure force sum(w p n) with the hidden part predicted."""
    hidden = np.ones(len(field.values), bool)
    hidden[observed] = False
    weights, truth = field.weights[hidden], field.values[hidden]
    error = prediction[hidden] - truth
    _, spread = moments(field.values, field.weights)
    square, energy = float(weights @ error ** 2), float(weights @ truth ** 2)
    result = dict(nrmse=math.sqrt(square / weights.sum()) / spread if spread > 0 else math.nan,
                  mae=float(weights @ np.abs(error)) / float(weights.sum()),
                  rel_l2=math.sqrt(square / energy) if energy > 0 else math.nan, force_rel=math.nan)
    if field.normals is not None:
        completed = field.values.copy()
        completed[hidden] = prediction[hidden]
        force = (field.weights * field.values) @ field.normals
        estimate = (field.weights * completed) @ field.normals
        size = float(np.linalg.norm(force))
        result["force_rel"] = float(np.linalg.norm(estimate - force)) / size if size > 0 else math.nan
    return result


# ---------------------------------------------------------------------------------------------------------------
# Methods


class Method:
    name = "method"

    def supports(self, setting: dict) -> bool:
        return setting["kind"] != "none"

    def describe(self) -> dict:
        return dict(name=self.name)

    def predict_car(self, surface: Surface, observed: list[Observed]) -> list[np.ndarray]:
        raise NotImplementedError


class Copy(Method):
    name = "copy"

    def describe(self) -> dict:
        return dict(name=self.name, rule="value of the nearest observed field point (Euclidean)")

    def predict_car(self, surface, observed):
        return [o.values[cKDTree(surface.points[o.index]).query(surface.points, workers=-1)[1]] for o in observed]


def unit_frame(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Points about the area-weighted centroid, divided by the RMS radius (Areal's frame)."""
    w = weights / weights.sum()
    centre = w @ points
    return (points - centre) / math.sqrt(float(w @ ((points - centre) ** 2).sum(1)))


def kernel(xa, xb, na, nb, length: float, normal_length: float | None):
    """Matern-5/2 on scaled distance; with a normal length, the sign-free normal distance |nn^T - mm^T|_F joins
    the position distance."""
    distance = torch.cdist(xa, xb).square() / length ** 2
    if normal_length is not None:
        distance = distance + (2.0 - 2.0 * (na @ nb.T).square()).clamp_min(0) / normal_length ** 2
    scaled = math.sqrt(5.0) * distance.clamp_min(0).sqrt()
    return (1.0 + scaled + scaled.square() / 3.0) * torch.exp(-scaled)


def gp_subset(x: np.ndarray, observed: np.ndarray, cap: int) -> np.ndarray:
    """Positions into `observed` used by the GP: all of them, or past `cap` the three quarters nearest the hidden
    region plus a farthest-point spread of the rest."""
    if len(observed) <= cap:
        return np.arange(len(observed))
    hidden = np.setdiff1d(np.arange(len(x)), observed)
    gap = cKDTree(x[hidden]).query(x[observed], workers=-1)[0]
    near = np.argsort(gap, kind="stable")[: cap * 3 // 4]
    rest = np.setdiff1d(np.arange(len(observed)), near)
    spread = rest[farthest_indices(x[observed[rest]], np.ones(len(rest)), cap - len(near))]
    return np.sort(np.concatenate([near, spread]))


def gp_posterior_mean(x, normals, observed, values, params, cap, device, chunk=8192) -> np.ndarray:
    use = gp_subset(x, observed, cap)
    tensor = lambda a: torch.as_tensor(a, device=device, dtype=torch.float64)  # noqa: E731
    xo, no, y = tensor(x[observed[use]]), tensor(normals[observed[use]]), tensor(values[use])
    length, normal_length = params["length"], params.get("normal_length")
    gram = kernel(xo, xo, no, no, length, normal_length)
    eye = torch.eye(len(y), device=device, dtype=torch.float64)
    eta = params["eta"]
    for _ in range(12):
        factor, info = torch.linalg.cholesky_ex(gram + eta * eye)
        if int(info) == 0:
            break
        eta *= 10.0
    else:
        raise RuntimeError("GP kernel matrix is not positive definite")
    solved = torch.cholesky_solve(torch.stack([y, torch.ones_like(y)], 1), factor)
    mean = solved[:, 0].sum() / solved[:, 1].sum()  # generalised least-squares constant
    alpha = solved[:, 0] - mean * solved[:, 1]
    xa, na = tensor(x), tensor(normals)
    out = np.empty(len(x))
    for start in range(0, len(x), chunk):
        cross = kernel(xa[start:start + chunk], xo, na[start:start + chunk], no, length, normal_length)
        out[start:start + chunk] = (mean + cross @ alpha).cpu().numpy()
    return out


class GaussianProcess(Method):
    def __init__(self, name: str, hyper: dict, device: torch.device):
        self.name, self.hyper, self.device = name, hyper, device

    def describe(self) -> dict:
        return dict(name=self.name, kernel="Matern-5/2 on position" + (
            " and sign-free normal" if self.name == "gp_geo" else ""), mean="generalised least-squares constant",
            hyperparameters=self.hyper)

    def predict_car(self, surface, observed):
        x = unit_frame(surface.points, surface.weights)
        out = []
        for o in observed:
            if o.setting["name"] not in self.hyper["settings"]:
                raise SystemExit(f"{self.name}: no tuned hyperparameters for {o.setting['name']}; rerun tune-gp")
            params = dict(self.hyper["settings"][o.setting["name"]])
            if self.name != "gp_geo":
                params["normal_length"] = None
            out.append(gp_posterior_mean(x, surface.normals, o.index, o.values, params, self.hyper["cap"],
                                         self.device))
        return out


@dataclass
class Prepared:
    arrays: dict  # the encoder's view of the car, as `areal train-field` builds it
    sample_to_field: np.ndarray  # nearest field point of every encoder sample
    field_to_sample: np.ndarray  # nearest encoder sample of every field point
    anchor_index: np.ndarray  # the encoder sample at every anchor


def prepare_car(record: dict, points: np.ndarray, experiment) -> Prepared:
    cloud, _ = load_cloud_file(record["path"])
    model = experiment.model
    surfels = model.latents * model.surfels_per_token if model.surfels_per_token > 1 else 0
    arrays = prepare_cloud(cloud, model.latents, experiment.data.encoder_points, np.random.default_rng(0), surfels)
    world = arrays["points"].astype(np.float64) * float(arrays["scale"][0]) + arrays["center"].astype(np.float64)
    return Prepared(arrays, cKDTree(points).query(world, workers=-1)[1], cKDTree(world).query(points, workers=-1)[1],
                    cKDTree(arrays["points"]).query(arrays["anchors"], workers=-1)[1])


def encoder_inputs(prepared: Prepared, field_count: int, observed: np.ndarray, values: np.ndarray) -> dict:
    """The encoder's field channels, built from the observed values alone. An encoder sample carries a value when
    its nearest field point is observed; every observed point also reaches its own nearest sample, so a sparse
    sensor is never dropped (a sample reached by several averages them)."""
    n = len(prepared.arrays["points"])
    present, carried = np.zeros(n, np.float32), np.zeros(n, np.float32)
    if len(observed):
        is_observed = np.zeros(field_count, bool)
        is_observed[observed] = True
        value_at = np.zeros(field_count, np.float64)
        value_at[observed] = values
        own = is_observed[prepared.sample_to_field]
        present[own] = 1.0
        carried[own] = value_at[prepared.sample_to_field[own]]
        nearest = prepared.field_to_sample[observed]
        extra = ~own[nearest]
        if extra.any():
            total = np.bincount(nearest[extra], weights=values[extra], minlength=n)
            count = np.bincount(nearest[extra], minlength=n)
            take = count > 0
            present[take] = 1.0
            carried[take] = total[take] / count[take]
    sample = {key: prepared.arrays[key] for key in CLOUD_KEYS if key in prepared.arrays}
    sample.update(values=carried[:, None], present=present[:, None],
                  anchor_values=carried[prepared.anchor_index][:, None],
                  anchor_present=present[prepared.anchor_index][:, None])
    return sample


def collate_inputs(samples: list[dict]) -> dict:
    batch = collate_clouds([dict({key: s[key] for key in CLOUD_KEYS if key in s}, id="", family="", split="")
                            for s in samples])
    n = batch["points"].size(1)
    for key in ("values", "present"):
        batch[key] = torch.from_numpy(np.stack([np.pad(s[key], ((0, n - len(s[key])), (0, 0))) for s in samples]))
    for key in ("anchor_values", "anchor_present"):
        batch[key] = torch.from_numpy(np.stack([s[key] for s in samples]))
    return batch


def normalising(meta: dict, values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Shift and scale for one observation set, from training-set statistics or the observed values; never from
    hidden values."""
    if meta["normalise"] == "dataset" or len(values) == 0:
        if meta.get("mean") is None:
            raise ValueError("this model has no training-set statistics, so it cannot predict with nothing observed")
        return float(meta["mean"]), float(meta["std"])
    mean, spread = moments(values, weights)
    floor = meta["floor"] * meta["std"] if meta.get("std") else 1e-6
    return mean, max(spread, floor)


class Areal(Method):
    def __init__(self, name, model, experiment, meta, device, batch_size, amp, source=None):
        self.name, self.model, self.experiment, self.meta = name, model, experiment, meta
        self.device, self.batch_size, self.amp, self.source = device, batch_size, autocast(device, amp), source
        self.prepared = {}

    @classmethod
    def load(cls, name, path, device, batch_size, amp):
        model, experiment, state = load_field_encoder(path, device)
        meta = state.get("benchmark")
        source = dict(model=str(Path(path).resolve()), epoch=state.get("epoch"), legacy=meta is None)
        if meta is None:
            print(f"warning: {path} was not trained by train-areal. It learned with whole-field statistics, which it "
                  "does not get here, so its scores are not comparable.", file=sys.stderr)
            meta = dict(normalise="visible", mean=None, std=None, floor=0.0)
        return cls(name, model, experiment, meta, device, batch_size, amp, source)

    def supports(self, setting):
        return setting["kind"] != "none" or self.meta.get("mean") is not None

    def describe(self) -> dict:
        return dict(name=self.name, source=self.source, benchmark=self.meta)

    @torch.no_grad()
    def predict_car(self, surface, observed):
        self.model.eval()
        if surface.id not in self.prepared:
            self.prepared[surface.id] = prepare_car(surface.record, surface.points, self.experiment)
        prepared = self.prepared[surface.id]
        queries = torch.as_tensor(surface.points, dtype=torch.float32, device=self.device)
        out = []
        for start in range(0, len(observed), self.batch_size):
            chunk = observed[start:start + self.batch_size]
            shifts = [normalising(self.meta, o.values, surface.weights[o.index]) for o in chunk]
            samples = [encoder_inputs(prepared, len(surface.points), o.index, (o.values - mean) / scale)
                       for o, (mean, scale) in zip(chunk, shifts, strict=True)]
            batch = to_device(collate_inputs(samples), self.device)
            with self.amp():
                geometry = self.model.encode(batch, batch["values"], batch["present"], batch["anchor_values"],
                                             batch["anchor_present"])
                predicted = self.model.predict(geometry, queries[None].expand(len(chunk), -1, -1))
            for row, (mean, scale) in zip(predicted.float().cpu().numpy(), shifts, strict=True):
                out.append(row.astype(np.float64) * scale + mean)
        return out


def make_method(args, config: dict, device: torch.device) -> Method:
    if args.method == "copy":
        return Copy()
    if args.method in ("gp", "gp_geo"):
        path = Path(args.bench) / "gp" / f"{args.method}.json"
        if not path.exists():
            raise SystemExit(f"{path} not found: run tune-gp first")
        return GaussianProcess(args.method, json.loads(path.read_text()), device)
    if args.method == "areal":
        if not args.model:
            raise SystemExit("--model is required for areal")
        return Areal.load(args.name or "areal", args.model, device, args.batch_size, args.amp)
    raise SystemExit(f"unknown method {args.method}")


# ---------------------------------------------------------------------------------------------------------------
# Running a method


def predict_split(method: Method, config: dict, meta: dict, arrays: dict, fields, cars=None, keys=None,
                  progress=True):
    """Run a method on a split's observation sets. It sees each car's geometry and the observed values only.
    Returns predictions by observation key, per-car timings and the settings it skipped."""
    records = {record["id"]: record for record in records_of(config)}
    settings = {setting["name"]: setting for setting in meta["settings"]}
    groups = by_car(meta["entries"])
    predictions, timings, skipped = {}, [], set()
    for number, (car, entries) in enumerate(groups.items(), 1):
        if cars is not None and car not in cars:
            continue
        field = load_field(fields, car)
        observed = []
        for entry in entries:
            if keys is not None and entry["key"] not in keys:
                continue
            setting = settings[entry["setting"]]
            if not method.supports(setting):
                skipped.add(setting["name"])
                continue
            index = arrays[entry["key"]]
            observed.append(Observed(entry["key"], setting, index, field.values[index].copy()))
        if not observed:
            continue
        surface = surface_of(records[car], field)
        started = time.perf_counter()
        outputs = method.predict_car(surface, observed)
        seconds = time.perf_counter() - started
        for o, prediction in zip(observed, outputs, strict=True):
            prediction = np.asarray(prediction, np.float64)
            if prediction.shape != (len(field.points),) or not np.isfinite(prediction).all():
                raise RuntimeError(f"{method.name}: invalid prediction for {car} {o.setting['name']}")
            predictions[o.key] = prediction
        timings.append(dict(car=car, problems=len(observed), seconds=seconds))
        if progress:
            print(f"{method.name} {number}/{len(groups)} {car}: {len(observed)} sets in {seconds:.2f}s", flush=True)
    return predictions, timings, sorted(skipped)


def cmd_predict(args) -> None:
    require_final(args.split, args.final)
    bench, device = Path(args.bench), resolve_device(args.device)
    config = bench_config(bench)
    meta, arrays = read_observations(bench, args.split)
    method = make_method(args, config, device)
    predictions, timings, skipped = predict_split(method, config, meta, arrays, config["fields"])
    folder = bench / "predictions" / method.name
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"{args.split}.npz", **{k: v.astype(np.float32) for k, v in predictions.items()})
    seconds, problems = sum(t["seconds"] for t in timings), sum(t["problems"] for t in timings)
    save_json(folder / f"{args.split}.json", dict(
        method=method.describe(), split=args.split, skipped_settings=skipped,
        problems=problems, seconds=seconds, seconds_per_problem=seconds / max(problems, 1), device=str(device),
        timings=timings, created=now()))
    print(f"{method.name}: {problems} predictions, {seconds / max(problems, 1):.4f}s each -> {folder}")


def cmd_check(args) -> None:
    """Replace the hidden values by noise in a scratch copy of the field file and rebuild the method on that copy;
    a method that only uses what it is shown gives the same predictions."""
    require_final(args.split, args.final)
    bench, device = Path(args.bench), resolve_device(args.device)
    config = bench_config(bench)
    meta, arrays = read_observations(bench, args.split)
    settings = {setting["name"]: setting for setting in meta["settings"]}
    method = make_method(args, config, device)
    rng = np.random.default_rng(20261023)
    failures = checked = 0
    with tempfile.TemporaryDirectory() as temporary:
        for car, entries in list(by_car(meta["entries"]).items())[: args.cars]:
            field = load_field(config["fields"], car)
            _, spread = moments(field.values, field.weights)
            step = max(1, len(entries) // args.per_car)
            for entry in entries[::step][: args.per_car]:
                if not method.supports(settings[entry["setting"]]):
                    continue
                key = entry["key"]
                real, _, _ = predict_split(method, config, meta, arrays, config["fields"], {car}, {key},
                                           progress=False)
                hidden = np.ones(len(field.values), bool)
                hidden[arrays[key]] = False
                noisy = field.values.copy()
                noisy[hidden] = rng.normal(field.values.mean(), 10.0 * spread + 1.0, int(hidden.sum()))
                folder = Path(temporary) / key
                folder.mkdir()
                write_field(folder / f"{car}.npz", field, noisy)
                scrambled_config = dict(config, fields=str(folder))
                rebuilt = make_method(args, scrambled_config, device)
                scrambled, _, _ = predict_split(rebuilt, scrambled_config, meta, arrays, folder, {car}, {key},
                                                progress=False)
                change = float(np.abs(real[key] - scrambled[key]).max()) / max(spread, 1e-12)
                passed = change <= args.tolerance
                failures += not passed
                checked += 1
                print(f"{'PASS' if passed else 'FAIL'} {method.name} {car} {entry['setting']}: largest change "
                      f"{change:.2e} x field spread")
    print(f"{checked - failures}/{checked} passed")
    if failures or not checked:
        raise SystemExit(1)


# ---------------------------------------------------------------------------------------------------------------
# Gaussian-process tuning


def gp_grid_errors(x, normals, observed, values, weights, spread, combos, etas, cap, device, chunk=8192):
    """Hidden-point nRMSE of the GP posterior mean for every (length, normal length) pair and noise ratio."""
    use = observed[gp_subset(x, observed, cap)]
    hidden = np.setdiff1d(np.arange(len(x)), observed)
    tensor = lambda a: torch.as_tensor(a, device=device, dtype=torch.float64)  # noqa: E731
    xo, no, y = tensor(x[use]), tensor(normals[use]), tensor(values[use])
    ones = torch.ones_like(y)
    errors = np.full((len(combos), len(etas)), np.nan)
    for i, (length, normal_length) in enumerate(combos):
        eigenvalues, vectors = torch.linalg.eigh(kernel(xo, xo, no, no, length, normal_length))
        eigenvalues = eigenvalues.clamp_min(0)
        c, d = vectors.T @ y, vectors.T @ ones
        alphas, means = [], []
        for eta in etas:
            inverse = 1.0 / (eigenvalues + eta)
            mean = (d * inverse) @ c / ((d * inverse) @ d)
            alphas.append(vectors @ ((c - mean * d) * inverse))
            means.append(mean)
        alpha, mean = torch.stack(alphas, 1), torch.stack(means)
        square = torch.zeros(len(etas), device=device, dtype=torch.float64)
        for start in range(0, len(hidden), chunk):
            part = hidden[start:start + chunk]
            cross = kernel(tensor(x[part]), xo, tensor(normals[part]), no, length, normal_length)
            square += (tensor(weights[part])[:, None] * (mean + cross @ alpha - tensor(values[part])[:, None]) ** 2
                       ).sum(0)
        errors[i] = (torch.sqrt(square / float(weights[hidden].sum())) / spread).cpu().numpy()
    return errors


def cmd_tune_gp(args) -> None:
    bench, device = Path(args.bench), resolve_device(args.device)
    config = bench_config(bench)
    records = records_of(config, "train")
    if not records:
        raise SystemExit("tune-gp needs training cars")
    rng = np.random.default_rng(config["seed"])
    chosen = [records[i] for i in sorted(rng.choice(len(records), min(args.cars, len(records)), replace=False))]
    settings = [setting for setting in settings_of(config["protocol"]) if setting["kind"] != "none"]
    combos = {variant: [(length, normal_length) for length in args.lengths
                        for normal_length in (args.normal_lengths if variant == "gp_geo" else [None])]
              for variant in args.variants}
    errors = {variant: {s["name"]: np.full((len(combos[variant]), len(args.etas), len(chosen)), np.nan)
                        for s in settings} for variant in args.variants}
    started = time.perf_counter()
    for c, record in enumerate(chosen):
        field = load_field(config["fields"], record["id"])
        surface = surface_of(record, field)
        x = unit_frame(field.points, field.weights)
        patches = Patches(field.points, field.weights, config["protocol"]["patch_counts"])
        _, spread = moments(field.values, field.weights)
        if spread <= 0:
            continue
        for setting in settings:
            rng = observation_rng(config["seed"], "tune", record["id"], setting["name"], 0)
            observed = draw_observed(setting, patches, field.weights, rng)
            for variant in args.variants:
                errors[variant][setting["name"]][:, :, c] = gp_grid_errors(
                    x, surface.normals, observed, field.values, field.weights, spread, combos[variant], args.etas,
                    args.max_observed, device)
        print(f"tune-gp {c + 1}/{len(chosen)} {record['id']} ({time.perf_counter() - started:.0f}s)", flush=True)
    for variant in args.variants:
        chosen_settings = {}
        for setting in settings:
            mean = np.nanmean(errors[variant][setting["name"]], axis=2)
            i, j = np.unravel_index(np.nanargmin(mean), mean.shape)
            length, normal_length = combos[variant][i]
            chosen_settings[setting["name"]] = dict(length=length, normal_length=normal_length, eta=args.etas[j],
                                                    tune_nrmse=float(mean[i, j]), grid_nrmse=mean.tolist())
            edge = [name for name, value, grid in (("length", length, args.lengths),
                                                    ("normal length", normal_length, args.normal_lengths),
                                                    ("eta", args.etas[j], args.etas))
                    if value is not None and value in (min(grid), max(grid)) and len(grid) > 1]
            note = f" (at the edge of the grid: {', '.join(edge)})" if edge else ""
            print(f"{variant} {setting['name']}: length {length}, normal length {normal_length}, eta {args.etas[j]}, "
                  f"tuning nRMSE {mean[i, j]:.3f}{note}")
        save_json(bench / "gp" / f"{variant}.json", dict(
            variant=variant, cap=args.max_observed, cars=[record["id"] for record in chosen],
            grid=dict(lengths=args.lengths, normal_lengths=args.normal_lengths if variant == "gp_geo" else None,
                      etas=args.etas, combos=combos[variant]),
            settings=chosen_settings, created=now()))


# ---------------------------------------------------------------------------------------------------------------
# Training the Areal field model without leaks


def training_setting(rng: np.random.Generator, protocol: dict, mix: dict) -> dict:
    """One draw from the training mixture, which covers every benchmark setting: nothing observed, scattered
    sensors (log-uniform count), or hidden patches (a benchmark patch count, uniform hidden fraction)."""
    u = rng.random()
    if u < mix["unobserved"]:
        return dict(kind="none")
    if u < mix["unobserved"] + mix["sensors"]:
        low, high = math.log(min(protocol["sensors"])), math.log(max(protocol["sensors"]))
        return dict(kind="sensors", count=max(1, int(round(math.exp(rng.uniform(low, high))))))
    low = max(0.02, min(protocol["hidden"]) - 0.05)
    high = min(0.98, max(protocol["hidden"]) + 0.05)
    return dict(kind="patch", count=int(rng.choice(protocol["patch_counts"])), fraction=float(rng.uniform(low, high)))


def training_statistics(fields: list[Field]) -> tuple[float, float]:
    """Mean and spread of the field over the training cars: each car counts equally, each point by its area."""
    means, squares = [], []
    for field in fields:
        w = field.weights / field.weights.sum()
        means.append(float(w @ field.values))
        squares.append(float(w @ field.values ** 2))
    mean = float(np.mean(means))
    return mean, math.sqrt(max(float(np.mean(squares)) - mean ** 2, 1e-24))


class Completion(Dataset):
    """Training examples: a random observation set on the field points, encoder inputs from the observed values
    alone, and query points drawn in proportion to area with their normalised values as targets (so equal query
    weights give the area-weighted loss)."""

    def __init__(self, records, fields, experiment, protocol, queries, seed):
        self.cars = []
        for record in records:
            field = load_field(fields, record["id"])
            self.cars.append((record, field, prepare_car(record, field.points, experiment),
                              Patches(field.points, field.weights, protocol["patch_counts"])))
        self.protocol, self.queries, self.seed, self.epoch, self.meta = protocol, queries, seed, 0, None

    def __len__(self):
        return len(self.cars)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict:
        record, field, prepared, patches = self.cars[index]
        rng = np.random.default_rng(stable_seed(self.seed, record["id"] + ":train", self.epoch))
        observed = draw_observed(training_setting(rng, self.protocol, self.meta["mix"]), patches, field.weights, rng)
        mean, scale = normalising(self.meta, field.values[observed], field.weights[observed])
        values = (field.values - mean) / scale  # targets everywhere; the inputs below see the observed entries only
        sample = encoder_inputs(prepared, len(field.points), observed, values[observed])
        chosen = rng.choice(len(field.points), self.queries, replace=True, p=field.weights / field.weights.sum())
        hidden = np.ones(len(field.points), np.float32)
        hidden[observed] = 0.0
        sample.update(id=record["id"], family=record["family"], split=record["split"],
                      queries=field.points[chosen].astype(np.float32), target=values[chosen].astype(np.float32),
                      hidden=hidden[chosen], query_weights=np.full(self.queries, 1.0 / self.queries, np.float32))
        return sample


def selection_problems(config: dict, records: list[dict], limit: int) -> list:
    """Fixed validation problems for choosing the epoch: every hidden fraction at the middle patch count and every
    sensor count, on up to `limit` validation cars. Drawn apart from the benchmark's own observation sets."""
    protocol = config["protocol"]
    middle = protocol["patch_counts"][len(protocol["patch_counts"]) // 2]
    settings = [s for s in settings_of(protocol)
                if (s["kind"] == "patch" and s["count"] == middle) or s["kind"] == "sensors"]
    rng = np.random.default_rng(config["seed"])
    if len(records) > limit:
        records = [records[i] for i in sorted(rng.choice(len(records), limit, replace=False))]
    problems = []
    for record in records:
        field = load_field(config["fields"], record["id"])
        patches = Patches(field.points, field.weights, [middle])
        observed = []
        for setting in settings:
            rng = observation_rng(config["seed"], "select", record["id"], setting["name"], 0)
            index = draw_observed(setting, patches, field.weights, rng)
            observed.append(Observed(f"{record['id']}|{setting['name']}", setting, index, field.values[index].copy()))
        problems.append((surface_of(record, field), field, observed))
    return problems


def selection_score(method: Method, problems: list) -> float:
    scores = []
    for surface, field, observed in problems:
        predictions = method.predict_car(surface, observed)
        scores += [problem_metrics(field, o.index, p)["nrmse"] for o, p in zip(observed, predictions, strict=True)]
    return float(np.nanmean(scores))


def cmd_train_areal(args) -> None:
    bench, device = Path(args.bench), resolve_device(args.device)
    config = bench_config(bench)
    torch.manual_seed(args.seed)
    base, experiment, state = load_model(args.checkpoint, "cpu")
    if args.scratch:
        base = GeometryAutoencoder(experiment.model)  # same architecture, fresh weights (seeded above)
    train_records, val_records = records_of(config, "train"), records_of(config, "val")
    if not train_records or not val_records:
        raise SystemExit("train-areal needs train and val records")
    data = Completion(train_records, config["fields"], experiment, config["protocol"], args.queries, args.seed)
    mean, spread = training_statistics([field for _, field, _, _ in data.cars])
    mix = dict(unobserved=args.mix_unobserved, sensors=args.mix_sensors)
    if mix["unobserved"] + mix["sensors"] > 1:
        raise SystemExit("--mix-unobserved plus --mix-sensors must not exceed 1")
    meta = dict(version=1, normalise=args.normalise, mean=mean, std=spread, floor=args.floor, mix=mix,
                protocol=config["protocol"], scratch=args.scratch, base_checkpoint=str(Path(args.checkpoint).resolve()),
                train_cars=len(train_records), queries=args.queries,
                visible_weight=args.visible_weight, epochs=args.epochs, seed=args.seed)
    data.meta = meta

    model = FieldEncoder(base).to(device)
    new = [model.head.weight, model.head.bias, model.base.encoder.embed[0].weight, model.base.encoder.embed[0].bias]
    new_ids = {id(parameter) for parameter in new}
    old = [parameter for parameter in model.parameters() if id(parameter) not in new_ids]
    lr = args.lr if args.lr is not None else (args.new_lr if args.scratch else 1e-4)
    meta.update(lr=lr, new_lr=args.new_lr)
    optimizer = torch.optim.AdamW([dict(params=old, lr=lr), dict(params=new, lr=args.new_lr)],
                                  weight_decay=args.weight_decay)
    steps = args.epochs * math.ceil(len(data) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[lr, args.new_lr], total_steps=steps,
                                                    pct_start=0.1, anneal_strategy="cos")
    amp = autocast(device, args.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp == "fp16")

    problems = selection_problems(config, val_records, args.select_cars)
    selector = Areal("selection", model, experiment, meta, device, args.eval_batch_size, args.amp)
    copy_score = selection_score(Copy(), problems)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    extra = dict(config=state["config"], base_checkpoint=str(Path(args.checkpoint).resolve()), benchmark=meta)
    print(f"{len(data)} training cars; field mean {mean:.4g}, spread {spread:.4g}; {len(problems)} validation cars; "
          f"copying the nearest observed value scores {copy_score:.4f} there", flush=True)
    history, best, started = [], math.inf, time.time()
    for epoch in range(1, args.epochs + 1):
        data.set_epoch(epoch)
        loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                            collate_fn=collate_field, generator=torch.Generator().manual_seed(args.seed * 7919 + epoch))
        model.train()
        losses = []
        for batch in loader:
            batch = to_device(batch, device)
            with amp():
                loss = field_loss(model, batch, args.visible_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach()))
        score = selection_score(selector, problems)
        record = dict(epoch=epoch, train_loss=float(np.mean(losses)), val_nrmse=score, copy_val_nrmse=copy_score,
                      seconds=round(time.time() - started))
        history.append(record)
        if score < best:
            best = score
            save_field_encoder(out / "field_best.pt", model, epoch=epoch, **extra)
        save_field_encoder(out / "field_last.pt", model, epoch=epoch, **extra)
        save_json(out / "history.json", history)
        print(f"epoch {epoch}/{args.epochs} loss {record['train_loss']:.4f} | val nRMSE {score:.4f} "
              f"(best {best:.4f}, copy {copy_score:.4f}) | {record['seconds']}s", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Scoring


def bootstrap(values, boot: int, rng: np.random.Generator) -> dict:
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(mean=math.nan, low=math.nan, high=math.nan, cars=0)
    draws = values[rng.integers(0, len(values), (boot, len(values)))].mean(1)
    return dict(mean=float(values.mean()), low=float(np.percentile(draws, 2.5)),
                high=float(np.percentile(draws, 97.5)), cars=int(len(values)))


def summarise(rows: list[dict], methods: list[str], meta: dict, reference: str | None, boot: int, seed: int) -> dict:
    """Per setting and method: each metric averaged over repeats per car, then over cars with a bootstrap
    interval; and the paired per-car difference from the reference method."""
    table = {}
    for row in rows:
        table.setdefault((row["method"], row["setting"]), {}).setdefault(row["car"], []).append(row)
    per_car = {key: {car: {metric: float(np.mean([r[metric] for r in group])) for metric in METRICS}
                     for car, group in cars.items()} for key, cars in table.items()}
    rng = np.random.default_rng(seed)
    summary = []
    for setting in meta["settings"]:
        shares = [e["hidden_fraction"] for e in meta["entries"] if e["setting"] == setting["name"]]
        item = dict(setting, achieved_hidden=float(np.mean(shares)), methods={}, paired={})
        for method in methods:
            cars = per_car.get((method, setting["name"]))
            if not cars:
                continue
            item["methods"][method] = {metric: bootstrap([v[metric] for v in cars.values()], boot, rng)
                                       for metric in METRICS}
            anchor = per_car.get((reference, setting["name"]))
            if reference and anchor and method != reference:
                common = sorted(set(cars) & set(anchor))
                difference = np.array([cars[c]["nrmse"] - anchor[c]["nrmse"] for c in common])
                stats = bootstrap(difference, boot, rng)
                finite = difference[np.isfinite(difference)]
                stats["reference_better"] = float((finite > 0).mean()) if len(finite) else math.nan
                item["paired"][method] = stats
        summary.append(item)
    return dict(reference=reference, settings=summary)


def cell(stats: dict | None, signed: bool = False) -> str:
    if not stats or not math.isfinite(stats["mean"]):
        return "–"
    form = "{:+.3f}" if signed else "{:.3f}"
    return f"{form.format(stats['mean'])} [{form.format(stats['low'])}, {form.format(stats['high'])}]"


def report(summary: dict, methods: list[str], split: str, timings: dict, cars: int) -> str:
    settings = summary["settings"]
    lines = [f"# Field completion, {split} split ({cars} cars)", "",
             "Hidden-region nRMSE: area-weighted RMSE on the hidden points divided by the car's area-weighted field "
             "spread (1.0 is about the error of predicting the car's own mean). Mean over cars, 95% bootstrap "
             "interval over cars. MAE, relative L2 and force error are in the JSON and CSV.", ""]
    patches = [s for s in settings if s["kind"] == "patch"]
    if patches:
        lines += ["## Hidden patches", "", "| patches | hidden: target → achieved | " + " | ".join(methods) + " |",
                  "|---|---|" + "---|" * len(methods)]
        for s in patches:
            lines.append(f"| {s['count']} | {s['fraction']:.2f} → {s['achieved_hidden']:.2f} | "
                         + " | ".join(cell(s["methods"].get(m, {}).get("nrmse")) for m in methods) + " |")
        lines.append("")
    sensors = [s for s in settings if s["kind"] == "sensors"]
    if sensors:
        lines += ["## Scattered sensors", "", "| sensors | hidden share | " + " | ".join(methods) + " |",
                  "|---|---|" + "---|" * len(methods)]
        for s in sensors:
            lines.append(f"| {s['count']} | {s['achieved_hidden']:.3f} | "
                         + " | ".join(cell(s["methods"].get(m, {}).get("nrmse")) for m in methods) + " |")
        lines.append("")
    for s in settings:
        if s["kind"] == "none":
            lines += ["## Nothing observed (surrogate)", "", "| " + " | ".join(methods) + " |",
                      "|" + "---|" * len(methods),
                      "| " + " | ".join(cell(s["methods"].get(m, {}).get("nrmse")) for m in methods) + " |", ""]
    reference = summary["reference"]
    others = [m for m in methods if m != reference]
    if reference in methods and others:
        lines += [f"## Paired difference from {reference}", "",
                  f"Per car, method nRMSE minus {reference} nRMSE: positive means {reference} is better. In brackets, "
                  f"the share of cars where {reference} is better.", "",
                  "| setting | " + " | ".join(others) + " |", "|---|" + "---|" * len(others)]
        for s in settings:
            parts = []
            for m in others:
                stats = s["paired"].get(m)
                parts.append("–" if not stats or not math.isfinite(stats["mean"])
                             else f"{cell(stats, signed=True)} ({stats['reference_better']:.0%})")
            lines.append(f"| {s['name']} | " + " | ".join(parts) + " |")
        lines.append("")
    lines += ["## Time", "", "| method | seconds per observation set | device |", "|---|---|---|"]
    for m in methods:
        info = timings.get(m, {})
        lines.append(f"| {m} | {info.get('seconds_per_problem', math.nan):.4f} | {info.get('device', '')} |")
    return "\n".join(lines) + "\n"


def method_styles(methods: list[str]) -> tuple[dict, dict]:
    ordered = [m for m in METHOD_ORDER if m in methods] + sorted(m for m in methods if m not in METHOD_ORDER)
    slots = {m: METHOD_ORDER.index(m) for m in ordered if m in METHOD_ORDER}
    free = [i for i in range(len(PALETTE)) if i not in slots.values()]
    for m in ordered:
        if m not in slots:
            if not free:
                raise SystemExit("more than eight methods: plot a subset with --methods")
            slots[m] = free.pop(0)
    return {m: PALETTE[slots[m]] for m in ordered}, {m: MARKERS[slots[m]] for m in ordered}


def plot(summary: dict, methods: list[str], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed: no figure")
        return False
    surface, ink, muted, rule = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    colour, marker = method_styles(methods)
    settings = summary["settings"]
    counts = sorted({s["count"] for s in settings if s["kind"] == "patch"})
    panels = [("patch", count) for count in counts]
    if any(s["kind"] == "sensors" for s in settings):
        panels.append(("sensors", None))
    if not panels:
        return False
    figure, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.3), sharey=True, squeeze=False,
                                facecolor=surface)
    handles = {}
    for axis, (kind, count) in zip(axes[0], panels, strict=True):
        axis.set_facecolor(surface)
        rows = [s for s in settings if s["kind"] == kind and (count is None or s["count"] == count)]
        for name in methods:
            points = [(s["achieved_hidden"] if kind == "patch" else s["count"], s["methods"][name]["nrmse"])
                      for s in rows if name in s["methods"] and math.isfinite(s["methods"][name]["nrmse"]["mean"])]
            if not points:
                continue
            points.sort(key=lambda p: p[0])
            x = [p[0] for p in points]
            axis.fill_between(x, [p[1]["low"] for p in points], [p[1]["high"] for p in points], color=colour[name],
                              alpha=0.10, linewidth=0)
            (line,) = axis.plot(x, [p[1]["mean"] for p in points], color=colour[name], linewidth=1.8,
                                marker=marker[name], markersize=5.5, markeredgecolor=surface, markeredgewidth=1.2,
                                solid_capstyle="round", solid_joinstyle="round", label=name)
            handles.setdefault(name, line)
        axis.set_title(f"{count} patches" if kind == "patch" else "Scattered sensors", loc="left", color=ink,
                       fontsize=10)
        axis.set_xlabel("Hidden share of surface area" if kind == "patch" else "Observed points", color=muted,
                        fontsize=9)
        if kind == "sensors":
            axis.set_xscale("log")
            ticks = sorted(s["count"] for s in rows)
            axis.set_xticks(ticks, [str(t) for t in ticks])
            axis.minorticks_off()
        axis.grid(True, color=rule, linewidth=0.6, linestyle="-")
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(rule)
        axis.tick_params(colors=muted, labelsize=8)
        axis.set_ylim(bottom=0)
    axes[0][0].set_ylabel("Hidden-region nRMSE", color=muted, fontsize=9)
    names = [m for m in methods if m in handles]
    figure.legend([handles[m] for m in names], names, loc="upper center", ncol=len(names), frameon=False,
                  fontsize=9, labelcolor=ink, bbox_to_anchor=(0.5, 1.0))
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(path, dpi=200, facecolor=surface, bbox_inches="tight")
    plt.close(figure)
    return True


def cmd_score(args) -> None:
    require_final(args.split, args.final)
    bench = Path(args.bench)
    config = bench_config(bench)
    meta, arrays = read_observations(bench, args.split)
    root = bench / "predictions"
    methods = args.methods
    if not methods and root.exists():
        methods = sorted(p.name for p in root.iterdir() if (p / f"{args.split}.npz").exists())
    if not methods:
        raise SystemExit("no predictions to score")
    missing = [m for m in methods if not (root / m / f"{args.split}.json").exists()]
    if missing:
        raise SystemExit(f"no {args.split} predictions for {', '.join(missing)}")
    ordered = [m for m in METHOD_ORDER if m in methods] + sorted(m for m in methods if m not in METHOD_ORDER)
    loaded, timings = {}, {}
    for name in ordered:
        info = json.loads((root / name / f"{args.split}.json").read_text())
        with np.load(root / name / f"{args.split}.npz") as packed:
            loaded[name] = (info, {key: packed[key] for key in packed.files})
        timings[name] = dict(seconds_per_problem=info["seconds_per_problem"], device=info["device"])
    rows = []
    for car, entries in by_car(meta["entries"]).items():
        field = load_field(config["fields"], car)
        for entry in entries:
            for name in ordered:
                info, predictions = loaded[name]
                if entry["key"] not in predictions:
                    if entry["setting"] in info["skipped_settings"]:
                        continue
                    raise SystemExit(f"{name} has no prediction for {car} {entry['setting']}")
                prediction = predictions[entry["key"]].astype(np.float64)
                rows.append(dict(method=name, car=car, setting=entry["setting"], repeat=entry["repeat"],
                                 hidden_fraction=entry["hidden_fraction"], observed=entry["observed"],
                                 **problem_metrics(field, arrays[entry["key"]], prediction)))
    reference = args.reference if args.reference in ordered else None
    summary = summarise(rows, ordered, meta, reference, args.boot, config["seed"])
    summary.update(split=args.split, cars=len(meta["cars"]), timings=timings,
                   methods={name: loaded[name][0]["method"] for name in ordered}, created=now())
    folder = bench / "scores"
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{args.split}_per_problem.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(clean(rows))
    save_json(folder / f"{args.split}.json", summary)
    text = report(summary, ordered, args.split, timings, len(meta["cars"]))
    (folder / f"{args.split}.md").write_text(text)
    drawn = plot(summary, ordered, folder / f"{args.split}_curves.png")
    print(text)
    print(f"written: {folder / f'{args.split}.md'}, .json, _per_problem.csv" + (", _curves.png" if drawn else ""))


# ---------------------------------------------------------------------------------------------------------------
# Command line


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("observations", help="fix the observed points of every car in a split")
    p.add_argument("--manifest", required=True, help="the manifest used by `areal train-field`")
    p.add_argument("--fields", required=True, help="directory of <id>.npz field files")
    p.add_argument("--bench", required=True, help="output directory for the whole benchmark")
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--patch-counts", type=int, nargs="+", default=[16, 64, 256],
                   help="patches per car: fewer patches make larger holes")
    p.add_argument("--hidden", type=float, nargs="+", default=[0.1, 0.25, 0.4, 0.6, 0.8, 0.9],
                   help="target hidden shares of surface area")
    p.add_argument("--sensors", type=int, nargs="+", default=[5, 10, 20, 50, 100, 200, 500])
    p.add_argument("--repeats", type=int, default=1, help="observation sets per car and setting")
    p.add_argument("--unobserved", action="store_true", help="also score the surrogate case, nothing observed")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("train-areal", help="fine-tune a geometry checkpoint into a field model, leak-free")
    p.add_argument("--bench", required=True)
    p.add_argument("--checkpoint", required=True, help="geometry checkpoint from `areal train`")
    p.add_argument("--out", required=True)
    p.add_argument("--scratch", action="store_true", help="same architecture, random initialisation (the control)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=None,
                   help="learning rate of the pretrained weights (default 1e-4; with --scratch, --new-lr)")
    p.add_argument("--new-lr", type=float, default=1e-3, help="learning rate of the new field input and head")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--queries", type=int, default=2048, help="query points per training example")
    p.add_argument("--visible-weight", type=float, default=0.25, help="loss weight of observed query points")
    p.add_argument("--normalise", choices=("dataset", "visible"), default="dataset",
                   help="dataset: training-set mean and spread; visible: each car's observed values")
    p.add_argument("--floor", type=float, default=0.1,
                   help="visible normalisation: smallest spread, as a share of the training-set spread")
    p.add_argument("--mix-unobserved", type=float, default=0.1, help="share of examples with nothing observed")
    p.add_argument("--mix-sensors", type=float, default=0.35, help="share of examples with scattered sensors")
    p.add_argument("--select-cars", type=int, default=64, help="validation cars used to choose the epoch")
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("tune-gp", help="choose GP hyperparameters per setting on training cars")
    p.add_argument("--bench", required=True)
    p.add_argument("--variants", nargs="+", choices=("gp", "gp_geo"), default=["gp", "gp_geo"])
    p.add_argument("--cars", type=int, default=8, help="training cars to tune on")
    p.add_argument("--lengths", type=float, nargs="+", default=[0.03, 0.06, 0.12, 0.25, 0.5, 1.0],
                   help="kernel lengths, in units of the car's RMS radius")
    p.add_argument("--normal-lengths", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    p.add_argument("--etas", type=float, nargs="+", default=[1e-6, 1e-4, 1e-3, 1e-2, 1e-1],
                   help="noise variance relative to the kernel variance")
    p.add_argument("--max-observed", type=int, default=4000, help="largest observed set the GP solves exactly")
    p.add_argument("--device", default="auto")

    for command, text in (("predict", "run one method on every observation set of a split"),
                          ("check", "confirm a method's predictions do not depend on hidden values")):
        p = sub.add_parser(command, help=text)
        p.add_argument("--bench", required=True)
        p.add_argument("--split", choices=("train", "val", "test"), default="val")
        p.add_argument("--final", action="store_true", help="required for the test split")
        p.add_argument("--method", required=True, choices=("copy", "gp", "gp_geo", "areal"))
        p.add_argument("--model", help="field checkpoint for --method areal")
        p.add_argument("--name", help="name for an areal model, e.g. areal_scratch (default areal)")
        p.add_argument("--batch-size", type=int, default=16, help="observation sets per forward pass (areal)")
        p.add_argument("--device", default="auto")
        p.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
        if command == "check":
            p.add_argument("--cars", type=int, default=3)
            p.add_argument("--per-car", type=int, default=4)
            p.add_argument("--tolerance", type=float, default=1e-5,
                           help="largest allowed change, as a share of the car's field spread")

    p = sub.add_parser("score", help="score every method's predictions on a split")
    p.add_argument("--bench", required=True)
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--final", action="store_true", help="required for the test split")
    p.add_argument("--methods", nargs="*", help="default: every method with predictions for the split")
    p.add_argument("--reference", default="areal", help="method the others are compared with, per car")
    p.add_argument("--boot", type=int, default=2000, help="bootstrap resamples over cars")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    commands = {"observations": cmd_observations, "train-areal": cmd_train_areal, "tune-gp": cmd_tune_gp,
                "predict": cmd_predict, "check": cmd_check, "score": cmd_score}
    commands[args.command](args)


if __name__ == "__main__":
    main()
