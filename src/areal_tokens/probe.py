"""Frozen spatial readouts with train-only ridge fitting and held-out families.

The readout is linear after a fixed Gaussian spatial interpolation. Interpolation
itself depends on geometry; this is not a globally linear coordinate-to-UDF model.
All token controls use the identical interpolation and query features.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from .data import GeometryDataset, collate_clouds, read_manifest, to_device
from .io import load_model, write_json
from .model import GeometryAutoencoder
from .prepare import save_npz
from .validation import validate_manifest

VARIANTS = (
    "query_position",
    "anchor_geometry",
    "initial_tokens",
    "trained_tokens",
    "random_initial_tokens",
    "random_tokens",
)


@dataclass
class RidgeModel:
    coefficient: np.ndarray
    intercept: float
    alpha: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        return features @ self.coefficient + self.intercept


class RidgeMoments:
    """Streaming float64 sufficient statistics; no all-cars feature cache."""

    def __init__(self, width):
        self.weight = 0.0
        self.x = np.zeros(width, dtype=np.float64)
        self.xx = np.zeros((width, width), dtype=np.float64)
        self.xy = np.zeros(width, dtype=np.float64)
        self.y = 0.0

    def update(self, features, target, *, shape_weight=1.0):
        x, y = np.asarray(features, dtype=np.float64), np.asarray(target, dtype=np.float64)
        if x.ndim != 2 or y.shape != (len(x),) or len(x) == 0 or shape_weight <= 0:
            raise ValueError("Invalid ridge observations or shape weight")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite ridge features or targets")
        weight = shape_weight / len(x)
        self.weight += shape_weight
        self.x += x.sum(0) * weight
        self.xx += x.T @ x * weight
        self.xy += x.T @ y * weight
        self.y += y.sum() * weight

    def fit(self, alphas) -> list[RidgeModel]:
        if self.weight <= 0 or not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas):
            raise ValueError("Ridge needs observations and finite positive alphas")
        mean_x, mean_y = self.x / self.weight, self.y / self.weight
        covariance = self.xx / self.weight - np.outer(mean_x, mean_x)
        scale = np.sqrt(np.maximum(np.diag(covariance), 0))
        scale[scale < 1e-8] = 1.0
        covariance /= np.outer(scale, scale)
        covariance = (covariance + covariance.T) / 2
        cross = (self.xy / self.weight - mean_x * mean_y) / scale
        values, vectors = np.linalg.eigh(covariance)
        projected = vectors.T @ cross
        fitted = []
        for alpha in alphas:
            coefficient = (vectors @ (projected / (np.maximum(values, 0) + alpha))) / scale
            fitted.append(RidgeModel(coefficient, float(mean_y - mean_x @ coefficient), float(alpha)))
        return fitted


@torch.inference_mode()
def frozen_features(model, random_model, sample, device, bandwidth):
    if bandwidth <= 0 or not np.isfinite(bandwidth):
        raise ValueError("Interpolation bandwidth must be finite and positive")
    batch = to_device(collate_clouds([sample]), device)
    packet, random_packet = model.encoder(batch), random_model.encoder(batch)
    query = (batch["queries"] - packet.center[:, None]) / packet.scale[:, None]
    distances = torch.cdist(query.float(), packet.positions.float()).square()
    log_measure = torch.where(packet.weights > 0, packet.weights.clamp_min(1e-38).log(), -torch.inf)
    interpolation = torch.softmax(-distances / (2 * bandwidth**2) + log_measure[:, None], dim=-1)
    q_features = model.encoder.position(query)
    geometry = torch.cat([model.encoder.position(packet.positions), packet.normals], dim=-1)
    values = dict(
        anchor_geometry=geometry,
        initial_tokens=packet.initial_tokens,
        trained_tokens=packet.tokens,
        random_initial_tokens=random_packet.initial_tokens,
        random_tokens=random_packet.tokens,
    )
    features = {"query_position": q_features[0].cpu().numpy().astype(np.float64)}
    for name, value in values.items():
        spatial = interpolation @ value.float()
        features[name] = torch.cat([q_features, spatial], dim=-1)[0].cpu().numpy().astype(np.float64)
    nearest = torch.cdist(query.float(), batch["points"].float()).min(dim=-1).values[0].cpu().numpy()
    target = (batch["udf"] / batch["scale"])[0].cpu().numpy().astype(np.float64)
    return features, target, nearest


def score_row(sample, target, prediction):
    squared = (prediction - target) ** 2
    variance = float(np.var(target))
    mse = float(np.mean(squared))
    return dict(
        id=sample["id"],
        family=sample["family"],
        mse=mse,
        physical_mse=mse * float(sample["scale"][0]) ** 2,
        mae=float(np.mean(np.abs(prediction - target))),
        mean_y=float(np.mean(target)),
        mean_y2=float(np.mean(target**2)),
        r2=1 - mse / variance if variance > 1e-14 else None,
    )


def summarize_scores(rows):
    if not rows:
        raise ValueError("Cannot score an empty probe split")
    groups = defaultdict(list)
    for row in rows:
        groups[row["family"]].append(row["mse"])
    family_mse = {family: float(np.mean(values)) for family, values in groups.items()}
    mse = float(np.mean([row["mse"] for row in rows]))
    variance = float(
        np.mean([row["mean_y2"] for row in rows]) - np.mean([row["mean_y"] for row in rows]) ** 2
    )
    return dict(
        shapes=len(rows),
        families=len(groups),
        normalized_rmse=float(np.sqrt(mse)),
        physical_rmse=float(np.sqrt(np.mean([row["physical_mse"] for row in rows]))),
        pooled_r2=1 - mse / variance if variance > 1e-14 else None,
        family_macro_mse=float(np.mean(list(family_mse.values()))),
        family_mse=family_mse,
    )


def paired_family_gain(reference, trained, seed, repeats=2000):
    keys = sorted(set(reference) & set(trained))
    differences = np.array([reference[key] - trained[key] for key in keys])
    result = dict(
        families=len(keys),
        mean_normalized_mse_gain=float(differences.mean()),
        definition="positive means trained tokens have lower error",
    )
    if len(keys) < 2:
        return dict(result, ci95=None, note="Need multiple independent families for a bootstrap interval")
    rng = np.random.default_rng(seed)
    bootstrap = np.array([rng.choice(differences, len(keys), replace=True).mean() for _ in range(repeats)])
    result["ci95"] = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    return result


def run_probe(
    checkpoint,
    manifest,
    output,
    device,
    *,
    queries=512,
    bandwidth=0.35,
    alphas=(1e-5, 1e-3, 0.1, 1.0, 10.0),
    random_seed=None,
) -> dict:
    """Fit on train cars, select ridge on val cars, score once on test cars."""
    if not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("Probe alphas must be finite and positive")
    model, config, _ = load_model(checkpoint, device)
    model.requires_grad_(False)
    random_seed = config.train.seed if random_seed is None else random_seed
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(random_seed)
        random_model = GeometryAutoencoder(config.model).to(device).eval().requires_grad_(False)
    # Also check cross-split exact copies and family boundaries before fitting.
    validate_manifest(manifest, require_targets=config.data.target_mode == "provided")
    records, metadata = read_manifest(manifest)
    data_config = replace(config.data, queries=queries)
    datasets = {
        split: GeometryDataset(
            [record for record in records if record["split"] == split],
            config.model,
            data_config,
            config.train.seed,
            mask_fraction=0.0,
            fixed_queries=True,
        )
        for split in ("train", "val", "test")
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Probe output must be empty: {output}")
    family_counts = Counter(record["family"] for record in datasets["train"].records)
    moments = {}
    for sample in datasets["train"]:
        features, target, _ = frozen_features(model, random_model, sample, device, bandwidth)
        for name, feature in features.items():
            if name not in moments:
                moments[name] = RidgeMoments(feature.shape[1])
            moments[name].update(feature, target, shape_weight=1 / family_counts[sample["family"]])
    fitted = {name: moment.fit(alphas) for name, moment in moments.items()}
    validation = {name: [[] for _ in alphas] for name in VARIANTS}
    for sample in datasets["val"]:
        features, target, _ = frozen_features(model, random_model, sample, device, bandwidth)
        for name in VARIANTS:
            for index, ridge in enumerate(fitted[name]):
                validation[name][index].append(score_row(sample, target, ridge.predict(features[name])))
    summaries = {name: [summarize_scores(rows) for rows in values] for name, values in validation.items()}
    selected = {
        name: int(np.argmin([value["family_macro_mse"] for value in values]))
        for name, values in summaries.items()
    }
    test = {name: [] for name in (*VARIANTS, "nearest_visible_sample")}
    for sample in datasets["test"]:
        features, target, nearest = frozen_features(model, random_model, sample, device, bandwidth)
        for name in VARIANTS:
            prediction = fitted[name][selected[name]].predict(features[name])
            test[name].append(score_row(sample, target, prediction))
        test["nearest_visible_sample"].append(score_row(sample, target, nearest))
    results, coefficients = {}, {}
    for name in VARIANTS:
        ridge = fitted[name][selected[name]]
        results[name] = dict(
            feature_width=len(ridge.coefficient),
            selected_alpha=ridge.alpha,
            validation=[
                dict(alpha=float(alpha), **summary)
                for alpha, summary in zip(alphas, summaries[name], strict=True)
            ],
            test=summarize_scores(test[name]),
            test_shapes=test[name],
        )
        coefficients[name + "_coefficient"] = ridge.coefficient
        coefficients[name + "_intercept"] = np.array(ridge.intercept)
        coefficients[name + "_alpha"] = np.array(ridge.alpha)
    results["nearest_visible_sample"] = dict(
        test=summarize_scores(test["nearest_visible_sample"]), test_shapes=test["nearest_visible_sample"]
    )
    trained = results["trained_tokens"]["test"]["family_mse"]
    gains = {
        name: paired_family_gain(results[name]["test"]["family_mse"], trained, config.train.seed)
        for name in ("query_position", "anchor_geometry", "initial_tokens", "random_tokens")
    }
    report = dict(
        checkpoint=str(checkpoint),
        random_seed=random_seed,
        data=metadata,
        queries_per_shape=queries,
        bandwidth=bandwidth,
        mask_fraction=0.0,
        target="independent off-surface UDF / retained per-shape RMS scale",
        method="fixed area-weighted Gaussian interpolation + query Fourier features + linear ridge",
        selection="family-weighted train fitting; train-only standardization; val family macro MSE",
        results=results,
        paired_family_gains=gains,
    )
    save_npz(output / "ridge_models.npz", **coefficients)
    write_json(output / "report.json", report)
    return report
