"""Frozen reconstruction evaluation and portable vector-token export."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import GeometryDataset, collate_clouds, read_manifest, to_device
from .io import load_model, write_json
from .losses import reconstruction_loss, reconstruction_metrics
from .model import GeometryAutoencoder
from .prepare import save_npz


def make_loader(dataset, batch_size, workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_clouds,
        generator=torch.Generator().manual_seed(0),
        persistent_workers=False,
    )


def aggregate_rows(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot aggregate an empty evaluation")
    result = dict(
        shapes=len(rows),
        loss=float(np.mean([row["loss"] for row in rows])),
        rmse=float(np.sqrt(np.mean([row["mse"] for row in rows]))),
        normalized_rmse=float(np.sqrt(np.mean([row["normalized_mse"] for row in rows]))),
        mae=float(np.mean([row["mae"] for row in rows])),
    )
    normal_count = sum(row["normal_count"] for row in rows)
    result["normal_degrees"] = (
        sum(row["normal_angle_sum"] for row in rows) / normal_count if normal_count else None
    )
    for region in ("hidden", "visible"):
        selected = [row[f"{region}_mse"] for row in rows if row[f"{region}_mse"] is not None]
        result[f"{region}_rmse"] = float(np.sqrt(np.mean(selected))) if selected else None
        result[f"{region}_shapes"] = len(selected)
    return result


@torch.inference_mode()
def evaluate_model(
    model: GeometryAutoencoder,
    dataset: GeometryDataset,
    config: ExperimentConfig,
    device: torch.device,
    *,
    workers=0,
    ablation="none",
) -> dict:
    if ablation not in ("none", "zero_tokens", "initial_tokens"):
        raise ValueError("Ablation must be none, zero_tokens or initial_tokens")
    was_training = model.training
    model.eval()
    rows = []
    try:
        for batch in make_loader(dataset, config.train.batch_size, workers):
            batch = to_device(batch, device)
            packet = model.encoder(batch)
            if ablation == "zero_tokens":
                packet = packet.with_tokens(torch.zeros_like(packet.tokens))
            elif ablation == "initial_tokens":
                packet = packet.with_tokens(packet.initial_tokens)
            prediction = model.decode(packet, batch["queries"])
            if not all(torch.isfinite(value).all() for value in prediction.values()):
                raise FloatingPointError("Nonfinite reconstruction prediction")
            metrics = reconstruction_metrics(
                prediction, batch, unoriented=config.train.normal_loss == "unoriented"
            )
            # Preserve per-shape loss, including shapes with no near-surface normals.
            for i, identity in enumerate(batch["id"]):
                single_batch = {
                    key: value[i : i + 1] if torch.is_tensor(value) else value for key, value in batch.items()
                }
                single_prediction = {key: value[i : i + 1] for key, value in prediction.items()}
                row = dict(
                    id=identity,
                    family=batch["family"][i],
                    loss=float(reconstruction_loss(single_prediction, single_batch, config.train)["loss"]),
                    **{key: float(value[i]) for key, value in metrics.items()},
                )
                squared = (prediction["udf"][i] - batch["udf"][i]).square()
                for region, selected in (
                    ("hidden", batch["query_hidden"][i]),
                    ("visible", ~batch["query_hidden"][i]),
                ):
                    row[f"{region}_mse"] = float(squared[selected].mean()) if selected.any() else None
                    row[f"{region}_queries"] = int(selected.sum())
                rows.append(row)
    finally:
        model.train(was_training)
    groups = defaultdict(list)
    for row in rows:
        groups[row["family"]].append(row)
    families = {key: aggregate_rows(value) for key, value in groups.items()}
    # A large family must not dominate this second summary.
    family_macro = dict(
        rmse=float(np.sqrt(np.mean([value["rmse"] ** 2 for value in families.values()]))),
        normalized_rmse=float(
            np.sqrt(np.mean([value["normalized_rmse"] ** 2 for value in families.values()]))
        ),
        mae=float(np.mean([value["mae"] for value in families.values()])),
    )
    return dict(
        aggregate=aggregate_rows(rows),
        family_macro=family_macro,
        families=families,
        shapes=rows,
        ablation=ablation,
        mask_fraction=dataset.mask_fraction,
        target_mode=dataset.data.target_mode,
        hidden_region_definition="nearest reference surface sample belongs to a masked patch",
    )


def evaluate_checkpoint(
    checkpoint, manifest, output, device, *, split="test", masks=(0.0, 0.6), ablation="none", workers=0
) -> dict:
    if split not in ("train", "val", "test"):
        raise ValueError("Unknown split")
    model, config, _ = load_model(checkpoint, device)
    if masks is None:
        masks = (0.0, config.data.mask_fraction)
    records, metadata = read_manifest(manifest)
    chosen = [record for record in records if record["split"] == split]
    settings = []
    for mask in masks:
        dataset = GeometryDataset(
            chosen, config.model, config.data, config.train.seed, mask_fraction=mask, fixed_queries=True
        )
        settings.append(evaluate_model(model, dataset, config, device, workers=workers, ablation=ablation))
    report = dict(checkpoint=str(checkpoint), split=split, data=metadata, settings=settings)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report


@torch.inference_mode()
def encode_checkpoint(
    checkpoint, manifest, output, device, *, split="all", common_queries=None, workers=0
) -> dict:
    model, config, _ = load_model(checkpoint, device)
    records, metadata = read_manifest(manifest)
    if split != "all":
        records = [record for record in records if record["split"] == split]
    dataset = GeometryDataset(
        records, config.model, config.data, config.train.seed, mask_fraction=0.0, supervision=False
    )
    query = None
    if common_queries is not None:
        query_array = np.load(common_queries, allow_pickle=False)
        if query_array.ndim != 2 or query_array.shape[1] != 3 or not np.isfinite(query_array).all():
            raise ValueError("Common queries must be finite .npy world coordinates [Q,3]")
        query = torch.as_tensor(query_array, dtype=torch.float32, device=device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Embedding output must be empty: {output}")
    exports = []
    for batch in make_loader(dataset, config.train.batch_size, workers):
        batch = to_device(batch, device)
        packet = model.encoder(batch)
        common_features = None
        if query is not None:
            common_features = model.query_features(packet, query[None].expand(len(packet.tokens), -1, -1))
        for i, identity in enumerate(batch["id"]):
            valid = packet.weights[i] > 0
            arrays = dict(
                tokens=packet.tokens[i, valid],
                positions=packet.world_positions[i, valid],
                normalized_positions=packet.positions[i, valid],
                normals=packet.normals[i, valid],
                weights=packet.weights[i, valid],
                center=packet.center[i],
                scale=packet.scale[i],
                area=packet.area[i],
            )
            if common_features is not None:
                arrays.update(query_positions=query, query_features=common_features[i])
            arrays = {key: value.float().cpu().numpy() for key, value in arrays.items()}
            name = f"shape_{len(exports):07d}.npz"
            save_npz(output / name, **arrays)
            exports.append(dict(id=identity, family=batch["family"][i], split=batch["split"][i], path=name))
    report = dict(
        schema="geometry_latents_embeddings_v1",
        data=metadata,
        checkpoint=str(checkpoint),
        width=config.model.width,
        records=exports,
    )
    write_json(output / "manifest.json", report)
    return report
