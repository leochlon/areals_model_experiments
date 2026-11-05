"""A shape-and-field encoder: the geometry encoder with a per-point field channel, trained by masked completion."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig
from .data import collate_clouds, load_cloud_file, read_manifest, stable_seed, to_device
from .geometry import Cloud, prepare_cloud, visible_mask
from .io import load_model, save_checkpoint, write_json
from .model import GeometryAutoencoder, LatentGeometry


class FieldEncoder(nn.Module):
    """Two input columns are added to the encoder's first layer, a standardised field value and a presence flag,
    both zero-initialised so the model starts as the geometry encoder it wraps. A linear head on the decoder's query
    features predicts the standardised field at any world coordinate. A car whose flag is zero everywhere is encoded
    from geometry alone."""

    def __init__(self, base: GeometryAutoencoder):
        super().__init__()
        self.base = base
        first = base.encoder.embed[0]
        widened = nn.Linear(first.in_features + 2, first.out_features)
        with torch.no_grad():
            widened.weight.zero_()
            widened.weight[:, : first.in_features] = first.weight
            widened.bias.copy_(first.bias)
        base.encoder.embed[0] = widened
        self.head = nn.Linear(base.config.width, 1)
        self.config = base.config

    @staticmethod
    def _channels(values: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        return torch.cat([values * present, present], dim=-1)

    def encode(self, batch: dict, values, present, anchor_values, anchor_present) -> LatentGeometry:
        encoder = self.base.encoder
        dtype = batch["normals"].dtype
        points = encoder.embed(
            torch.cat(
                [encoder.position(batch["points"]), batch["normals"], self._channels(values, present).to(dtype)],
                dim=-1,
            )
        )
        initial = encoder.embed(
            torch.cat(
                [
                    encoder.position(batch["anchors"]),
                    batch["anchor_normals"],
                    self._channels(anchor_values, anchor_present).to(dtype),
                ],
                dim=-1,
            )
        )
        initial = initial + encoder.latent_token
        tokens = encoder.read(initial, points, batch["anchors"], batch["points"], batch["point_weights"])
        for block in encoder.blocks:
            tokens = block(tokens, tokens, batch["anchors"], batch["anchors"], batch["anchor_weights"])
        return LatentGeometry(
            tokens,
            batch["anchors"],
            batch["anchor_normals"],
            batch["anchor_weights"],
            batch["center"],
            batch["scale"],
            batch["area"],
            initial,
            batch.get("surfels"),
            batch.get("surfel_normals"),
            batch.get("surfel_weights"),
            batch.get("complete"),
        )

    def features(self, geometry: LatentGeometry, world_queries: torch.Tensor) -> torch.Tensor:
        return self.base.query_features(geometry, world_queries)

    def predict(self, geometry: LatentGeometry, world_queries: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(geometry, world_queries)).float().squeeze(-1)


def save_field_encoder(path: str | Path, model: FieldEncoder, config: dict, **extra) -> None:
    save_checkpoint(path, dict(version=1, kind="field_encoder", model=model.state_dict(), config=config, **extra))


def load_field_encoder(path: str | Path, device: torch.device | str = "cpu"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("kind") != "field_encoder" or state.get("version") != 1:
        raise ValueError("Not a field encoder checkpoint")
    config = ExperimentConfig.from_dict(state["config"])
    model = FieldEncoder(GeometryAutoencoder(config.model))
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval(), config, state


@dataclass(frozen=True)
class FieldTrainConfig:
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-4
    new_learning_rate: float = 1e-3
    weight_decay: float = 0.05
    hidden_range: tuple[float, float] = (0.0, 1.0)
    full_hidden_probability: float = 0.15
    mask_patches: int = 64
    queries: int = 2048
    visible_weight: float = 0.25
    sign_flip: bool = True
    ratio_range: tuple[float, float] = (0.5, 2.0)
    eval_hidden: tuple[float, ...] = (0.6, 1.0)
    workers: int = 4
    seed: int = 0
    amp: str = "bf16"

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.mask_patches, self.queries) < 1:
            raise ValueError("epochs, batch_size, mask_patches and queries must be positive")
        if min(self.learning_rate, self.new_learning_rate) <= 0 or min(self.weight_decay, self.visible_weight) < 0:
            raise ValueError("Invalid optimization scales")
        if not 0 <= self.full_hidden_probability <= 1 or self.workers < 0:
            raise ValueError("full_hidden_probability must lie within [0, 1] and workers be nonnegative")
        if not 0 <= self.hidden_range[0] <= self.hidden_range[1] <= 1:
            raise ValueError("hidden_range must lie within [0, 1]")
        if not 0 < self.ratio_range[0] <= self.ratio_range[1]:
            raise ValueError("ratio_range must be positive and ordered")
        if self.amp not in ("none", "bf16", "fp16"):
            raise ValueError("amp must be none, bf16, or fp16")


def load_field_file(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as packed:
        missing = {"points", "values", "weights"} - set(packed.files)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}")
        arrays = {key: np.asarray(packed[key], dtype=np.float32) for key in ("points", "values", "weights")}
    points, values, weights = arrays["points"], arrays["values"].reshape(-1), arrays["weights"].reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or len(values) != len(points) or len(weights) != len(points):
        raise ValueError(f"{path}: points [N,3], values [N] and weights [N] must agree")
    if not all(np.isfinite(a).all() for a in (points, values, weights)):
        raise ValueError(f"{path}: nonfinite field data")
    return dict(points=points, values=values, weights=weights)


def random_condition(rng: np.random.Generator, config: FieldTrainConfig) -> dict:
    """A sign convention and independent scales on the positive and negative parts of the field."""
    sign = -1.0 if config.sign_flip and rng.random() < 0.5 else 1.0
    low, high = np.log(config.ratio_range)
    return dict(sign=sign, positive=float(np.exp(rng.uniform(low, high))), negative=float(np.exp(rng.uniform(low, high))))


def apply_condition(values: np.ndarray, condition: dict) -> np.ndarray:
    scaled = condition["positive"] * np.maximum(values, 0) + condition["negative"] * np.minimum(values, 0)
    return (condition["sign"] * scaled).astype(np.float32)


IDENTITY = dict(sign=1.0, positive=1.0, negative=1.0)


class FieldDataset(Dataset):
    """One prepared cloud per record with the field of the nearest mesh vertex at every sample point and anchor.
    Masks, conditions and query points are drawn per record and epoch."""

    def __init__(self, records: list[dict], fields: str | Path, model_config, data_config, config: FieldTrainConfig):
        if not records:
            raise ValueError("Dataset split is empty")
        self.records, self.config, self.epoch = records, config, 0
        self.prepared = [self._prepare(record, Path(fields) / f"{record['id']}.npz", model_config, data_config)
                         for record in records]

    @staticmethod
    def _prepare(record: dict, field_path: Path, model_config, data_config) -> dict:
        cloud, _ = load_cloud_file(record["path"])
        surfel_count = model_config.latents * model_config.surfels_per_token if model_config.surfels_per_token > 1 else 0
        prepared = prepare_cloud(
            cloud, model_config.latents, data_config.encoder_points, np.random.default_rng(0), surfel_count
        )
        field = load_field_file(field_path)
        tree = cKDTree(field["points"])
        world = prepared["points"] * prepared["scale"] + prepared["center"]
        anchors = prepared["anchors"] * prepared["scale"] + prepared["center"]
        prepared["world"] = world.astype(np.float32)
        prepared["values"] = field["values"][tree.query(world, workers=1)[1]]
        prepared["anchor_values"] = field["values"][tree.query(anchors, workers=1)[1]]
        prepared["anchor_index"] = cKDTree(prepared["points"]).query(prepared["anchors"], workers=1)[1]
        return prepared

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def sample(self, index: int, hidden_fraction: float, condition: dict, rng, all_queries: bool = False) -> dict:
        p, config = self.prepared[index], self.config
        values, anchor_values = apply_condition(p["values"], condition), apply_condition(p["anchor_values"], condition)
        weights = p["point_weights"]
        mean = float((values * weights).sum() / weights.sum())
        std = float(np.sqrt(((values - mean) ** 2 * weights).sum() / weights.sum())) + 1e-6
        values, anchor_values = (values - mean) / std, (anchor_values - mean) / std
        if hidden_fraction >= 1.0:
            present = np.zeros(len(values), np.float32)
        else:
            cloud = Cloud(p["points"].astype(np.float64), p["normals"].astype(np.float64), weights.astype(np.float64))
            present = visible_mask(cloud, hidden_fraction, config.mask_patches, rng).astype(np.float32)
        if all_queries:
            chosen = np.arange(len(values))
        else:
            chosen = rng.choice(len(values), config.queries, replace=True, p=weights / weights.sum())
        sample = {key: p[key] for key in ("points", "normals", "point_weights", "anchors", "anchor_normals",
                                          "anchor_weights", "center", "scale", "area", "surfels", "surfel_normals",
                                          "surfel_weights") if key in p}
        record = self.records[index]
        sample.update(
            id=record["id"], family=record["family"], split=record["split"],
            values=values[:, None], present=present[:, None],
            anchor_values=anchor_values[:, None], anchor_present=present[p["anchor_index"]][:, None],
            queries=p["world"][chosen], target=values[chosen], hidden=1.0 - present[chosen],
            query_weights=(weights[chosen] / weights[chosen].sum()).astype(np.float32),
        )
        return sample

    def __getitem__(self, index: int) -> dict:
        config = self.config
        rng = np.random.default_rng(stable_seed(config.seed, self.records[index]["id"] + ":field", self.epoch))
        low, high = config.hidden_range
        hidden = 1.0 if rng.random() < config.full_hidden_probability else float(rng.uniform(low, high))
        condition = random_condition(rng, config) if hidden < 1.0 else IDENTITY
        return self.sample(index, hidden, condition, rng)


def collate_field(samples: list[dict]) -> dict:
    cloud_keys = ("id", "family", "split", "points", "normals", "point_weights", "anchors", "anchor_normals",
                  "anchor_weights", "center", "scale", "area", "surfels", "surfel_normals", "surfel_weights")
    batch = collate_clouds([{key: sample[key] for key in cloud_keys if key in sample} for sample in samples])
    n_points = batch["points"].size(1)
    for key in ("values", "present"):
        batch[key] = torch.from_numpy(np.stack([np.pad(s[key], ((0, n_points - len(s[key])), (0, 0))) for s in samples]))
    for key in ("anchor_values", "anchor_present", "queries", "target", "hidden", "query_weights"):
        batch[key] = torch.from_numpy(np.stack([sample[key] for sample in samples]))
    return batch


def field_loss(model: FieldEncoder, batch: dict, visible_weight: float) -> torch.Tensor:
    geometry = model.encode(batch, batch["values"], batch["present"], batch["anchor_values"], batch["anchor_present"])
    error = (model.predict(geometry, batch["queries"]) - batch["target"]) ** 2
    weight = (batch["hidden"] + visible_weight * (1 - batch["hidden"])) * batch["query_weights"]
    return ((error * weight).sum(-1) / weight.sum(-1).clamp_min(1e-12)).mean()


@torch.no_grad()
def evaluate_field(model: FieldEncoder, dataset: FieldDataset, hidden_fraction: float, device, batch_size: int, amp) -> dict:
    """RMSE at hidden points in each car's own standardised units (1.0 is the mean predictor), and the nearest visible
    value as a reference. Fully hidden cars are scored in the canonical convention."""
    model.eval()
    scores, copies = [], []
    for start in range(0, len(dataset), batch_size):
        samples = []
        for index in range(start, min(start + batch_size, len(dataset))):
            rng = np.random.default_rng(stable_seed(dataset.config.seed, dataset.records[index]["id"] + ":eval", 0))
            condition = random_condition(rng, dataset.config) if hidden_fraction < 1.0 else IDENTITY
            samples.append(dataset.sample(index, hidden_fraction, condition, rng, all_queries=True))
        batch = to_device(collate_field(samples), device)
        with amp():
            geometry = model.encode(batch, batch["values"], batch["present"], batch["anchor_values"], batch["anchor_present"])
            prediction = model.predict(geometry, batch["queries"]).float().cpu().numpy()
        for b, sample in enumerate(samples):
            hidden = sample["hidden"] > 0.5
            if not hidden.any():
                continue
            weights = sample["query_weights"][hidden] / sample["query_weights"][hidden].sum()
            scores.append(float(np.sqrt((weights * (prediction[b, hidden] - sample["target"][hidden]) ** 2).sum())))
            if (~hidden).any():
                nearest = cKDTree(sample["queries"][~hidden]).query(sample["queries"][hidden], workers=1)[1]
                copy = sample["target"][~hidden][nearest]
                copies.append(float(np.sqrt((weights * (copy - sample["target"][hidden]) ** 2).sum())))
    model.train()
    return dict(nrmse=float(np.mean(scores)), copy_nrmse=float(np.mean(copies)) if copies else None)


def train_field(manifest: str | Path, fields: str | Path, checkpoint: str | Path, out: str | Path,
                config: FieldTrainConfig = FieldTrainConfig(), device: torch.device | str = "cpu") -> dict:
    """Fine-tune a geometry checkpoint into a field encoder by masked completion of the field in `fields`
    (`<id>.npz` per manifest record with points, values and weights). Writes field_best.pt, field_last.pt and
    history.json to `out`; the best epoch is chosen on the validation split at the first eval_hidden fraction."""
    device = torch.device(device)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    base, experiment, state = load_model(checkpoint, "cpu")
    records, _ = read_manifest(manifest)
    by_split = {split: [r for r in records if r["split"] == split] for split in ("train", "val")}
    train_data = FieldDataset(by_split["train"], fields, experiment.model, experiment.data, config)
    val_data = FieldDataset(by_split["val"] or by_split["train"][:8], fields, experiment.model, experiment.data, config)

    model = FieldEncoder(base).to(device)
    new = [model.head.weight, model.head.bias, model.base.encoder.embed[0].weight, model.base.encoder.embed[0].bias]
    new_ids = {id(p) for p in new}
    old = [p for p in model.parameters() if id(p) not in new_ids]
    optimizer = torch.optim.AdamW(
        [dict(params=old, lr=config.learning_rate), dict(params=new, lr=config.new_learning_rate)],
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = (len(train_data) + config.batch_size - 1) // config.batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[config.learning_rate, config.new_learning_rate],
        total_steps=config.epochs * steps_per_epoch, pct_start=0.1, anneal_strategy="cos",
    )
    use_amp = config.amp != "none" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if config.amp == "bf16" else torch.float16
    amp = (lambda: torch.autocast(device.type, dtype=amp_dtype)) if use_amp else nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and config.amp == "fp16")

    history, best, started = [], float("inf"), time.time()
    extra = dict(config=state["config"], base_checkpoint=str(checkpoint), settings=asdict(config))
    for epoch in range(1, config.epochs + 1):
        train_data.set_epoch(epoch)
        loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.workers,
                            collate_fn=collate_field)
        losses = []
        for batch in loader:
            batch = to_device(batch, device)
            with amp():
                loss = field_loss(model, batch, config.visible_weight)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach()))
        validation = {fraction: evaluate_field(model, val_data, fraction, device, config.batch_size, amp)
                      for fraction in config.eval_hidden}
        record = dict(epoch=epoch, train_loss=float(np.mean(losses)), seconds=round(time.time() - started),
                      validation={str(k): v for k, v in validation.items()})
        history.append(record)
        score = validation[config.eval_hidden[0]]["nrmse"]
        if score < best:
            best = score
            save_field_encoder(out / "field_best.pt", model, epoch=epoch, **extra)
        save_field_encoder(out / "field_last.pt", model, epoch=epoch, **extra)
        write_json(out / "history.json", history)
        shown = " ".join(f"hidden {k:.0%}: {v['nrmse']:.3f}" for k, v in validation.items())
        print(f"epoch {epoch}/{config.epochs} loss {record['train_loss']:.4f} | val nRMSE {shown} | {record['seconds']}s",
              flush=True)
    return dict(output=str(out), best_nrmse=best, epochs=config.epochs, history=str(out / "history.json"))
