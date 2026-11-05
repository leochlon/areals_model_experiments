"""Field encoder: a geometry checkpoint with a per-point field channel, trained by masked completion of the field."""

from __future__ import annotations

import copy
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
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
from .model import GeometryAutoencoder, GeometryEncoder, LatentGeometry, packet_extras

CLOUD_KEYS = ("points", "normals", "point_weights", "anchors", "anchor_normals", "anchor_weights", "center", "scale",
              "area", "surfels", "surfel_normals", "surfel_weights")
PRIORS = ("frozen", "finetune", "probe")
NORMALIZATIONS = ("dataset", "shape")


def widen(encoder: GeometryEncoder) -> None:
    """Two columns join the encoder's first layer, a field value and a presence flag, with zero weights, so the
    widened encoder starts as the geometry encoder."""
    first = encoder.embed[0]
    widened = nn.Linear(first.in_features + 2, first.out_features)
    with torch.no_grad():
        widened.weight.zero_()
        widened.weight[:, : first.in_features] = first.weight
        widened.bias.copy_(first.bias)
    encoder.embed[0] = widened


def encode_field(encoder: GeometryEncoder, batch: dict, values, present, anchor_values, anchor_present):
    """A widened encoder's packet: every point and anchor carries its value (zeroed where absent) and presence."""
    dtype = batch["normals"].dtype

    def embed(positions, normals, value, flag):
        return encoder.embed(torch.cat([encoder.position(positions), normals,
                                        torch.cat([value * flag, flag], dim=-1).to(dtype)], dim=-1))

    points = embed(batch["points"], batch["normals"], values, present)
    anchors = embed(batch["anchors"], batch["anchor_normals"], anchor_values, anchor_present)
    return LatentGeometry(encoder.tokens(points, anchors, batch), batch["anchors"], batch["anchor_normals"],
                          batch["anchor_weights"], batch["center"], batch["scale"], batch["area"],
                          **packet_extras(batch))


class FieldEncoder(nn.Module):
    """A geometry checkpoint as a shape-and-field encoder. A linear head on decoder features predicts the normalized
    field at any point; with no values present, it predicts the field from geometry alone. `prior` sets how the
    geometry model is used:

    frozen: the geometry model stays fixed. A trainable copy of its encoder takes two more input columns, a field
        value and a presence flag, and its tokens are added to the fixed tokens through a zero-initialized layer; a
        trainable copy of the decoder reads the sum. Training starts exactly from the geometry packet and the
        geometry decoder's features, and the geometry model never changes.
    finetune: the two columns join the geometry encoder itself and every weight trains.
    probe: only the head trains, on the fixed decoder's features, and field values are ignored. A baseline."""

    def __init__(self, base: GeometryAutoencoder, prior: str = "finetune"):
        super().__init__()
        if prior not in PRIORS:
            raise ValueError(f"prior must be one of {PRIORS}")
        self.base, self.config, self.prior = base, base.config, prior
        if prior == "frozen":
            self.branch = copy.deepcopy(base.encoder).requires_grad_(True)
            widen(self.branch)
            self.join = nn.Linear(base.config.width, base.config.width)
            nn.init.zeros_(self.join.weight)
            nn.init.zeros_(self.join.bias)
            self.field_decoder = copy.deepcopy(base.decoder).requires_grad_(True)
        if prior == "finetune":
            widen(base.encoder)
        else:
            base.requires_grad_(False)
        self.head = nn.Linear(base.config.width, 1)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.prior != "finetune":
            self.base.eval()  # a fixed prior
        return self

    def new_parameters(self) -> list[nn.Parameter]:
        """The weights with no pretrained values: the head, the join and the widened first layer."""
        modules = [self.head]
        if self.prior == "finetune":
            modules.append(self.base.encoder.embed[0])
        elif self.prior == "frozen":
            modules += [self.join, self.branch.embed[0]]
        return [parameter for module in modules for parameter in module.parameters()]

    def encode(self, batch: dict, values, present, anchor_values, anchor_present) -> LatentGeometry:
        if self.prior == "finetune":
            return encode_field(self.base.encoder, batch, values, present, anchor_values, anchor_present)
        with torch.no_grad():
            geometry = self.base.encoder(batch)
        if self.prior == "probe":
            return geometry
        field = encode_field(self.branch, batch, values, present, anchor_values, anchor_present)
        return geometry.with_tokens(geometry.tokens + self.join(field.tokens))

    def predict(self, geometry: LatentGeometry, world_queries: torch.Tensor) -> torch.Tensor:
        if self.prior == "frozen":
            complete = geometry.complete
            if complete is None:
                complete = geometry.tokens.new_ones(len(geometry.tokens), 1)
            features = self.field_decoder(geometry, world_queries, complete.to(geometry.tokens.dtype))
        else:
            with torch.no_grad() if self.prior == "probe" else nullcontext():
                features = self.base.query_features(geometry, world_queries)
        return self.head(features).float().squeeze(-1)


def load_field_encoder(path: str | Path, device: torch.device | str = "cpu"):
    """The field encoder (in eval mode), its geometry configuration and the checkpoint state. Predictions are in
    normalized units; state["normalization"] says which (mode "dataset": value = prediction x std + mean)."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("kind") != "field_encoder":
        raise ValueError(f"{path} is not a field encoder checkpoint")
    config = ExperimentConfig.from_dict(state["config"])
    model = FieldEncoder(GeometryAutoencoder(config.model), state.get("settings", {}).get("prior", "finetune"))
    model.load_state_dict(state["model"])
    state.setdefault("normalization", dict(mode="shape"))  # checkpoints from before the option
    return model.to(device).eval(), config, state


@dataclass(frozen=True)
class FieldTrainConfig:
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-4  # pretrained weights
    new_learning_rate: float = 1e-3  # the new input columns, the join and the head
    weight_decay: float = 0.05
    hidden_range: tuple[float, float] = (0.0, 1.0)
    full_hidden_probability: float = 0.15
    mask_patches: int = 64
    queries: int = 2048
    visible_weight: float = 0.25
    sign_flip: bool = True
    ratio_range: tuple[float, float] = (0.5, 2.0)
    eval_hidden: tuple[float, ...] = (0.6, 1.0)  # the first picks field_best.pt
    workers: int = 4
    seed: int = 0
    amp: str = "bf16"
    # how the geometry model is used: "frozen" (a fixed prior), "finetune", or "probe" (a head on fixed features)
    prior: str = "frozen"
    # "dataset": by the training set's mean and deviation; "shape": each field by its own, hidden part included
    normalize: str = "dataset"

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.mask_patches, self.queries) < 1:
            raise ValueError("epochs, batch_size, mask_patches and queries must be positive")
        if not 0 <= self.hidden_range[0] <= self.hidden_range[1] <= 1 or not 0 <= self.full_hidden_probability <= 1:
            raise ValueError("hidden fractions must lie in [0, 1]")
        if not self.eval_hidden or not all(0 < fraction <= 1 for fraction in self.eval_hidden):
            raise ValueError("eval_hidden needs fractions in (0, 1]")
        if not 0 < self.ratio_range[0] <= self.ratio_range[1]:
            raise ValueError("ratio_range must be positive and ordered")
        if self.amp not in ("none", "bf16", "fp16"):
            raise ValueError("amp must be none, bf16 or fp16")
        if self.prior not in PRIORS or self.normalize not in NORMALIZATIONS:
            raise ValueError(f"prior must be one of {PRIORS} and normalize one of {NORMALIZATIONS}")


def load_field_file(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as packed:
        missing = {"points", "values", "weights"} - set(packed.files)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}")
        points, values, weights = (np.asarray(packed[k], dtype=np.float32) for k in ("points", "values", "weights"))
    values, weights = values.reshape(-1), weights.reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or not len(values) == len(weights) == len(points):
        raise ValueError(f"{path}: points [N,3], values [N] and weights [N] must agree")
    return dict(points=points, values=values, weights=weights)


IDENTITY = dict(sign=1.0, positive=1.0, negative=1.0)


def random_condition(rng: np.random.Generator, config: FieldTrainConfig) -> dict:
    """A random sign convention and independent scales for the field's positive and negative parts, so no single
    convention is learned."""
    sign = -1.0 if config.sign_flip and rng.random() < 0.5 else 1.0
    low, high = np.log(config.ratio_range)
    return dict(sign=sign, positive=float(np.exp(rng.uniform(low, high))),
                negative=float(np.exp(rng.uniform(low, high))))


def apply_condition(values: np.ndarray, condition: dict) -> np.ndarray:
    scaled = condition["positive"] * np.maximum(values, 0) + condition["negative"] * np.minimum(values, 0)
    return (condition["sign"] * scaled).astype(np.float32)


class FieldDataset(Dataset):
    """Each record's prepared cloud (complete input) with the field value of the nearest field point at every sample
    and anchor; masks, conditions and queries are drawn per record and epoch. `normalization` is a
    `field_statistics` result; without one, each field is standardized by its own mean and deviation."""

    def __init__(self, records: list[dict], fields: str | Path, model_config, data_config, config: FieldTrainConfig,
                 normalization: dict | None = None):
        self.records, self.config, self.epoch = records, config, 0
        self.normalization = normalization or dict(mode="shape")
        self.prepared = [self._prepare(record, Path(fields) / f"{record['id']}.npz", model_config, data_config)
                         for record in records]

    @staticmethod
    def _prepare(record, field_path, model_config, data_config) -> dict:
        cloud, _ = load_cloud_file(record["path"])
        per = model_config.surfels_per_token
        prepared = prepare_cloud(cloud, model_config.latents, data_config.encoder_points, np.random.default_rng(0),
                                 model_config.latents * per if per > 1 else 0)
        field = load_field_file(field_path)
        tree = cKDTree(field["points"])
        world = prepared["points"] * prepared["scale"] + prepared["center"]
        prepared["world"] = world.astype(np.float32)
        prepared["values"] = field["values"][tree.query(world, workers=1)[1]]
        prepared["anchor_values"] = field["values"][
            tree.query(prepared["anchors"] * prepared["scale"] + prepared["center"], workers=1)[1]]
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
        if self.normalization["mode"] == "dataset":
            mean, std = self.normalization["mean"], self.normalization["std"]
        else:  # the whole field's statistics, hidden part included
            mean = float((values * weights).sum() / weights.sum())
            std = float(np.sqrt(((values - mean) ** 2 * weights).sum() / weights.sum())) + 1e-6
        values, anchor_values = (values - mean) / std, (anchor_values - mean) / std
        if hidden_fraction >= 1.0:
            present = np.zeros(len(values), np.float32)
        else:
            cloud = Cloud(p["points"].astype(np.float64), p["normals"].astype(np.float64), weights.astype(np.float64))
            present = visible_mask(cloud, hidden_fraction, config.mask_patches, rng).astype(np.float32)
        chosen = (np.arange(len(values)) if all_queries
                  else rng.choice(len(values), config.queries, replace=True, p=weights / weights.sum()))
        record = self.records[index]
        return dict({key: p[key] for key in CLOUD_KEYS if key in p}, id=record["id"], family=record["family"],
                    split=record["split"], values=values[:, None], present=present[:, None],
                    anchor_values=anchor_values[:, None], anchor_present=present[p["anchor_index"]][:, None],
                    queries=p["world"][chosen], target=values[chosen], hidden=1.0 - present[chosen],
                    query_weights=(weights[chosen] / weights[chosen].sum()).astype(np.float32))

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng(stable_seed(self.config.seed, self.records[index]["id"] + ":field", self.epoch))
        low, high = self.config.hidden_range
        hidden = 1.0 if rng.random() < self.config.full_hidden_probability else float(rng.uniform(low, high))
        return self.sample(index, hidden, random_condition(rng, self.config) if hidden < 1.0 else IDENTITY, rng)


def field_statistics(dataset: FieldDataset) -> dict:
    """The raw field's mean and standard deviation over a dataset, each shape weighted equally and each point by its
    area: the normalization for `normalize="dataset"`."""
    fields = [(p["values"].astype(np.float64), p["point_weights"].astype(np.float64)) for p in dataset.prepared]
    mean = float(np.mean([(values * weights).sum() / weights.sum() for values, weights in fields]))
    std = float(np.sqrt(np.mean([((values - mean) ** 2 * weights).sum() / weights.sum() for values, weights in fields])))
    if not std > 0:
        raise ValueError("the field is constant over the training set")
    return dict(mode="dataset", mean=mean, std=std)


def collate_field(samples: list[dict]) -> dict:
    batch = collate_clouds([{key: s[key] for key in CLOUD_KEYS + ("id", "family", "split") if key in s}
                            for s in samples])
    width = batch["points"].size(1)
    for key in ("values", "present"):
        batch[key] = torch.from_numpy(np.stack([np.pad(s[key], ((0, width - len(s[key])), (0, 0))) for s in samples]))
    for key in ("anchor_values", "anchor_present", "queries", "target", "hidden", "query_weights"):
        batch[key] = torch.from_numpy(np.stack([s[key] for s in samples]))
    return batch


def field_loss(model: FieldEncoder, batch: dict, visible_weight: float) -> torch.Tensor:
    geometry = model.encode(batch, batch["values"], batch["present"], batch["anchor_values"], batch["anchor_present"])
    error = (model.predict(geometry, batch["queries"]) - batch["target"]) ** 2
    weight = (batch["hidden"] + visible_weight * (1 - batch["hidden"])) * batch["query_weights"]
    return ((error * weight).sum(-1) / weight.sum(-1).clamp_min(1e-12)).mean()


@torch.no_grad()
def evaluate_field(model: FieldEncoder, dataset: FieldDataset, hidden_fraction: float, device, batch_size: int,
                   amp) -> dict:
    """RMSE at hidden points in units of each shape's field deviation (1.0 is about its mean), and the same for
    copying the nearest visible value. With dataset normalization, also the RMSE in the field's own units."""
    model.eval()
    scores, copies, errors = [], [], []
    dataset_units = dataset.normalization["mode"] == "dataset"
    for start in range(0, len(dataset), batch_size):
        samples = []
        for index in range(start, min(start + batch_size, len(dataset))):
            rng = np.random.default_rng(stable_seed(dataset.config.seed, dataset.records[index]["id"] + ":eval", 0))
            condition = random_condition(rng, dataset.config) if hidden_fraction < 1.0 else IDENTITY
            samples.append(dataset.sample(index, hidden_fraction, condition, rng, all_queries=True))
        batch = to_device(collate_field(samples), device)
        with amp():
            geometry = model.encode(batch, batch["values"], batch["present"], batch["anchor_values"],
                                    batch["anchor_present"])
            prediction = model.predict(geometry, batch["queries"]).float().cpu().numpy()
        for b, sample in enumerate(samples):
            hidden = sample["hidden"] > 0.5
            if not hidden.any():
                continue
            target, spread = sample["target"], 1.0  # "shape" targets are standardized already
            if dataset_units:
                everywhere = sample["query_weights"]
                spread = max(float(np.sqrt((everywhere * (target - (everywhere * target).sum()) ** 2).sum())), 1e-6)
            weights = sample["query_weights"][hidden] / sample["query_weights"][hidden].sum()
            error = float(np.sqrt((weights * (prediction[b, hidden] - target[hidden]) ** 2).sum()))
            scores.append(error / spread)
            errors.append(error * dataset.normalization.get("std", 1.0))
            if (~hidden).any():
                nearest = cKDTree(sample["queries"][~hidden]).query(sample["queries"][hidden], workers=1)[1]
                copied = target[~hidden][nearest]
                copies.append(float(np.sqrt((weights * (copied - target[hidden]) ** 2).sum())) / spread)
    model.train()
    result = dict(nrmse=float(np.mean(scores)), copy_nrmse=float(np.mean(copies)) if copies else None)
    if dataset_units:
        result["rmse"] = float(np.mean(errors))
    return result


def train_field(manifest, fields, checkpoint, out, config: FieldTrainConfig | None = None,
                device: torch.device | str = "cpu") -> dict:
    """Train a field encoder from a geometry checkpoint on the fields in `fields` (one <id>.npz per record with
    points, values and weights). Writes field_best.pt (best validation score at the first eval_hidden fraction),
    field_last.pt and history.json. A probe ignores field values, so it trains and validates with the whole field
    hidden."""
    config = config or FieldTrainConfig()
    if config.prior == "probe":
        config = replace(config, full_hidden_probability=1.0, eval_hidden=(1.0,))
    device = torch.device(device)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    base, experiment, state = load_model(checkpoint)
    records, _ = read_manifest(manifest)
    train_records = [r for r in records if r["split"] == "train"]
    val_records = [r for r in records if r["split"] == "val"] or train_records[:8]
    train_data = FieldDataset(train_records, fields, experiment.model, experiment.data, config)
    if config.normalize == "dataset":
        train_data.normalization = field_statistics(train_data)
    val_data = FieldDataset(val_records, fields, experiment.model, experiment.data, config, train_data.normalization)
    model = FieldEncoder(base, config.prior).to(device)
    new = model.new_parameters()
    new_ids = {id(p) for p in new}
    pretrained = [p for p in model.parameters() if p.requires_grad and id(p) not in new_ids]
    groups = [dict(params=pretrained, lr=config.learning_rate)] if pretrained else []
    groups.append(dict(params=new, lr=config.new_learning_rate))
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    steps = config.epochs * ((len(train_data) + config.batch_size - 1) // config.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[group["lr"] for group in groups],
                                                    total_steps=steps, pct_start=0.1, anneal_strategy="cos")
    use_amp = config.amp != "none" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if config.amp == "bf16" else torch.float16
    amp = (lambda: torch.autocast(device.type, dtype=amp_dtype)) if use_amp else nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and config.amp == "fp16")
    history, best, started = [], float("inf"), time.time()
    extra = dict(kind="field_encoder", config=state["config"], base_checkpoint=str(checkpoint),
                 settings=asdict(config), normalization=train_data.normalization)
    for epoch in range(1, config.epochs + 1):
        train_data.set_epoch(epoch)
        losses = []
        for batch in DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.workers,
                                collate_fn=collate_field):
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
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)), seconds=round(time.time() - started),
                   validation={str(k): v for k, v in validation.items()})
        history.append(row)
        score = validation[config.eval_hidden[0]]["nrmse"]
        if score < best:
            best = score
            save_checkpoint(out / "field_best.pt", dict(extra, model=model.state_dict(), epoch=epoch))
        save_checkpoint(out / "field_last.pt", dict(extra, model=model.state_dict(), epoch=epoch))
        write_json(out / "history.json", history)
        shown = " ".join(f"hidden {k:.0%}: {v['nrmse']:.3f}" for k, v in validation.items())
        print(f"epoch {epoch}/{config.epochs} loss {row['train_loss']:.4f} | val nRMSE {shown} | {row['seconds']}s",
              flush=True)
    return dict(output=str(out), prior=config.prior, best_nrmse=best, epochs=config.epochs,
                history=str(out / "history.json"))
