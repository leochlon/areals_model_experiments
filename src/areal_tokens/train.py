"""Reproducible epoch-boundary training with AMP, accumulation and optional DDP."""

from __future__ import annotations

import json
import math
import os
import random
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .config import ExperimentConfig
from .data import GeometryDataset, collate_clouds, read_manifest, stable_seed, to_device
from .evaluate import evaluate_model
from .io import CHECKPOINT_VERSION, environment, save_checkpoint, write_json
from .losses import reconstruction_loss
from .model import GeometryAutoencoder
from .validation import load_audit, validate_manifest


def resolve_device(request: str = "auto") -> torch.device:
    if request == "auto":
        request = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(request)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported training devices are cpu and cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng() -> dict:
    state = np.random.get_state()
    return dict(
        python=random.getstate(),
        numpy=[state[0], state[1].tolist(), *state[2:]],
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.array(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def lr_at_step(config, step: int, steps_per_epoch: int) -> float:
    total = config.epochs * steps_per_epoch
    warmup = config.warmup_steps if config.warmup_steps > 0 else config.warmup_epochs * steps_per_epoch
    warmup = min(warmup, total)
    if step < warmup:
        factor = (step + 1) / warmup
    else:
        fraction = (step - warmup) / max(1, total - warmup - 1)
        fraction = min(max(fraction, 0), 1)
        factor = config.min_lr_ratio + (1 - config.min_lr_ratio) * (1 + math.cos(math.pi * fraction)) / 2
    return config.learning_rate * factor


def optimizer_for(model, config):
    decay, no_decay = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [dict(params=decay, weight_decay=config.weight_decay), dict(params=no_decay, weight_decay=0.0)],
        lr=config.learning_rate,
    )


def _rank_zero_call(function, rank, world):
    """Propagate rank-zero I/O/validation errors rather than leaving peers waiting."""
    result = [None]
    if rank == 0:
        try:
            result[0] = dict(ok=True, value=function())
        except Exception as error:
            result[0] = dict(ok=False, error=f"{type(error).__name__}: {error}")
    if world > 1:
        dist.broadcast_object_list(result, src=0)
    if not result[0]["ok"]:
        raise RuntimeError(result[0]["error"])
    return result[0]["value"]


def _require_finite(value, device, world, message):
    flag = torch.as_tensor(int(bool(torch.isfinite(value).all())), device=device, dtype=torch.int32)
    if world > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not flag.item():
        raise FloatingPointError(message)


def _write_history(output, history):
    temporary = output / "metrics.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in history))
    temporary.replace(output / "metrics.jsonl")


def _commit_epoch(output, state, improved, validation):
    # last.pt is the authoritative recovery point. Its history repairs a
    # partially written metrics log after an interrupted write.
    if improved:
        save_checkpoint(output / "best.pt", state)
    save_checkpoint(output / "last.pt", state)
    _write_history(output, state["history"])
    write_json(output / "validation.latest.json", validation)
    print(json.dumps(state["history"][-1], allow_nan=False), flush=True)


def train(
    manifest,
    config: ExperimentConfig,
    output,
    *,
    device="auto",
    resume=None,
    stop_after_epochs=None,
    deterministic=False,
    audit=None,
) -> dict:
    """Resume only from last.pt in the same run directory, at epoch boundaries.

    stop_after_epochs is the absolute completed-epoch limit for this invocation;
    it does not change the planned learning-rate schedule. Exact replay requires
    the same device/backend/software, world size, configuration and data content.
    audit is an optional report written by `validate --out`; for this exact
    manifest it replaces rereading and hashing every file at startup.
    """
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = resolve_device(device)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic)
    if device.type == "cuda":
        if world > 1:
            device = torch.device("cuda", local_rank)
        elif device.index is None:  # "cuda" without an index is rejected by torch.cuda.set_device
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    owns_group = world > 1 and not dist.is_initialized()
    if owns_group:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    try:
        return _train(
            manifest,
            config,
            Path(output).resolve(),
            device,
            resume,
            stop_after_epochs,
            deterministic,
            rank,
            world,
            audit,
        )
    finally:
        if owns_group:
            dist.destroy_process_group()


def _train(manifest, config, output, device, resume, stop_after_epochs, deterministic, rank, world, audit):
    if stop_after_epochs is not None and not 1 <= stop_after_epochs <= config.train.epochs:
        raise ValueError("stop_after_epochs must be within the planned epoch count")
    if config.train.amp == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training requires CUDA; use bf16 or none on CPU")
    if config.train.amp == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bf16; select fp16 or none")
    records, metadata = read_manifest(manifest)
    train_records = [record for record in records if record["split"] == "train"]
    val_records = [record for record in records if record["split"] == "val"]
    train_data = GeometryDataset(train_records, config.model, config.data, config.train.seed, training=True)
    val_data = GeometryDataset(val_records, config.model, config.data, config.train.seed, fixed_queries=True)
    require_targets = config.data.target_mode == "provided"

    def audit_data():
        if audit is not None:
            return load_audit(audit, manifest, require_targets=require_targets)
        return validate_manifest(manifest, require_targets=require_targets, workers=config.train.workers)

    data_audit = _rank_zero_call(audit_data, rank, world)

    def initialize_output():
        output.mkdir(parents=True, exist_ok=True)
        if resume:
            source = Path(resume).resolve()
            if source.parent != output or source.name != "last.pt":
                raise ValueError("Resume requires this run directory's last.pt")
        elif any(output.iterdir()):
            raise FileExistsError(f"New run output must be empty: {output}")
        return None

    _rank_zero_call(initialize_output, rank, world)
    seed_everything(config.train.seed + rank)
    raw_model = GeometryAutoencoder(config.model).to(device)
    optimizer = optimizer_for(raw_model, config.train)
    scaler = torch.amp.GradScaler("cuda", enabled=config.train.amp == "fp16")
    amp_dtype = torch.bfloat16 if config.train.amp == "bf16" else torch.float16
    runtime = dict(
        world_size=world,
        device_type=device.type,
        deterministic=deterministic,
        torch=str(torch.__version__),
        numpy=str(np.__version__),
        threads=torch.get_num_threads(),
        python=environment()["python"],
        cuda_version=torch.version.cuda,
        device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    )
    epoch_start, global_step, best_score, history = 0, 0, math.inf, []
    saved_rng = None
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        # Compare through the dataclasses so a checkpoint written before a
        # configuration key existed still resumes under that key's default.
        if (
            state.get("version") != CHECKPOINT_VERSION
            or ExperimentConfig.from_dict(state["config"]).to_dict() != config.to_dict()
        ):
            raise ValueError("Checkpoint version/configuration differs; resume is not a warm start")
        if state["runtime"] != runtime:
            raise ValueError(
                "Runtime/world size changed; exact epoch-boundary resume requires identical settings"
            )
        raw_model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        epoch_start, global_step = state["epoch"], state["global_step"]
        best_score, history, saved_rng = state["best_score"], state["history"], state["rng"][rank]
    model = (
        DistributedDataParallel(
            raw_model, device_ids=[device.index] if device.type == "cuda" else None, broadcast_buffers=False
        )
        if world > 1
        else raw_model
    )
    if saved_rng is not None:
        restore_rng(saved_rng)
    sampler = DistributedSampler(
        train_data, world, rank, shuffle=True, seed=config.train.seed, drop_last=False
    )
    samples_per_rank = len(sampler)
    batches_per_epoch = math.ceil(samples_per_rank / config.train.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / config.train.accumulation)
    if global_step != epoch_start * steps_per_epoch:
        raise ValueError("Checkpoint optimizer-step count disagrees with epoch boundary")

    def write_metadata():
        write_json(output / "config.json", config.to_dict())
        write_json(output / "environment.json", dict(**environment(), runtime=runtime))
        write_json(output / "data_audit.json", data_audit)
        write_json(output / "manifest.resolved.json", dict(metadata, records=records))
        _write_history(output, history)

    _rank_zero_call(write_metadata, rank, world)
    final_epoch = min(config.train.epochs, stop_after_epochs or config.train.epochs)
    if final_epoch < epoch_start:
        raise ValueError("stop_after_epochs precedes the resumed checkpoint")
    for epoch in range(epoch_start, final_epoch):
        train_data.set_epoch(epoch)
        sampler.set_epoch(epoch)
        loader = DataLoader(
            train_data,
            batch_size=config.train.batch_size,
            sampler=sampler,
            num_workers=config.train.workers,
            collate_fn=collate_clouds,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
            generator=torch.Generator().manual_seed(stable_seed(config.train.seed, "loader", epoch)),
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = torch.zeros(5, dtype=torch.float64, device=device)
        last_lr = 0.0
        gradient_norm = 0.0
        for index, batch in enumerate(loader):
            batch = to_device(batch, device)
            window_start = (index // config.train.accumulation) * config.train.accumulation
            window_end = min(window_start + config.train.accumulation, len(loader))
            window_samples = (
                min(window_end * config.train.batch_size, samples_per_rank)
                - window_start * config.train.batch_size
            )
            is_update = index + 1 == window_end
            if index == window_start:
                last_lr = lr_at_step(config.train, global_step, steps_per_epoch)
                for group in optimizer.param_groups:
                    group["lr"] = last_lr
            synchronization = model.no_sync() if world > 1 and not is_update else nullcontext()
            with synchronization:
                with torch.autocast(device.type, dtype=amp_dtype, enabled=config.train.amp != "none"):
                    prediction = model(batch)
                    losses = reconstruction_loss(prediction, batch, config.train)
                _require_finite(
                    losses["loss"],
                    device,
                    world,
                    f"Nonfinite loss at epoch {epoch + 1}, batch {index}; last checkpoint preserved",
                )
                weight = len(batch["id"]) / window_samples
                scaler.scale(losses["loss"] * weight).backward()
            count = len(batch["id"])
            with torch.no_grad():
                running[:4] += (
                    torch.stack(
                        [
                            losses[key].detach().double()
                            for key in ("loss", "distance", "normal", "normal_fraction")
                        ]
                    )
                    * count
                )
                running[4] += count
            if is_update:
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config.train.gradient_clip)
                _require_finite(
                    norm,
                    device,
                    world,
                    "Nonfinite gradients; reduce learning rate or use bf16/none; last checkpoint preserved",
                )
                gradient_norm = float(norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        if world > 1:
            dist.all_reduce(running)
        train_metrics = dict(
            zip(
                ("loss", "distance", "normal", "normal_fraction"),
                (running[:4] / running[4]).cpu().tolist(),
                strict=True,
            )
        )
        validation = _rank_zero_call(
            lambda: evaluate_model(raw_model, val_data, config, device, workers=config.train.workers),
            rank,
            world,
        )
        score = validation["aggregate"]["normalized_rmse"]
        improved = score < best_score
        best_score = min(score, best_score)
        row = dict(
            epoch=epoch + 1,
            global_step=global_step,
            learning_rate=last_lr,
            gradient_norm=gradient_norm,
            train=train_metrics,
            val=validation["aggregate"],
            best_normalized_rmse=best_score,
            distributed_repeated_examples=world * samples_per_rank - len(train_data),
        )
        history.append(row)
        rng = [None] * world
        if world > 1:
            dist.all_gather_object(rng, capture_rng())
        else:
            rng[0] = capture_rng()

        state = dict(
            version=CHECKPOINT_VERSION,
            config=config.to_dict(),
            model=raw_model.state_dict(),
            optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),
            epoch=epoch + 1,
            global_step=global_step,
            best_score=best_score,
            history=history,
            rng=rng,
            runtime=runtime,
        )
        _rank_zero_call(partial(_commit_epoch, output, state, improved, validation), rank, world)
    return dict(
        output=str(output),
        completed_epochs=max(epoch_start, final_epoch),
        global_step=global_step,
        best_normalized_rmse=best_score,
        checkpoint=str(output / "last.pt"),
        world_size=world,
    )
