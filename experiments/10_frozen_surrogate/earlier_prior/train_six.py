"""Train fresh geometry weights on the expanded pool using six GPUs."""

import importlib
import json
import math
import os
import socket
import time
from datetime import datetime, timezone

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from areal_tokens.config import ExperimentConfig
from areal_tokens.data import GeometryDataset, collate_clouds, read_manifest, stable_seed, to_device
from areal_tokens.io import CHECKPOINT_VERSION, environment, save_checkpoint, write_json
from areal_tokens.losses import reconstruction_loss
from areal_tokens.model import GeometryAutoencoder
from areal_tokens.train import _commit_epoch, _rank_zero_call, _require_finite, capture_rng, lr_at_step, optimizer_for, seed_everything

from benchmark_support import MANIFEST, ROOT, install_shared_scoring
from fixed_global_batches import FixedGlobalBatchSampler


def main():
    rank, world, local_rank = (int(os.environ[name]) for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    assert world == 6
    assert torch.cuda.device_count() == 6
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.preferred_linalg_library("cusolver")
    # Ragged masked point counts make cuDNN rebuild expensive attention plans.
    # Native efficient SDPA avoids this per-shape CPU cost; the bf16 forward and
    # gradient check is recorded in attention_backend_check.json.
    torch.backends.cuda.enable_cudnn_sdp(False)
    dist.init_process_group("nccl")
    try:
        run(rank, world, device)
    finally:
        dist.destroy_process_group()


def run(rank, world, device):
    started = time.monotonic()
    config = ExperimentConfig.load(ROOT / "config.json")
    assert config.model.dropout == 0 and config.model.hint_noise == 0
    assert config.train.amp == "bf16" and config.train.accumulation == 1
    records, metadata = _rank_zero_call(lambda: read_manifest(MANIFEST), rank, world)
    training_records = [r for r in records if r["split"] == "train"]
    validation_records = [r for r in records if r["split"] == "val"]
    assert len(training_records) == 61822 and len(validation_records) == 1844
    assert sum(r["split"] == "test" for r in records) == 1825
    steps_per_epoch = math.ceil(len(training_records) / 64)
    total_steps = config.train.epochs * steps_per_epoch
    assert config.train.epochs == 15 and steps_per_epoch == 966 and total_steps == 14490
    training_data = GeometryDataset(training_records, config.model, config.data, config.train.seed, training=True)
    validation_data = GeometryDataset(validation_records, config.model, config.data, config.train.seed, fixed_queries=True)
    assert not (ROOT / "run/last.pt").exists(), "This run must begin from fresh weights"
    seed_everything(config.train.seed + rank)
    raw_model = GeometryAutoencoder(config.model).to(device)
    optimizer = optimizer_for(raw_model, config.train)
    assert len(optimizer.state) == 0
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    history, global_step, epoch_start, best_score = [], 0, 0, float("inf")
    model = DistributedDataParallel(raw_model, device_ids=[device.index], broadcast_buffers=False)
    initial_ranks = [None] * world
    dist.all_gather_object(initial_ranks, dict(rank=rank, local_rank=device.index, host=socket.gethostname(),
                          gpu=torch.cuda.get_device_name(device), pid=os.getpid()))
    assert len({r["local_rank"] for r in initial_ranks}) == 6
    assert all("B200" in r["gpu"] for r in initial_ranks)
    install_shared_scoring()
    evaluation = importlib.import_module("areal_tokens.evaluate")
    initial_steps = global_step
    runtime = dict(world_size=world, device_type="cuda", deterministic=False,
                   torch=str(torch.__version__), numpy=environment()["numpy"],
                   threads=torch.get_num_threads(), python=environment()["python"],
                   cuda_version=torch.version.cuda, device_name=torch.cuda.get_device_name(device),
                   trainer="fixed_global_batch_v1", global_batch_size=64,
                   training_linalg="cusolver", validation_linalg="magma",
                   training_cudnn_sdpa=False, validation_cudnn_sdpa=True,
                   rank_batch_sizes=[11, 11, 11, 11, 10, 10],
                   final_rank_batch_sizes=[11, 11, 10, 10, 10, 10])

    def progress(phase, **extra):
        if rank == 0:
            elapsed = time.monotonic() - started
            updates = global_step - initial_steps
            write_json(ROOT / "progress.json", {
                "phase": phase, "utc": datetime.now(timezone.utc).isoformat(),
                "optimizer_steps": global_step, "total_optimizer_steps": total_steps,
                "epoch": min(15, global_step // steps_per_epoch + 1), "completed_epochs": global_step // steps_per_epoch,
                "planned_epochs": 15, "gpus": world, "global_batch_size": 64,
                "elapsed_seconds_this_launch": elapsed,
                "estimated_remaining_seconds": elapsed / updates * (total_steps - global_step) if updates else None,
                **extra,
            })

    def initialize():
        write_json(ROOT / "run/environment.six.json", dict(**environment(), runtime=runtime))
        write_json(ROOT / "six_gpu_started.json", {
            "initialization": "random native GeometryAutoencoder constructor; DDP broadcasts seed-17 rank-zero weights",
            "initial_epoch": epoch_start, "initial_global_step": global_step,
            "weights_and_optimizer_restored": False,
            "initial_optimizer_state_entries": 0, "ranks": initial_ranks,
            "runtime": runtime, "data": metadata, "training_geometries": len(training_records),
        })
    _rank_zero_call(initialize, rank, world)
    progress("starting", completed_epochs=epoch_start)
    for epoch in range(epoch_start, config.train.epochs):
        training_data.set_epoch(epoch)
        batches = FixedGlobalBatchSampler(len(training_data), 64, world, rank, config.train.seed, epoch)
        assert len(batches) == steps_per_epoch
        loader = DataLoader(training_data, batch_sampler=batches, num_workers=config.train.workers,
                            collate_fn=collate_clouds, pin_memory=True, persistent_workers=False,
                            generator=torch.Generator().manual_seed(stable_seed(config.train.seed, "loader", epoch)))
        model.train()
        running = torch.zeros(5, dtype=torch.float64, device=device)
        progress("training", epoch=epoch + 1, completed_epochs=epoch)
        for index, batch in enumerate(loader):
            batch = to_device(batch, device)
            count, global_count = len(batch["id"]), batches.global_size(index)
            assert count == len(batches.order[index * 64:(index + 1) * 64][rank::world])
            rate = lr_at_step(config.train, global_step, steps_per_epoch)
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(batch)
                losses = reconstruction_loss(prediction, batch, config.train)
            _require_finite(losses["loss"], device, world, f"Nonfinite loss at epoch {epoch + 1}, update {index}")
            # DDP averages gradients over ranks. Undo that average and weight
            # each rank by its actual number of shapes, retaining a global mean.
            (losses["loss"] * (world * count / global_count)).backward()
            norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config.train.gradient_clip)
            _require_finite(norm, device, world, "Nonfinite distributed gradient")
            optimizer.step()
            global_step += 1
            with torch.no_grad():
                running[:4] += torch.stack([losses[k].detach().double()
                                             for k in ("loss", "distance", "normal", "normal_fraction")]) * count
                running[4] += count
            if global_step == initial_steps + 1 or global_step % 25 == 0:
                progress("training", epoch=epoch + 1, completed_epochs=epoch, learning_rate=rate)
        dist.all_reduce(running)
        assert int(running[4].item()) == len(training_data)
        train_metrics = dict(zip(("loss", "distance", "normal", "normal_fraction"),
                                 (running[:4] / running[4]).cpu().tolist(), strict=True))
        progress("validation", epoch=epoch + 1, completed_epochs=epoch)

        def validate():
            # The reference evaluation batch remains 32 even though training
            # uses 10/11 shapes per rank. Keep its working eigensolver backend.
            torch.backends.cuda.preferred_linalg_library("magma")
            torch.backends.cuda.enable_cudnn_sdp(True)
            try:
                return evaluation.evaluate_model(raw_model, validation_data, config, device,
                                                 workers=config.train.workers)
            finally:
                torch.backends.cuda.preferred_linalg_library("cusolver")
                torch.backends.cuda.enable_cudnn_sdp(False)

        validation = _rank_zero_call(validate, rank, world)
        score = validation["aggregate"]["normalized_rmse"]
        improved = score < best_score
        best_score = min(best_score, score)
        row = dict(epoch=epoch + 1, global_step=global_step, learning_rate=rate,
                   gradient_norm=float(norm), train=train_metrics, val=validation["aggregate"],
                   best_normalized_rmse=best_score, distributed_repeated_examples=0,
                   world_size=world, global_batch_size=64)
        history.append(row)
        rng = [None] * world
        dist.all_gather_object(rng, capture_rng())
        state = dict(version=CHECKPOINT_VERSION, config=config.to_dict(), model=raw_model.state_dict(),
                     optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), epoch=epoch + 1,
                     global_step=global_step, best_score=best_score, history=history, rng=rng,
                     runtime=runtime, training_recipe=json.loads((ROOT / "scale_recipe.json").read_text()))

        def commit():
            _commit_epoch(ROOT / "run", state, improved, validation)
            write_json(ROOT / "validation" / f"epoch_{epoch + 1:02d}.json", validation)
            if row["val"]["normal_degrees"] <= min(r["val"]["normal_degrees"] for r in history):
                save_checkpoint(ROOT / "run/best_normal.pt", state)
            progress("trained" if epoch == 14 else "training", completed_epochs=epoch + 1,
                     last_validation=row["val"])
        _rank_zero_call(commit, rank, world)
    if rank == 0:
        write_json(ROOT / "training_result.json", {
            "completed_epochs": 15, "global_step": global_step, "world_size": world,
            "global_batch_size": 64, "best_normalized_rmse": best_score,
            "initialization": "random", "training_geometries": len(training_records),
        })


if __name__ == "__main__":
    main()
