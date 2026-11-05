"""Matched DrivAerML runs using NVIDIA's AeroJEPA training loop and losses."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
EXAMPLE = ROOT / "physicsnemo/examples/cfd/external_aerodynamics/aerojepa"
sys.path[:0] = [str(ROOT), str(ROOT / "physicsnemo"), str(EXAMPLE)]

import hydra
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from drivaerml import DrivAerMLDataset
from src.datapipes import superwing_collate
from src.losses import build_recon_loss_from_config, build_sigreg_from_config
from src.training import (
    ExponentialMovingAverage, build_lr_scheduler, build_optimizer,
    get_autocast_context, move_batch_to_device, set_seed,
)
from train import _run_epoch, _save_checkpoint, _slice_batch_sample

PRIOR = "/workspace/home/areal_objaverse_scratch61822_15ep_20260927T152147Z/run/best_normal.pt"


class FrozenPriorEMA(ExponentialMovingAverage):
    @torch.no_grad()
    def update(self, model):
        super().update(model)
        for name, tensor in model.state_dict().items():
            if name.startswith("trunk.context_encoder.encoder."):
                self.shadow[name].copy_(tensor.detach())


@torch.inference_mode()
def score_pressure(model, loader, dataset, device, world_size):
    """Geometry-only inference; mean per-car raw-Cp relative L2."""
    model.eval()
    rows = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        for index in range(batch["context_pos"].shape[0]):
            sample = _slice_batch_sample(batch, index)
            with get_autocast_context(device, "bf16"):
                pred = model(
                    context_pos=sample["context_pos"],
                    context_feat=sample["context_feat"],
                    gen_params=sample["gen_params"],
                    query_pos=sample["query_pos"],
                    query_sdf=sample["query_sdf"],
                )
            raw_pred = pred.float() * dataset.std + dataset.mean
            raw_true = sample["query_target"].float() * dataset.std + dataset.mean
            error = raw_pred - raw_true
            rows.append({
                "case_id": sample["case_id"],
                "rl2": float(error.norm() / raw_true.norm().clamp_min(1e-12)),
                "mse": float(error.square().mean()),
                "points": len(raw_true),
            })
    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, rows)
        rows = [row for shard in gathered for row in shard]
    rows.sort(key=lambda row: row["case_id"])
    return {"rl2": sum(row["rl2"] for row in rows) / len(rows), "cars": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("native", "areal"), required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-per-rank", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--areal-tokens", type=int, default=256)
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", device_id=device)
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s", force=True)
    torch.set_num_threads(4)
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    run_name = args.arm if args.areal_tokens == 256 else f"{args.arm}_{args.areal_tokens}"
    out = ROOT / "runs" / run_name
    out.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.create({"use_te": False})
    cfg.model = OmegaConf.load(EXAMPLE / "conf/model/aerojepa.yaml")
    cfg.training = OmegaConf.load(EXAMPLE / "conf/training/superwing.yaml")
    cfg.model.trunk.context_encoder.max_point_tokens = 256
    cfg.model.trunk.target_encoder.point_input_dim = 4
    cfg.model.trunk.decoder.out_dim = 1
    cfg.training.epochs = args.epochs
    cfg.training.batch_size = args.batch_per_rank
    cfg.training.eval_batch_size = 1
    cfg.training.max_eval_batches = None
    cfg.comparison = {"arm": args.arm, "seed": args.seed, "world_size": world_size,
                      "global_batch_size": args.batch_per_rank * world_size,
                      "areal_checkpoint": PRIOR if args.arm == "areal" else None,
                      "areal_frozen": args.arm == "areal",
                      "areal_tokens": args.areal_tokens,
                      "selection_metric": "geometry_only_raw_cp_mean_per_car_rl2_ema"}

    model = hydra.utils.instantiate(cfg.model)
    if args.arm == "areal":
        from areal_context import ArealContextEncoder
        with torch.random.fork_rng(devices=[]):
            model.trunk.context_encoder = ArealContextEncoder(PRIOR, token_dim=128,
                                                              latent_count=args.areal_tokens)
    prior_state = ({k: v.detach().cpu().clone() for k, v in model.context_encoder.encoder.state_dict().items()}
                   if args.arm == "areal" else None)
    model.to(device)
    train_data = DrivAerMLDataset(ROOT, "train", seed=args.seed, geometry_tokens=args.areal_tokens)
    val_data = DrivAerMLDataset(ROOT, "val", seed=args.seed, geometry_tokens=args.areal_tokens)
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank,
                                       shuffle=True, seed=args.seed, drop_last=False)
    val_sampler = DistributedSampler(val_data, num_replicas=world_size, rank=rank,
                                     shuffle=False, drop_last=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_per_rank, sampler=train_sampler,
                              num_workers=0, collate_fn=superwing_collate, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=1, sampler=val_sampler,
                            num_workers=0, collate_fn=superwing_collate, pin_memory=True)
    losses = dict(
        recon_loss_fn=build_recon_loss_from_config(cfg.training.loss.recon).to(device),
        sigreg_loss_fn=build_sigreg_from_config(cfg.training.loss.sigreg).to(device),
        sigreg_context_loss_fn=build_sigreg_from_config(cfg.training.loss.sigreg_context).to(device),
    )
    optimizer = build_optimizer(model, cfg.training.optimizer)
    scheduler = build_lr_scheduler(optimizer, name=cfg.training.scheduler.name,
                                    epochs=args.epochs, steps_per_epoch=len(train_loader),
                                    warmup_epochs=cfg.training.scheduler.warmup_epochs)
    ema = FrozenPriorEMA(model, decay=cfg.training.ema.decay)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    common = dict(model=model, device=device, precision="bf16", loss_cfg=cfg.training.loss,
                  world_size=world_size, is_main=rank == 0, max_batches=None, log_every=10, **losses)
    if rank == 0:
        metadata = dict(arm=args.arm, train_cars=len(train_data), val_cars=len(val_data),
                        epochs=args.epochs, steps_per_epoch=len(train_loader),
                        total_updates=args.epochs * len(train_loader),
                        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                        physicsnemo_commit="426f7552da4b4fa675e404e8a4f437e27681b668",
                        config=OmegaConf.to_container(cfg, resolve=True))
        (out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        logging.info("START arm=%s train=%d val=%d GPUs=%d updates=%d", run_name,
                     len(train_data), len(val_data), world_size, args.epochs * len(train_loader))
    best, best_epoch = float("inf"), None
    started = time.time()
    for epoch in range(args.epochs):
        train_data.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        epoch_started = time.time()
        train_metrics = _run_epoch(loader=train_loader, optimizer=optimizer,
                                   lr_scheduler=scheduler, ema=ema, epoch=epoch,
                                   grad_clip_norm=cfg.training.grad_clip_norm, scaler=scaler, **common)
        ema.apply_to(model)
        try:
            val_metrics = _run_epoch(loader=val_loader, optimizer=None, lr_scheduler=None,
                                     ema=None, epoch=epoch, grad_clip_norm=0.0, **common)
            score = score_pressure(model, val_loader, val_data, device, world_size)
        finally:
            ema.restore(model)
        improved = score["rl2"] < best
        if improved:
            best, best_epoch = score["rl2"], epoch + 1
        row = dict(epoch=epoch + 1, updates=(epoch + 1) * len(train_loader),
                   train=train_metrics, validation=val_metrics, val_rl2=score["rl2"],
                   best_val_rl2=best, best_epoch=best_epoch,
                   lr=optimizer.param_groups[0]["lr"],
                   epoch_seconds=time.time() - epoch_started, elapsed_seconds=time.time() - started)
        if rank == 0:
            with (out / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            (out / "latest.json").write_text(json.dumps(row, indent=2) + "\n")
            logging.info("EPOCH %d/%d arm=%s train=%.6f val=%.6f pressure_rL2=%.6f best=%.6f seconds=%.1f",
                         epoch + 1, args.epochs, args.arm, train_metrics["loss"], val_metrics["loss"],
                         score["rl2"], best, row["epoch_seconds"])
            if improved:
                _save_checkpoint(path=out / "best.pt", model=model, optimizer=optimizer,
                                 lr_scheduler=scheduler, ema=ema, epoch=epoch,
                                 best_val=val_metrics["loss"], cfg=cfg)
                (out / "best_validation.json").write_text(json.dumps(dict(epoch=epoch + 1, **score), indent=2) + "\n")
            if epoch + 1 == args.epochs:
                _save_checkpoint(path=out / "last.pt", model=model, optimizer=optimizer,
                                 lr_scheduler=scheduler, ema=ema, epoch=epoch,
                                 best_val=val_metrics["loss"], cfg=cfg)
        if world_size > 1:
            dist.barrier()
    live_score = score_pressure(model, val_loader, val_data, device, world_size)
    if rank == 0:
        final = dict(arm=args.arm, completed_epochs=args.epochs, best_ema_rl2=best,
                     best_epoch=best_epoch, final_ema_rl2=score["rl2"],
                     final_live_rl2=live_score["rl2"], elapsed_seconds=time.time() - started)
        if args.arm == "areal":
            final["frozen_prior_unchanged"] = all(torch.equal(prior_state[k], v.detach().cpu()) for k, v
                                                  in model.context_encoder.encoder.state_dict().items())
        (out / "final.json").write_text(json.dumps(final, indent=2) + "\n")
        (out / "final_live_validation.json").write_text(json.dumps(live_score, indent=2) + "\n")
        _save_checkpoint(path=out / "final_live.pt", model=model, optimizer=optimizer,
                         lr_scheduler=scheduler, ema=None, epoch=args.epochs - 1,
                         best_val=live_score["rl2"], cfg=cfg)
        logging.info("COMPLETE %s", json.dumps(final))
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
