from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import sys
import time

import numpy as np
import torch
from box import Box
from torch_geometric.loader import DataLoader
import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "GABI"
OUT = ROOT / "bench/gabi"
sys.path.insert(0, str(REPO))

from datatools import HelmholtzCarDataset, norm_data
from models import GCNGeomAutoencoder
from training_scripts.train_single_gpu import compute_loss


def save_atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")
    device = torch.device("cuda")
    source_run = OUT / "runs/GCN_dim_100_uid_xNGi"
    source = source_run / "trained_models/best.pt"
    validations = [float(x) for x in re.findall(r"Validation - Loss: ([0-9.eE+-]+)",
        (ROOT / "train.log").read_text())]
    assert len(validations) == 4
    start_epoch = int(np.argmin(validations)) + 1
    assert start_epoch == 3
    best_loss = min(validations)
    state = torch.load(source, map_location="cpu", weights_only=False)
    assert set(state) == {"model_state_dict", "trainset_stats"}
    config = Box.from_yaml(filename=str(source_run / "config.yml"), Loader=yaml.FullLoader)
    assert config.hyperparameters.epochs == 455 and config.hyperparameters.batch_size == 100
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = OUT / f"runs/GCN_dim_100_resume_{stamp}"
    (run / "trained_models").mkdir(parents=True)
    config.to_yaml(filename=str(run / "config.yml"))
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    train = HelmholtzCarDataset(config.io_settings.train_dataset_path)
    val = HelmholtzCarDataset(config.io_settings.valid_dataset_path)
    batch_size = config.hyperparameters.batch_size
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
        num_workers=config.run_settings.num_t_workers,
        persistent_workers=config.run_settings.num_t_workers > 0, pin_memory=False)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False,
        num_workers=config.run_settings.num_v_workers,
        persistent_workers=config.run_settings.num_v_workers > 0, pin_memory=False)
    assert dict(config.data_dims) == train.get_data_dims_dict()
    model = GCNGeomAutoencoder(**config.data_dims, **config.hyperparameters, **config.model_settings)
    model.load_state_dict(state["model_state_dict"], strict=True)
    assert all(torch.equal(v, state["model_state_dict"][k]) for k, v in model.state_dict().items())
    model.trainset_stats = state["trainset_stats"]
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.hyperparameters.start_lr),
        weight_decay=float(config.hyperparameters.weight_decay))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(config.hyperparameters.lr_decay))
    resumed_lr = float(config.hyperparameters.start_lr) * float(config.hyperparameters.lr_decay)**start_epoch
    for group in optimizer.param_groups:
        group["lr"] = resumed_lr
    schedule_state = scheduler.state_dict()
    schedule_state.update(last_epoch=start_epoch, _step_count=start_epoch+1, _last_lr=[resumed_lr])
    scheduler.load_state_dict(schedule_state)
    best_path = run / "trained_models/best.pt"
    save_atomic(best_path, state)
    # The first run saved neither optimizer nor random state: AdamW starts fresh at the scheduled learning rate
    # and the random state is reset to seed 0. Weights and normalisation come from the source checkpoint.
    provenance = dict(source_checkpoint=str(source), retained_epoch=start_epoch, discarded_completed_epoch=4,
        target_epoch=455, restored_lr=resumed_lr, original_best_validation_loss=best_loss, run=str(run))
    (ROOT / "resume.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print("CONTINUATION", json.dumps(provenance), flush=True)
    history = []
    for epoch in range(start_epoch+1, int(config.hyperparameters.epochs)+1):
        started = time.perf_counter()
        train_metrics = np.zeros(3)
        model.train()
        for data in train_loader:
            data = norm_data(data, model.trainset_stats).to(device)
            optimizer.zero_grad()
            losses = compute_loss(model(data))
            if not all(torch.isfinite(value) for value in losses):
                raise RuntimeError(f"nonfinite training loss at epoch {epoch}")
            losses[0].backward()
            optimizer.step()
            train_metrics += [value.item() for value in losses]
        train_metrics /= len(train_loader)
        scheduler.step()
        val_metrics = np.zeros(3)
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                data = norm_data(data, model.trainset_stats).to(device)
                losses = compute_loss(model(data))
                if not all(torch.isfinite(value) for value in losses):
                    raise RuntimeError(f"nonfinite validation loss at epoch {epoch}")
                val_metrics += [value.item() for value in losses]
        val_metrics /= len(val_loader)
        weights = dict(model_state_dict=model.state_dict(), trainset_stats=model.trainset_stats)
        if val_metrics[0] < best_loss:
            best_loss = float(val_metrics[0])
            save_atomic(best_path, weights)
        record = dict(epoch=epoch, updates=epoch*len(train_loader), train=train_metrics.tolist(),
            validation=val_metrics.tolist(), lr=scheduler.get_last_lr()[0],
            best_validation=best_loss, seconds=time.perf_counter()-started)
        history.append(record)
        (run / "history.json").write_text(json.dumps(history, indent=2)+"\n")
        full = dict(**weights, epoch=epoch, optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(), best_validation_loss=best_loss,
            torch_rng_state=torch.get_rng_state(), cuda_rng_states=torch.cuda.get_rng_state_all(),
            numpy_rng_state=np.random.get_state(), python_rng_state=random.getstate(), provenance=provenance)
        save_atomic(run / "trained_models/last_full.pt", full)
        if epoch % int(config.io_settings.save_epochs) == 0:
            save_atomic(run / f"trained_models/e{epoch}.pt", weights)
        print(f"Epoch {epoch}/455 - LR: {record['lr']:.8f}, Loss: {train_metrics[0]:.6f}, "
            f"Recon: {train_metrics[1]:.6f}, MMD: {train_metrics[2]:.6f}", flush=True)
        print(f"  Validation - Loss: {val_metrics[0]:.6f}, Recon: {val_metrics[1]:.6f}, "
            f"MMD: {val_metrics[2]:.6f}; {record['seconds']:.2f}s", flush=True)
    trained = dict(run=str(run), checkpoint=str(best_path), selected="lowest validation loss across original and resumed runs",
        config=str(run / "config.yml"), continuation=provenance, created=datetime.now(timezone.utc).isoformat())
    (OUT / "trained.json").write_text(json.dumps(trained, indent=2)+"\n")
    print("TRAINING_COMPLETE", json.dumps(trained), flush=True)


if __name__ == "__main__":
    main()
