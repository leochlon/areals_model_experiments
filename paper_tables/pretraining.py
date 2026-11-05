"""Production geometry pretraining (Section 3.2, Appendix E, Table A.1 'Production', Table N.1).

Rebuilt from results/pretraining/ (training config, per-epoch validation history, checkpoint
metadata and the progress record at the stop point) and results/pretraining_extra/pool_ready.json
(the verified training-pool splits). Code: experiments/01_pretraining/.
"""
from __future__ import annotations

import json
import math

from paper_tables._core import RESULTS, Table, fmt, load_json


def _metrics() -> list[dict]:
    with open(RESULTS / "pretraining" / "metrics.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _int(value: float) -> str:
    return f"{int(round(value)):,}"


def tables() -> list[Table]:
    config = load_json("pretraining", "config.json")
    history = _metrics()
    checkpoints = load_json("pretraining", "checkpoints.json")["checkpoints"]
    progress = load_json("pretraining", "progress.json")
    pool = load_json("pretraining_extra", "pool_ready.json")

    selected_ckpt = checkpoints["best_normal.pt"]
    recipe = selected_ckpt["training_recipe"]
    runtime = selected_ckpt["runtime"]

    # Selection rule of train_six.py: best_normal.pt is rewritten whenever the epoch's validation
    # normal error is the lowest so far, so the selected epoch is the argmin over completed epochs.
    selected = min(history, key=lambda row: row["val"]["normal_degrees"])
    epoch = selected["epoch"]
    steps = selected["global_step"]
    nrmse = selected["val"]["normalized_rmse"]
    normal_deg = selected["val"]["normal_degrees"]
    val_scored = selected["val"]["shapes"]

    splits = pool["combined_splits"]
    train_n, val_n, test_n = splits["train"], splits["val"], splits["test"]
    global_batch = runtime["global_batch_size"]
    per_epoch = math.ceil(train_n / global_batch)
    params = selected_ckpt["parameters"]

    # ---------------- Table 1: the production run and its selected checkpoint ----------------
    t = Table(
        key="Pretraining (Section 3.2, Appendix E)",
        title="Production geometry pretraining on Objaverse-XL (Sketchfab subset): data, schedule, "
        "selected checkpoint and validation reconstruction with 60% of each surface hidden",
        columns=["Quantity", "Rebuilt", "Paper", "Source (results/)"],
    )
    t.rows.append(["Training meshes", _int(t.check("training meshes", train_n, 61822, 0)), "61,822",
                   "pretraining_extra/pool_ready.json"])
    t.rows.append(["Validation meshes", _int(t.check("validation meshes", val_n, 1844, 0)), "1,844",
                   "pretraining_extra/pool_ready.json"])
    t.rows.append(["Validation meshes scored at the selected epoch",
                   _int(t.check("validation meshes scored", val_scored, 1844, 0)), "1,844",
                   "pretraining/metrics.jsonl"])
    t.rows.append(["Test meshes (held out, not scored)", _int(t.check("test meshes", test_n, 1825, 0)),
                   "1,825", "pretraining_extra/pool_ready.json"])
    t.rows.append(["Planned epochs (cosine horizon)",
                   str(t.check("planned epochs", config["train"]["epochs"], 15, 0)), "15",
                   "pretraining/config.json"])
    t.rows.append(["Seed", str(t.check("seed", config["train"]["seed"], 17, 0)), "17",
                   "pretraining/config.json"])
    t.rows.append(["Updates per epoch (ceil(train / global batch))", _int(per_epoch), "—",
                   "derived"])
    t.rows.append(["Completed epochs before the stop", str(progress["completed_epochs"]), "—",
                   "pretraining/progress.json"])
    t.rows.append(["Selected epoch (lowest validation normal error)", str(t.check("selected epoch", epoch, 11, 0)),
                   "11", "pretraining/metrics.jsonl"])
    t.rows.append(["Updates at the selected checkpoint", _int(t.check("updates at selected checkpoint", steps, 10626, 0)),
                   "10,626", "pretraining/metrics.jsonl"])
    t.rows.append(["Validation distance nRMSE", fmt(t.check("distance nRMSE", nrmse, 0.098, 3), 3), "0.098",
                   "pretraining/metrics.jsonl"])
    t.rows.append(["Validation normal-line error (deg)", fmt(t.check("normal-line error (deg)", normal_deg, 22.0, 1), 1),
                   "22.0", "pretraining/metrics.jsonl"])
    t.rows.append(["Parameters (millions)", f"{fmt(t.check('parameters (M)', params / 1e6, 3.2, 1), 1)} ({params:,})",
                   "3.2", "pretraining/checkpoints.json"])

    # Consistency of the stored files with each other (reported, not paper numbers).
    same_val = all(selected_ckpt["metrics"]["val"][k] == selected["val"][k] for k in selected["val"])
    t.notes.append(
        f"best_normal.pt metadata: epoch {selected_ckpt['epoch']}, {selected_ckpt['global_step']:,} updates, "
        f"validation metrics identical to the metrics.jsonl row: {same_val}; "
        f"{epoch} x {per_epoch} = {epoch * per_epoch:,} updates."
    )
    t.notes.append(
        f"The run was planned for {progress['planned_epochs']} epochs ({progress['total_optimizer_steps']:,} updates) "
        f"and was stopped during epoch {progress['epoch']} after {progress['optimizer_steps']:,} updates; "
        f"{progress['completed_epochs']} epochs completed, and the selected checkpoint is the minimum over those epochs."
    )
    t.notes.append(
        f"The epoch with the lowest validation distance nRMSE is a different checkpoint "
        f"(best.pt, epoch {checkpoints['best.pt']['epoch']}, nRMSE "
        f"{checkpoints['best.pt']['metrics']['val']['normalized_rmse']:.4f}); it is not used in the paper."
    )
    t.notes.append(
        f"Recipe in the checkpoint: {recipe['train_geometries']:,}/{recipe['validation_geometries']:,}/"
        f"{recipe['test_geometries']:,} meshes, {recipe['gpus']} GPUs, global batch {recipe['global_batch_size']}."
    )

    # ---------------- Table 2: the recipe printed in Section 3.2 and Appendix E ----------------
    data, train, model = config["data"], config["train"], config["model"]
    r = Table(
        key="Pretraining recipe (Section 3.2, Appendix E)",
        title="Production pretraining recipe and architecture, read from the stored training config "
        "and checkpoint runtime record",
        columns=["Setting", "Rebuilt", "Paper", "Config key"],
    )

    def row(label, value, paper, decimals, key):
        r.check(label, value, paper, decimals)
        show = _int if decimals == 0 else (lambda v: fmt(v, decimals))
        r.rows.append([label, show(value), show(paper), key])

    row("Encoder points", data["encoder_points"], 4096, 0, "data.encoder_points")
    row("Decoder queries", data["queries"], 1024, 0, "data.queries")
    row("Hidden surface area (%)", 100 * data["mask_fraction"], 60, 0, "data.mask_fraction")
    row("Mask patches", data["mask_patches"], 32, 0, "data.mask_patches")
    row("Near-surface queries (%)", 100 * data["near_fraction"], 70, 0, "data.near_fraction")
    row("Examples shown whole (%)", 100 * data["full_input_probability"], 10, 0, "data.full_input_probability")
    r.rows.append(["Rotation augmentation", data["augment_rotation"], "SO(3)", "data.augment_rotation"])
    row("Normal sign-flip probability", data["normal_sign_flip_probability"], 0.5, 1,
        "data.normal_sign_flip_probability")
    row("Peak learning rate (x 1e-4)", train["learning_rate"] * 1e4, 3, 0, "train.learning_rate")
    row("Weight decay", train["weight_decay"], 0.05, 2, "train.weight_decay")
    r.rows.append(["Schedule", f"{train['warmup_steps']} warm-up updates, cosine to "
                   f"{train['min_lr_ratio']} x peak", "cosine decay", "train.warmup_steps, train.min_lr_ratio"])
    row("Huber threshold (distance loss)", train["huber_delta"], 0.05, 2, "train.huber_delta")
    r.rows.append(["Precision", train["amp"], "bfloat16", "train.amp"])
    row("Global batch", runtime["global_batch_size"], 64, 0, "checkpoint runtime.global_batch_size")
    row("GPUs", runtime["world_size"], 6, 0, "checkpoint runtime.world_size")
    r.rows.append(["GPU type", runtime["device_name"], "B200", "checkpoint runtime.device_name"])
    row("Width", model["width"], 192, 0, "model.width")
    row("Attention heads", model["heads"], 4, 0, "model.heads")
    row("Areal tokens", model["latents"], 256, 0, "model.latents")
    row("Encoder self-attention blocks", model["encoder_depth"], 4, 0, "model.encoder_depth")
    row("Decoder blocks", model["decoder_depth"], 2, 0, "model.decoder_depth")
    row("Fourier bands", model["fourier_bands"], 8, 0, "model.fourier_bands")
    r.notes.append(
        "Near-surface queries: training uses target_mode 'provided', i.e. the query sets written at corpus "
        "preparation (geom_latents.prepare.query_distribution, near_fraction=0.7 by default); the config's "
        "near_fraction carries the same value."
    )
    r.notes.append(
        "The encoder also has one cross-attention block that reads the visible samples into the tokens "
        "(areal_tokens.model.GeometryEncoder.read)."
    )
    return [t, r]
