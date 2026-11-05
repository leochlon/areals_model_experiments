"""Frozen areal tokens as a surrogate's geometry input: Table L.1, the Appendix L text (with the 0.9% in
Section 4.2), the "Frozen surrogate" paragraph of Appendix E and the frozen-surrogate row of Table A.1.

Sources (all under results/):
  frozen_surrogate/<run>/final.json                       final_live_rl2 of each Table L.1 row (16 validation cars)
  frozen_surrogate_extra/<run>/run.json                   recorded configuration: tokens, input points, epochs, updates, cars
  frozen_surrogate_extra/<run>/final_live_validation.json per-car relative L2 on the 2,048 fixed queries
  frozen_surrogate_extra/pointmae_1024/features_ready_metadata.json  Point-MAE feature cache: input points, point groups
  frozen_surrogate_extra/earlier_prior/                   the frozen geometry checkpoint's config, per-epoch metrics and size
  frozen_surrogate_extra/drivaerml_preparation/config.json  surface sampling used to prepare the DrivAerML cars
  reserved_cars/summary.json                              48-car, three-seed replication (frozen family)
"""
from __future__ import annotations

import json
import re

import numpy as np

from paper_tables._core import RESULTS, Table, fmt, load_json

# Table L.1 rows in the paper's order: (geometry prior, run directory, block).
ROWS = [
    ("AREALS", "areal", "matched 256"),
    ("AeroJEPA tutorial", "native", "matched 256"),
    ("AREALS", "areal_1024", "matched 1,024"),
    ("Utonia", "utonia_1024", "matched 1,024"),
    ("Point-MAE", "pointmae_1024", "matched 1,024"),
    ("AREALS", "areal_1024_input65536", "other budgets"),
    ("AREALS", "areal_32768_input65536", "other budgets"),
]
# Printed paper values for each row: (tokens, points, rel. L2).
PAPER = {
    "areal": (256, 4096, 0.3283),
    "native": (256, 4096, 0.3338),
    "areal_1024": (1024, 4096, 0.3244),
    "utonia_1024": (1024, 4096, 0.3294),
    "pointmae_1024": (1024, 1024, 0.3353),
    "areal_1024_input65536": (1024, 65536, 0.3198),
    "areal_32768_input65536": (32768, 65536, 0.3073),
}
# Caption: "†Interpolated from 64 point groups."
PAPER_POINTMAE_GROUPS = 64


def _runs():
    runs = {}
    for _, name, _ in ROWS:
        runs[name] = dict(
            final=load_json("frozen_surrogate", name, "final.json"),
            run=load_json("frozen_surrogate_extra", name, "run.json"),
            validation=load_json("frozen_surrogate_extra", name, "final_live_validation.json"),
        )
    return runs


def _earlier_prior():
    root = RESULTS / "frozen_surrogate_extra" / "earlier_prior"
    history = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines() if line.strip()]
    config = load_json("frozen_surrogate_extra", "earlier_prior", "config.json")
    result = load_json("frozen_surrogate_extra", "earlier_prior", "training_result.json")
    # train_six.py saves run/best_normal.pt whenever validation normal error is at or below every earlier
    # epoch, so the frozen checkpoint is the last epoch attaining the minimum.
    degrees = np.array([row["val"]["normal_degrees"] for row in history])
    selected = max(row["epoch"] for row in history if row["val"]["normal_degrees"] == degrees.min())
    return dict(history=history, config=config, result=result, selected_epoch=selected)


def _tokens(name, run, prior):
    comparison = run["config"]["comparison"]
    if name == "native":
        return run["config"]["model"]["trunk"]["context_encoder"]["max_point_tokens"]
    if name == "areal":
        # The first revision records no token key: the frozen encoder keeps its own latent count.
        return prior["config"]["model"]["latents"]
    if "areal_tokens" in comparison:
        return comparison["areal_tokens"]
    return comparison["geometry_tokens"]


def _pointmae_metadata():
    return load_json("frozen_surrogate_extra", "pointmae_1024", "features_ready_metadata.json")


def _points(name, run, preparation):
    comparison = run["config"]["comparison"]
    if comparison.get("dense_geometry_points"):
        return comparison["dense_geometry_points"]
    if name == "pointmae_1024":
        return _pointmae_metadata()["input_points"]
    return preparation["encoder_points"]


def _has_dense(run) -> bool:
    return bool(run["config"]["comparison"].get("dense_geometry_points"))


def _common(values, label):
    """The value shared by every run; NaN (so the check fails) if the stored runs disagree."""
    values = list(values)
    return values[0] if len(set(values)) == 1 else float("nan")


def _frozen_comparison(summary):
    return next(c for c in summary["comparisons"]
                if c["family"] == "frozen" and c["first"] == "areal" and c["second"] == "native")


def table_l1(runs, prior, preparation) -> Table:
    table = Table(
        key="Table L.1",
        title="Frozen areal tokens as a surrogate's geometry input (mean per-car relative L2 on 2,048 fixed "
              "pressure queries, 16 DrivAerML validation cars; better entry of each matched pair in bold)",
        columns=["Geometry prior", "Tokens", "Points", "Rel. L2", "Run"],
    )
    scores = {name: runs[name]["final"]["final_live_rl2"] for _, name, _ in ROWS}
    blocks = {}
    for _, name, block in ROWS:
        blocks.setdefault(block, []).append(name)
    for label, name, block in ROWS:
        run = runs[name]["run"]
        paper_tokens, paper_points, paper_rl2 = PAPER[name]
        tokens = table.check(f"{label} [{name}] tokens", _tokens(name, run, prior), paper_tokens, 0)
        points = table.check(f"{label} [{name}] input points", _points(name, run, preparation), paper_points, 0)
        rl2 = table.check(f"{label} [{name}] rel. L2", scores[name], paper_rl2, 4)
        shown = fmt(rl2, 4)
        if block != "other budgets" and scores[name] == min(scores[n] for n in blocks[block]):
            shown = f"**{shown}**"
        if name == "pointmae_1024":
            points_text = f"{points:,}"
            tokens_text = f"{tokens:,} †"
        else:
            points_text = f"{points:,}" if _has_dense(run) else f"≤{points:,}"
            tokens_text = f"{tokens:,}"
        table.rows.append([label, tokens_text, points_text, shown, f"`{name}`"])

    cars = _common((len(runs[n]["validation"]["cars"]) for n in scores), "validation cars")
    table.check("validation cars", cars, 16, 0)
    queries = _common((p["points"] for n in scores for p in runs[n]["validation"]["cars"]), "fixed queries per car")
    table.check("fixed pressure queries per car", queries, 2048, 0)
    worst = max(abs(np.mean([c["rl2"] for c in runs[n]["validation"]["cars"]]) - scores[n]) for n in scores)
    table.notes.append(f"Per-car scores in final_live_validation.json average to final.json's final_live_rl2 "
                       f"(largest difference {worst:.1e}).")
    pointmae = _pointmae_metadata()
    table.check("† Point-MAE point groups", pointmae["native_groups"], PAPER_POINTMAE_GROUPS, 0)
    table.notes.append(f"† Point-MAE reads {pointmae['input_points']:,} points in {pointmae['native_groups']} groups; "
                       f"its group features are mapped to the 1,024 anchors by {pointmae['anchor_mapping']} "
                       "(features_ready_metadata.json of the Point-MAE feature cache).")
    table.notes.append("≤4,096 is the surface sample count of the DrivAerML preparation (encoder_points); the 65,536 "
                       "and token counts are the values recorded in each run's run.json.")
    return table


def appendix_l(runs, prior, preparation, summary) -> Table:
    table = Table(key="Appendix L", title="Text numbers (with Section 4.2's 0.9%)", columns=["Claim", "Rebuilt", "Source"])
    areal = runs["areal"]["final"]["final_live_rl2"]
    native = runs["native"]["final"]["final_live_rl2"]
    reduction = 100 * (1 - areal / native)
    epochs = _common((runs[n]["run"]["epochs"] for n in runs), "epochs")
    train_cars = _common((runs[n]["run"]["train_cars"] for n in runs), "training cars")
    matched_tokens = _common((_tokens(n, runs[n]["run"], prior) for n in ("areal", "native")), "matched tokens")
    matched_points = _common((_points(n, runs[n]["run"], preparation) for n in ("areal", "native")), "matched points")

    comparison = _frozen_comparison(summary)
    frozen = summary["summaries"]["frozen"]
    setting, metric = comparison["setting"], comparison["metric"]
    mean_areal = frozen["areal"][setting][metric]["mean"]
    mean_native = frozen["native"][setting][metric]["mean"]
    seeds = _common((len(comparison["per_seed_means_first"]), len(comparison["per_seed_means_second"])), "seeds")
    cars48 = _common((comparison["first_minus_second"]["cars"], frozen["areal"][setting][metric]["cars"],
                      frozen["native"][setting][metric]["cars"]), "reserved cars")
    reduction48 = 100 * (1 - mean_areal / mean_native)
    low, high = comparison["first_minus_second"]["low"], comparison["first_minus_second"]["high"]
    difference = comparison["first_minus_second"]["mean"]
    best = runs["areal_32768_input65536"]
    best_run = best["run"]["config"]["comparison"]

    rows = [
        ("trained for 20 epochs", epochs, 20, 0, "run.json epochs (all seven runs)"),
        ("on the same 420 cars", train_cars, 420, 0, "run.json train_cars (all seven runs)"),
        ("lower error by 1.65% (%)", reduction, 1.65, 2, "1 − AREALS/AeroJEPA, Table L.1 rows 1–2"),
        ("matched 256-token budget", matched_tokens, 256, 0, "Table L.1 rows 1–2"),
        ("matched 4,096-point budget", matched_points, 4096, 0, "DrivAerML preparation encoder_points"),
        ("48 reserved cars", cars48, 48, 0, "reserved_cars/summary.json, frozen comparison"),
        ("three-seed comparison", seeds, 3, 0, "per-seed means in the frozen comparison"),
        ("48-car AREALS mean rel. L2", mean_areal, 0.32885, 5, f"summaries.frozen.areal.{setting}"),
        ("48-car AeroJEPA mean rel. L2", mean_native, 0.33193, 5, f"summaries.frozen.native.{setting}"),
        ("48-car reduction 0.9% (%)", reduction48, 0.9, 1, "1 − AREALS/AeroJEPA over 48 cars"),
        ("paired interval, lower", low, -0.00571, 5, "first_minus_second.low (car bootstrap)"),
        ("paired interval, upper", high, -0.00034, 5, "first_minus_second.high (car bootstrap)"),
        ("lowest error 0.3073", best["final"]["final_live_rl2"], 0.3073, 4, "areal_32768_input65536"),
        ("at 32,768 tokens", best_run["geometry_tokens"], 32768, 0, "areal_32768_input65536 run.json"),
        ("and 65,536 points", best_run["dense_geometry_points"], 65536, 0, "areal_32768_input65536 run.json"),
        ("61,822-mesh geometry checkpoint", prior["result"]["training_geometries"], 61822, 0,
         "earlier_prior/training_result.json"),
        ("selected at epoch 14", prior["selected_epoch"], 14, 0, "argmin validation normal error, metrics.jsonl"),
        ("Section 4.2: lower the error by 0.9% (%)", reduction48, 0.9, 1, "as above"),
    ]
    for label, value, paper, decimals, source in rows:
        table.check(label, value, paper, decimals)
        shown = f"{value:,}" if decimals == 0 else fmt(value, decimals)
        table.rows.append([label, shown, source])
    stored = 100 * comparison["relative_mean_reduction"]
    table.notes.append(f"Stored relative_mean_reduction is {stored:.4f}%; recomputed from the two means: {reduction48:.4f}%. "
                       f"Mean paired difference {difference:.5f}; interval is the 95% car bootstrap stored by the scorer.")
    return table


def appendix_e_and_a1(runs, prior, summary) -> Table:
    table = Table(key="Appendix E and Table A.1", title="Frozen-surrogate recipe and experiment-map row",
                  columns=["Claim", "Rebuilt", "Source"])
    comparison = _frozen_comparison(summary)
    epochs = _common((runs[n]["run"]["epochs"] for n in runs), "epochs")
    updates = _common((runs[n]["run"]["total_updates"] for n in runs), "updates")
    train_cars = _common((runs[n]["run"]["train_cars"] for n in runs), "training cars")
    val_cars = _common((runs[n]["run"]["val_cars"] for n in runs), "validation cars")
    seeds = len(comparison["per_seed_means_first"])
    cars48 = comparison["first_minus_second"]["cars"]
    queries48 = int(re.fullmatch(r"fixed-(\d+)-query", comparison["setting"]).group(1))
    rows = [
        ("E: trains for 20 epochs", epochs, 20, 0, "run.json epochs"),
        ("E: 1,060 updates", updates, 1060, 0, "run.json total_updates"),
        ("E: 48-car replication", cars48, 48, 0, "reserved_cars/summary.json"),
        ("E: three downstream seeds", seeds, 3, 0, "reserved_cars/summary.json"),
        ("E: same 2,048 fixed queries per car", queries48, 2048, 0, "frozen comparison setting"),
        ("E: earlier 61,822-mesh checkpoint", prior["result"]["training_geometries"], 61822, 0,
         "earlier_prior/training_result.json"),
        ("E: selected at epoch 14", prior["selected_epoch"], 14, 0, "earlier_prior/metrics.jsonl"),
        ("A.1: earlier 61,822-mesh checkpoint", prior["result"]["training_geometries"], 61822, 0,
         "earlier_prior/training_result.json"),
        ("A.1: (epoch 14)", prior["selected_epoch"], 14, 0, "earlier_prior/metrics.jsonl"),
        ("A.1: AeroJEPA tutorial, 420 cars", train_cars, 420, 0, "run.json train_cars"),
        ("A.1: 16 validation cars", val_cars, 16, 0, "run.json val_cars"),
        ("A.1: 48 reserved cars", cars48, 48, 0, "reserved_cars/summary.json"),
        ("A.1: seeds, 3", seeds, 3, 0, "reserved_cars/summary.json"),
        ("A.1: seeds, on 48 cars", cars48, 48, 0, "reserved_cars/summary.json"),
    ]
    for label, value, paper, decimals, source in rows:
        table.check(label, value, paper, decimals)
        table.rows.append([label, f"{value:,}", source])
    history = prior["history"]
    table.notes.append(f"Earlier prior: {len(history)} epochs of {history[-1]['global_step'] // len(history):,} updates; "
                       f"selected epoch {prior['selected_epoch']} has validation normal error "
                       f"{min(r['val']['normal_degrees'] for r in history):.2f} degrees "
                       f"(geometry tokens per shape: {prior['config']['model']['latents']}).")
    return table


def tables() -> list[Table]:
    runs = _runs()
    prior = _earlier_prior()
    preparation = load_json("frozen_surrogate_extra", "drivaerml_preparation", "config.json")
    summary = load_json("reserved_cars", "summary.json")
    return [
        table_l1(runs, prior, preparation),
        appendix_l(runs, prior, preparation, summary),
        appendix_e_and_a1(runs, prior, summary),
    ]
