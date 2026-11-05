"""Architecture ablations: decoder transfer, spatial interface, normal targets, head initialisation, sampling
density and finite tokens (Table E.1, Section 4.5, Section 4.6, Appendices E, F, G and H, Table A.1).

Rebuilt from
  results/architecture_ablations/decoder_transfer/   the 24-model, 40-car decoder study (summary.json), the
                                                    backbone parameter count (encoder_parameter_audit.json),
                                                    the A-B-A head-initialisation test (initialization_reversal.json)
                                                    and the four-epoch geometry rows with original and isotropic
                                                    normal heads (geometry_repair_aggregation.json);
  results/architecture_ablations/normal_targets/     seven controlled geometry arms x three seeds on 128 held-out
                                                    objects, recomputed from raw predictions (grouped.json);
  results/architecture_ablations/density_shift/      density-shift stress test (summary.json, independent_grouped.json);
  results/architecture_ablations/finite_tokens/      finite-token sensitivity (summary.json, raw_summary.json) and the
                                                    448-case recomputation (independent_verification.json);
  results/architecture_ablations/averaging_views/    resampled-view averaging (independent_grouped.json);
  results/reserved_cars/summary.json                 48-car means of the seven controlled variants;
  results/architecture_ablations_extra/              per-car scores, per-model task records, the three I/3-head
                                                    geometry test files and the frozen statistics protocol of the
                                                    decoder study, the decoder-reset initialisation check, the
                                                    21-contrast interface analysis and the fixed-anchor density rows.
Code: experiments/11_architecture_ablations/.
"""
from __future__ import annotations

import math
import re

import numpy as np

from paper_tables._core import RESULTS, Table, fmt, load_csv, load_json

AA = "architecture_ablations"
EXTRA = "architecture_ablations_extra"


def _ceil(value: float, exponent: int, digits: int) -> float:
    """`value` in units of 10**exponent, rounded up to `digits` decimals: how the paper prints an upper bound."""
    scale = 10 ** digits
    return math.ceil(value / 10 ** exponent * scale - 1e-9) / scale


def _contrast(summary: dict, first: str, second: str) -> dict:
    return next(c for c in summary["six_contrasts"] if (c["first"], c["second"]) == (first, second))


def _params(audit: list, arm: str) -> float:
    return next(a for a in audit if a["arm"] == arm)["registered_backbone_parameters"]


def _row(t: Table, label: str, value: float, paper: float, decimals: int, source: str, shown: str | None = None):
    t.check(label, value, paper, decimals)
    t.rows.append([label, shown if shown is not None else fmt(value, decimals), f"{paper:.{decimals}f}", source])


# ----------------------------------------------------------------------------------------------- decoder transfer

def table_e1(summary: dict, audit: list) -> Table:
    t = Table(
        key="Table E.1",
        title="The transfer runs through the query decoder: full-surface pressure nRMSE (lower is better) on 40 "
        "withheld DrivAerML cars, three seeds, identical field examples, queries and updates",
        columns=["Transfer pathway", "Backbone parameters (M)", "Seeds 0 / 1 / 2", "nRMSE", "Paper"],
    )
    rows = [
        ("Point-MAE encoder + fresh query decoder", "fresh_decoder_pointmae", "pointmae", 21.8, 0.2556),
        ("AREALS encoder + fresh query decoder", "fresh_decoder_areals", "areals", 2.3, 0.2708),
        ("AREALS encoder + pretrained query decoder", "isotropic_reference", "areals", None, 0.2399),
    ]
    for label, group, arm, paper_params, paper in rows:
        cell = summary["means"][group]
        params = _params(audit, arm) / 1e6
        shown_params = "(same encoder)"
        if paper_params is not None:
            t.check(f"{label}: backbone parameters (M)", params, paper_params, 1)
            shown_params = fmt(params, 1)
        t.check(f"{label}: nRMSE", cell["mean"], paper, 4)
        t.rows.append([label, shown_params, " / ".join(fmt(v, 4) for v in cell["per_seed"]),
                       fmt(cell["mean"], 4), f"{paper:.4f}"])
    t.check("Caption: withheld test cars", summary["test_cars"], 40, 0)
    t.check("Caption: seeds", len(summary["means"]["isotropic_reference"]["per_seed"]), 3, 0)
    t.notes.append(f"Means over {summary['test_cars']} test cars and {len(summary['means']['isotropic_reference']['per_seed'])} "
                   "seeds of per-car area-weighted nRMSE; each seed's car-mean is shown.")
    return t


def table_decoder_text(summary: dict, audit: list) -> Table:
    t = Table(
        key="Section 4.5 and Appendix E 'Decoder transfer'",
        title="Decoder-transfer contrasts, budgets and controls",
        columns=["Quantity", "Rebuilt", "Paper", "Source (results/)"],
    )
    src = f"{AA}/decoder_transfer/summary.json"
    keep_reset = _contrast(summary, "isotropic_reference", "fresh_decoder_areals")
    keep_pointmae = _contrast(summary, "isotropic_reference", "fresh_decoder_pointmae")
    m = summary["means"]
    _row(t, "4.5: Point-MAE + fresh query decoder", m["fresh_decoder_pointmae"]["mean"], 0.256, 3, src)
    _row(t, "4.5: AREALS + fresh query decoder", m["fresh_decoder_areals"]["mean"], 0.271, 3, src)
    _row(t, "4.5: AREALS + pretrained query decoder", m["isotropic_reference"]["mean"], 0.240, 3, src)
    _row(t, "4.5: keeping vs resetting the decoder, error reduction (%)",
         100 * keep_reset["relative_mean_error_reduction"], 11.4, 1, src)
    _row(t, "4.5: seeds in which keeping the decoder is lower",
         sum(d < 0 for d in keep_reset["per_seed_differences"]), 3, 0, src)
    _row(t, "4.5: pretrained decoder vs Point-MAE, error reduction (%)",
         100 * keep_pointmae["relative_mean_error_reduction"], 6.1, 1, src)
    lo, hi = keep_reset["bonferroni_all6_interval"]
    _row(t, "E: keep vs reset decoder, adjusted interval low", lo, -0.0360, 4, src)
    _row(t, "E: keep vs reset decoder, adjusted interval high", hi, -0.0263, 4, src)
    lo, hi = keep_pointmae["bonferroni_all6_interval"]
    _row(t, "E: pretrained decoder vs Point-MAE, adjusted interval low", lo, -0.0186, 4, src)
    _row(t, "E: pretrained decoder vs Point-MAE, adjusted interval high", hi, -0.0128, 4, src)
    _row(t, "E/A.1: training cars", summary["train_cars"], 380, 0, src)
    _row(t, "E/A.1: evaluation cars", summary["test_cars"], 40, 0, src)
    _row(t, "4.5: withheld DrivAerML cars", summary["test_cars"], 40, 0, src)
    init = load_json(EXTRA, "decoder_transfer", "decoder_initialization_check.json")
    _row(t, "E: non-decoder state tensors identical after decoder reset", init["shared_non_decoder_tensors"], 102, 0,
         f"{EXTRA}/decoder_transfer/decoder_initialization_check.json")
    t.notes.append(f"Decoder reset check: all non-decoder tensors exactly identical = "
                   f"{init['all_non_decoder_tensors_exactly_identical']}; the full and fresh-decoder models share "
                   "the same scalar head.")
    asrc = f"{AA}/decoder_transfer/encoder_parameter_audit.json"
    _row(t, "E: Point-MAE active backbone parameters (M)", _params(audit, "pointmae") / 1e6, 21.825, 3, asrc)
    _row(t, "E: AREALS backbone parameters (M)", _params(audit, "areals") / 1e6, 2.274, 3, asrc)
    tokens = {a["tokens"] for a in audit}
    cap = {a["input_point_cap"] for a in audit}
    _row(t, "E: tokens per shape (both encoders)", tokens.pop() if len(tokens) == 1 else float("nan"), 256, 0, asrc)
    _row(t, "E: input point cap (both encoders)", cap.pop() if len(cap) == 1 else float("nan"), 4096, 0, asrc)
    protocol = load_json(EXTRA, "decoder_transfer", "decoder_controls_protocol.json")
    stats = protocol["statistics"]
    draws = int(re.search(r"(\d+)-draw", stats).group(1))
    level = float(re.search(r"Bonferroni([0-9.]+)%", stats).group(1))
    psrc = f"{EXTRA}/decoder_transfer/decoder_controls_protocol.json"
    _row(t, "E: paired car-bootstrap draws", draws, 10000, 0, psrc)
    _row(t, "E: Bonferroni-adjusted interval level (%)", level, 99.17, 2, psrc)
    _row(t, "E: contrasts in the adjusted family", len(summary["six_contrasts"]), 6, 0, src)
    family_groups = sorted({g for c in summary["six_contrasts"] for g in (c["first"], c["second"])})
    pathways = {"fresh_decoder_pointmae", "fresh_decoder_areals", "isotropic_reference", "original_reference"}
    within = sum(c["first"] in pathways and c["second"] in pathways for c in summary["six_contrasts"])
    t.notes.append(f"The Bonferroni family is the six contrasts (the four primary contrasts of the 18 models "
                   f"plus the two decoder comparisons of decoder_controls_protocol.json); together they involve "
                   f"{len(family_groups)} model groups: "
                   f"{', '.join(family_groups)}.")

    # Per-model task records of the 24 field fits (runtime settings recorded by run_task.py).
    task_files = sorted((RESULTS / EXTRA / "decoder_transfer" / "tasks").glob("*.json")) + \
        sorted((RESULTS / EXTRA / "decoder_transfer" / "extra_tasks").glob("*.json"))
    task_rows = [load_json(*p.relative_to(RESULTS).parts) for p in task_files]
    dev_queries = {r["fixed_development_queries"] for r in task_rows}
    tsrc = f"{EXTRA}/decoder_transfer/{{tasks,extra_tasks}}/*.json"
    _row(t, "E: fixed development queries per car (all models)",
         dev_queries.pop() if len(dev_queries) == 1 else float("nan"), 8192, 0, tsrc)
    t.notes.append(f"{len(task_rows)} per-model task records: training cars {sorted({r['train_cars'] for r in task_rows if 'train_cars' in r})}, "
                   f"downstream seeds {sorted({r['true_downstream_seed'] for r in task_rows})}, evaluation query chunk "
                   f"{sorted({r['numerical_query_chunk'] for r in task_rows})}.")

    ratio = _params(audit, "pointmae") / _params(audit, "areals")
    _row(t, "4.5: Point-MAE backbone 'nearly ten times' the AREALS backbone (ratio)", ratio, 10, 0, asrc,
         shown=f"{ratio:.2f}")
    geometry = [load_json(EXTRA, "decoder_transfer", "architecture", "reference", f"seed{s}", "test.json") for s in range(3)]
    gsrc = f"{EXTRA}/decoder_transfer/architecture/reference/seed*/test.json"
    epochs = {g["selected_epoch"] for g in geometry}
    _row(t, "4.5/E.1: AREALS encoder pretrained for four epochs (selected = final epoch)",
         epochs.pop() if len(epochs) == 1 else float("nan"), 4, 0, gsrc)
    t.notes.append(f"The three AREALS geometry checkpoints (I/3 head, seeds 0-2) each ran {geometry[0]['steps']:,} updates "
                   f"and were selected at their last epoch.")

    # Independent rebuild of the six adjusted intervals from the stored per-car scores.
    rows = load_csv(EXTRA, "decoder_transfer", "per_car.csv")
    cars: list[str] = []
    for r in rows:  # file order is the frozen test-split order the scorer used
        if r["car"] not in cars:
            cars.append(r["car"])
    lookup = {(r["group"], int(r["seed"]), r["car"]): float(r["nrmse"]) for r in rows}
    seeds = sorted({int(r["seed"]) for r in rows})
    matrices = {g: np.array([[lookup[g, s, c] for c in cars] for s in seeds]) for g in {r["group"] for r in rows}}
    family = len(summary["six_contrasts"])
    alpha = 1 - level / 100
    sample = np.random.default_rng(20261024).integers(0, len(cars), (draws, len(cars)))
    deviation = 0.0
    for c in summary["six_contrasts"]:
        difference = (matrices[c["first"]] - matrices[c["second"]]).mean(0)
        low, high = np.quantile(difference[sample].mean(1), [alpha / 2, 1 - alpha / 2])
        deviation = max(deviation, abs(low - c["bonferroni_all6_interval"][0]),
                        abs(high - c["bonferroni_all6_interval"][1]),
                        abs(matrices[c["first"]].mean() - c["means"][0]))
    t.notes.append(f"The {family} adjusted intervals rebuilt from per_car.csv ({len(cars)} cars x {len(seeds)} seeds x "
                   f"{len(matrices)} groups; {draws:,} draws, rng seed 20261024 as in score_decoder_controls.py, "
                   f"level {level}%) agree with summary.json to {deviation:.1e}.")
    t.notes.append("Settings not stored in results/ (BF16 training and FP32 evaluation; the projection of Point-MAE's "
                   "384 channels to the common width 192) are set in fast_runtime.py and online_backbone.py.")
    return t


# ------------------------------------------------------------------------------------- interface and normal targets

def table_normal_lines(grouped: dict) -> Table:
    t = Table(
        key="Appendix F 'Normal lines' and Section 4.6",
        title="Unoriented normal-angle error (degrees, lower is better) of the four-epoch controlled geometry "
        "models on held-out objects, three seeds; 95% object-bootstrap interval (descriptive)",
        columns=["Normal target", "Seeds 0 / 1 / 2", "Mean [95% interval]", "Paper (App. F)", "Paper (Sec. 4.6)"],
    )
    arms = [("Projector (sign-free)", "reference", 25.34, 25.3),
            ("Oriented vector", "oriented_normal_target", 28.69, 28.7),
            ("Unoriented vector", "unoriented_vector_target", 39.17, 39.2)]
    for label, arm, paper_f, paper_46 in arms:
        cell = grouped["summaries"][arm]["normal_degrees"]
        seeds = [s["normal_degrees"] for s in grouped["seeds"] if s["arm"] == arm]
        t.check(f"F: {label}", cell["mean"], paper_f, 2)
        t.check(f"4.6: {label}", cell["mean"], paper_46, 1)
        t.rows.append([label, " / ".join(fmt(v, 2) for v in seeds),
                       f"{cell['mean']:.2f} [{cell['low']:.2f}, {cell['high']:.2f}]", f"{paper_f:.2f}", f"{paper_46:.1f}"])
    objects = {grouped["summaries"][a]["normal_degrees"]["objects"] for _, a, _, _ in arms}
    t.check("F/4.6/A.1: held-out objects", grouped["test_objects"], 128, 0)
    t.check("F/4.6: seeds", len([s for s in grouped["seeds"] if s["arm"] == "reference"]), 3, 0)
    t.notes.append(f"{grouped['models']} models ({len(grouped['summaries'])} arms x "
                   f"{grouped['models'] // len(grouped['summaries'])} seeds) re-evaluated from raw saved predictions; "
                   f"objects per summary: {sorted(objects)}.")
    return t


def table_interface(reserved: dict) -> Table:
    t = Table(
        key="Appendix F 'A coordinate-aware interface' and Section 4.6",
        title="Surface-anchored, rotary-positioned tokens against the unanchored configuration without rotary "
        "positions: full-surface nRMSE on the 48 reserved cars with no observed pressure",
        columns=["Quantity", "Rebuilt", "Paper", "Source (results/)"],
    )
    contrast = load_json(EXTRA, "coordinate_interface", "combined_interface_contrast.json")
    combined = contrast["combined_interface_comparison"]
    arch = reserved["summaries"]["architecture"]
    anchored = arch["reference"]["unobserved"]["nrmse"]["mean"]
    unanchored = arch["unanchored_no_rotary"]["unobserved"]["nrmse"]["mean"]
    rsrc = "reserved_cars/summary.json"
    csrc = f"{EXTRA}/coordinate_interface/combined_interface_contrast.json"
    _row(t, "F: anchored + rotary nRMSE", anchored, 0.2469, 4, rsrc)
    _row(t, "F: unanchored, no rotary nRMSE", unanchored, 0.3898, 4, rsrc)
    _row(t, "4.6: anchored + rotary nRMSE", anchored, 0.247, 3, rsrc)
    _row(t, "4.6: unanchored, no rotary nRMSE", unanchored, 0.390, 3, rsrc)
    reduction = 100 * (1 - anchored / unanchored)
    _row(t, "F: error reduction (%)", reduction, 36.7, 1, rsrc)
    _row(t, "4.6: error reduction (%)", reduction, 36.7, 1, rsrc)
    lo, hi = (100 * v for v in combined["relative_reduction_all21_bonferroni_interval"])
    _row(t, "F: adjusted interval of the reduction, low (%)", lo, 35.0, 1, csrc)
    _row(t, "F: adjusted interval of the reduction, high (%)", hi, 38.1, 1, csrc)
    _row(t, "4.6: adjusted interval of the reduction, low (%)", lo, 35.0, 1, csrc)
    _row(t, "4.6: adjusted interval of the reduction, high (%)", hi, 38.1, 1, csrc)
    _row(t, "F: pairwise contrasts in the adjustment", len(contrast["all21_pairs"]), 21, 0, csrc)
    _row(t, "F: architecture variants", contrast["groups"], 7, 0, csrc)
    _row(t, "F/4.6/A.1: reserved cars", contrast["cars"], 48, 0, csrc)
    _row(t, "F: upstream/head initialisations", len(combined["per_seed_differences"]), 3, 0, csrc)
    deviation = max(abs(anchored - combined["reference_mean"]), abs(unanchored - combined["unanchored_no_rotary_mean"]),
                    abs(reduction / 100 - combined["relative_mean_error_reduction"]))
    t.notes.append(f"The 48-car means in reserved_cars/summary.json and the contrast file agree to {deviation:.1e}.")
    t.notes.append(f"Per-seed differences (anchored − unanchored): "
                   + ", ".join(f"{d:+.4f}" for d in combined["per_seed_differences"])
                   + f"; adjusted interval of the difference [{combined['difference_all21_bonferroni_interval'][0]:+.4f}, "
                   f"{combined['difference_all21_bonferroni_interval'][1]:+.4f}].")
    return t


def table_isotropic(summary: dict, grouped: dict) -> Table:
    t = Table(
        key="Appendix F 'Isotropic projector initialisation'",
        title="Starting the symmetric normal head at I/3: A-B-A gradient test, four-epoch geometry and 40-car pressure",
        columns=["Quantity", "Rebuilt", "Paper", "Source (results/)"],
    )
    reversal = load_json(AA, "decoder_transfer", "initialization_reversal.json")
    phases = {p["phase"]: p for p in reversal["sequence"]}
    rsrc = f"{AA}/decoder_transfer/initialization_reversal.json"
    _row(t, "Encoder normal-loss gradient norm, original head", phases["original"]["encoder_normal_gradient_norm"],
         155.0, 1, rsrc)
    _row(t, "Encoder normal-loss gradient norm, I/3 head", phases["isotropic"]["encoder_normal_gradient_norm"], 0, 0, rsrc)
    _row(t, "Encoder normal-loss gradient norm, original head restored",
         phases["restored_original"]["encoder_normal_gradient_norm"], 155.0, 1, rsrc)
    _row(t, "Head gradient norm at I/3", phases["isotropic"]["normal_head_gradient_norm"], 2.17, 2, rsrc)
    t.rows.append(["Distance predictions identical across the three phases",
                   str(reversal["initial_distance_predictions_exactly_identical"]), "identical", rsrc])

    repair = load_json(AA, "decoder_transfer", "geometry_repair_aggregation.json")
    gsrc = f"{AA}/decoder_transfer/geometry_repair_aggregation.json"
    reference = [r for r in repair["rows"] if r["arm"] == "reference"]
    old_mse, new_mse = repair["summary"]["reference"]["old_mse"], repair["summary"]["reference"]["new_mse"]
    _row(t, "Distance MSE, original head", old_mse, 0.01167, 5, gsrc)
    _row(t, "Distance MSE, I/3 head", new_mse, 0.00854, 5, gsrc)
    _row(t, "Distance MSE reduction (%)", 100 * (1 - new_mse / old_mse), 26.8, 1, gsrc)
    _row(t, "Normal error (deg), original head", float(np.mean([r["old_angle"] for r in reference])), 25.34, 2, gsrc)
    _row(t, "Normal error (deg), I/3 head", float(np.mean([r["new_angle"] for r in reference])), 24.87, 2, gsrc)

    ssrc = f"{AA}/decoder_transfer/summary.json"
    c = _contrast(summary, "isotropic_reference", "original_reference")
    _row(t, "Pressure nRMSE, original head", summary["means"]["original_reference"]["mean"], 0.2564, 4, ssrc)
    _row(t, "Pressure nRMSE, I/3 head", summary["means"]["isotropic_reference"]["mean"], 0.2399, 4, ssrc)
    _row(t, "Pressure error reduction (%)", 100 * c["relative_mean_error_reduction"], 6.4, 1, ssrc)
    _row(t, "Seeds improved", sum(d < 0 for d in c["per_seed_differences"]), 3, 0, ssrc)
    _row(t, "Adjusted paired interval, low", c["bonferroni_all6_interval"][0], -0.01960, 5, ssrc)
    _row(t, "Adjusted paired interval, high", c["bonferroni_all6_interval"][1], -0.01378, 5, ssrc)

    # Cross-checks: the original-head geometry rows against the raw-prediction recomputation of the same 3 models.
    raw = grouped["summaries"]["reference"]
    seeds_raw = [s for s in grouped["seeds"] if s["arm"] == "reference"]
    deviation = max(abs(old_mse - raw["normalized_mse"]["mean"]),
                    max(abs(r["old_mse"] - s["normalized_mse"]) for r, s in zip(reference, seeds_raw)))
    angle_dev = max(abs(r["old_angle"] - s["normal_degrees"]) for r, s in zip(reference, seeds_raw))
    t.notes.append(f"Original-head rows agree with the independent raw-prediction recomputation "
                   f"(normal_targets/grouped.json) to {deviation:.1e} in MSE and {angle_dev:.1e} deg in angle.")
    t.notes.append(f"Mean of the three seed rows: MSE {np.mean([r['old_mse'] for r in reference]):.6f} -> "
                   f"{np.mean([r['new_mse'] for r in reference]):.6f} (the stored summary fields).")
    tests = [load_json(EXTRA, "decoder_transfer", "architecture", "reference", f"seed{r['seed']}", "test.json")
             for r in reference]
    new_dev = max(max(abs(r["new_mse"] - x["normalized_mse"]), abs(r["new_angle"] - x["normal_degrees"]))
                  for r, x in zip(reference, tests))
    t.notes.append(f"The I/3-head rows agree with the per-seed test.json files written by repair.py "
                   f"({EXTRA}/decoder_transfer/architecture/reference/seed*/test.json, {len(tests[0]['rows'])} objects "
                   f"each) to {new_dev:.1e}.")
    t.notes.append(f"The A-B-A test ran on CPU in FP32 on one fixed batch; restored-gradient maximum difference "
                   f"{reversal['restored_gradient_max_difference']:g}.")
    return t


# --------------------------------------------------------------------------------------------- sampling and tokens

def table_density() -> Table:
    t = Table(
        key="Section 4.6 and Appendix G 'Density shift'",
        title="Normalised output drift when one geometric half is sampled ninefold more densely, anchors and frame "
        "fixed; log-area (reference) against ordinary attention",
        columns=["Quantity", "Rebuilt", "Paper", "Source (results/)"],
    )
    g = load_json(AA, "density_shift", "independent_grouped.json")
    stored = load_json(AA, "density_shift", "summary.json")
    src = f"{AA}/density_shift/independent_grouped.json"
    mode = "biased_fixed_anchors_frame"
    area = g["summaries"]["reference"][mode]["prediction_change"]["mean"]
    ordinary = g["summaries"]["ordinary_attention"][mode]["prediction_change"]["mean"]
    diff = g["contrasts"][mode]["prediction_change_reference_minus_ordinary"]
    _row(t, "G: drift, log-area attention", area, 0.002535, 6, src)
    _row(t, "G: drift, ordinary attention", ordinary, 0.013706, 6, src)
    _row(t, "4.6: drift, log-area attention", area, 0.0025, 4, src)
    _row(t, "4.6: drift, ordinary attention", ordinary, 0.0137, 4, src)
    _row(t, "G: paired drift difference", diff["mean"], -0.011171, 6, src)
    _row(t, "G: object-grouped 95% interval, low", diff["low"], -0.012692, 6, src)
    _row(t, "G: object-grouped 95% interval, high", diff["high"], -0.009690, 6, src)
    rows = load_json(EXTRA, "density_shift", "density_fixed_rows.json")
    means = {arm: float(np.mean([r["prediction_change"] for r in rows if r["arm"] == arm]))
             for arm in ["reference", "ordinary_attention"]}
    _row(t, "4.6: less drift with log-area attention (%)",
         100 * (1 - means["reference"] / means["ordinary_attention"]), 81.5, 1,
         f"{EXTRA}/density_shift/density_fixed_rows.json")
    _row(t, "G/A.1: meshes", g["meshes"], 32, 0, src)
    arms = len(g["summaries"])
    _row(t, "G/A.1: geometry-pretraining seeds per attention type", g["models"] // arms, 3, 0, src)
    draws_per_case = {}
    for r in rows:
        draws_per_case.setdefault((r["arm"], r["seed"], r["id"]), set()).add(r["draw"])
    counts = {len(v) for v in draws_per_case.values()}
    draw_ids = sorted({d for v in draws_per_case.values() for d in v})
    _row(t, "G: density draws per seed", counts.pop() if len(counts) == 1 else float("nan"), 3, 0,
         f"{EXTRA}/density_shift/density_fixed_rows.json")
    t.notes.append(f"Density draws (sampling seeds) {draw_ids} for each of the {len(draws_per_case)} "
                   "(attention type, seed, mesh) cases.")
    _row(t, "G: saved prediction records checked", g["predictions"], 1344, 0, src)
    t.check("4.6: 95% interval of the difference excludes zero (1 = yes)", float(diff["high"] < 0), 1, 0)
    agree = max(abs(stored["settings"][a][m]["prediction_change"] - g["summaries"][a][m]["prediction_change"]["mean"])
                for a in g["summaries"] for m in g["summaries"][a])
    t.notes.append(f"Independent recomputation from the raw arrays: maximum metric disagreement "
                   f"{g['maximum_metric_disagreement']:g}; the scorer's summary.json agrees with it to {agree:.1e}.")
    t.notes.append(f"The 81.5% uses the {len(rows)} fixed-anchor rows (row means "
                   f"{means['reference']:.10f} and {means['ordinary_attention']:.10f}, equal to the object means above).")
    t.notes.append("The ninefold density ratio is a design setting (density_stress.py), not stored in results/. The "
                   "single-query 24% / 0.7% illustration of Figure G.1 is checked by the figures module.")
    return t


def table_g1() -> Table:
    t = Table(
        key="Table G.1",
        title="Finite-token prediction sensitivity on 32 raw meshes with the fixed production checkpoint: mean RMS "
        "change in unsigned-distance prediction divided by the object's RMS radius",
        columns=["Perturbation", "256 tokens", "1,024 tokens", "Paper (256 / 1,024)"],
    )
    s = load_json(AA, "finite_tokens", "summary.json")
    raw = load_json(AA, "finite_tokens", "raw_summary.json")
    # (label, mode, exponent of the printed unit or None, decimals, paper 256, paper 1024)
    rows = [("Point order", "point_order_permutation", -8, 2, 2.27, 1.58),
            ("Coupled coplanar subdivision", "subdivision_coupled", -9, 2, 6.60, 5.20),
            ("Independent surface resampling", "independent_resample", None, 5, 0.01316, 0.00921),
            ("Resampling, fixed anchors/frame/surfels", "resample_fixed_anchors_frame_surfels", None, 5, 0.00467, 0.00217)]
    for label, mode, exponent, decimals, p256, p1024 in rows:
        cells = []
        for tokens, paper in [("256", p256), ("1024", p1024)]:
            value = s["settings"][tokens][mode]["output_change"]
            scaled = value / 10 ** exponent if exponent is not None else value
            t.check(f"{label}, {tokens} tokens" + (f" (x1e{exponent})" if exponent else ""), scaled, paper, decimals)
            cells.append(f"{scaled:.{decimals}f}e{exponent:03d}" if exponent else fmt(value, decimals))
        papers = [f"{p:.{decimals}f}e{exponent:03d}" if exponent else f"{p:.{decimals}f}" for p in (p256, p1024)]
        t.rows.append([label, cells[0], cells[1], " / ".join(papers)])
    t.check("Caption/A.1: raw meshes", s["objects"], 32, 0)
    settings = sorted(int(k) for k in s["settings"])
    t.check("Column header: fewer tokens", settings[0], 256, 0)
    t.check("Column header: more tokens", settings[-1], 1024, 0)
    t.notes.append(f"The rerun with saved per-trial arrays (raw_summary.json) is identical to summary.json: {s == raw}.")
    return t


def table_attention() -> Table:
    t = Table(
        key="Appendix G 'Attention splitting'",
        title="Measure-preserving key splitting in a fixed attention fixture, and the independent recomputation of "
        "the saved finite-token predictions",
        columns=["Quantity", "Rebuilt", "Paper", "Source (results/)"],
    )
    s = load_json(AA, "finite_tokens", "summary.json")
    v = load_json(AA, "finite_tokens", "independent_verification.json")
    ssrc, vsrc = f"{AA}/finite_tokens/summary.json", f"{AA}/finite_tokens/independent_verification.json"
    area = s["attention"]["area"]["rms"]
    _row(t, "Log-area attention change, RMS (x1e-8)", area / 1e-8, 1.49, 2, ssrc)
    _row(t, "Ordinary attention change, RMS", s["attention"]["ordinary"]["rms"], 0.03192, 5, ssrc)
    _row(t, "Saved prediction cases", v["raw_prediction_cases"], 448, 0, vsrc)
    _row(t, "Token settings", v["token_settings"], 2, 0, vsrc)
    worst = v["maximum_disagreements"]
    _row(t, "Recomputation agrees on output change within (x1e-9, rounded up)",
         _ceil(worst["output_change"], -9, 1), 2.3, 1, vsrc)
    _row(t, "Recomputation agrees on distance NMSE within (x1e-19, rounded up)",
         _ceil(worst["distance_nmse"], -19, 1), 4.4, 1, vsrc)
    t.notes.append(f"The paper prints the two maxima as upper bounds, rounded up: output change "
                   f"{worst['output_change']:.4g} and distance NMSE {worst['distance_nmse']:.4g} (to the nearest digit "
                   f"these would print as 2.2e-9 and 4.3e-19).")
    t.notes.append(f"Fixture: log-area max |change| {s['attention']['area']['max_abs']:.3g}, ordinary "
                   f"{s['attention']['ordinary']['max_abs']:.3g}.")
    return t


def table_averaging() -> Table:
    t = Table(
        key="Appendix G 'Averaging resampled views'",
        title="Normal-angle error (degrees) of the frozen production checkpoint with 60% of the surface hidden: "
        "the standard view against the average of four independently resampled views",
        columns=["Quantity", "Rebuilt", "Paper", "Source (results/)"],
    )
    g = load_json(AA, "averaging_views", "independent_grouped.json")
    src = f"{AA}/averaging_views/independent_grouped.json"
    setting = next(x for x in g["settings"] if x["nuisance"] == "resample" and x["mask"] > 0)
    _row(t, "Hidden surface fraction (%)", 100 * setting["mask"], 60, 0, src)
    _row(t, "Meshes", g["meshes"], 32, 0, src)
    for key, label, before, after in [("normal_all", "All-query", 19.06, 18.53),
                                      ("normal_hidden", "Hidden-query", 24.72, 24.26)]:
        m = setting["metrics"][key]
        base = m["baseline"]["mean"]
        averaged = base + m["orbit_minus_baseline"]["mean"]
        _row(t, f"{label} normal error, one view", base, before, 2, src)
        _row(t, f"{label} normal error, average of resampled views", averaged, after, 2, src)
        t.notes.append(f"{label}: change {m['orbit_minus_baseline']['mean']:+.3f} deg, 95% object-bootstrap interval "
                       f"[{m['orbit_minus_baseline']['low']:+.3f}, {m['orbit_minus_baseline']['high']:+.3f}].")
    t.notes.append(f"Recomputed from {g['raw_prediction_files']} raw prediction files; maximum disagreement with the "
                   f"saved rows {g['maximum_metric_disagreement']['normal_degrees']:.1e} deg. The four-view count is "
                   "set in nuisance_raw.py (--draws 4).")
    return t


def table_controls(summary: dict) -> Table:
    t = Table(
        key="Appendix H (40-car decoder study)",
        title="Hidden-value perturbation checks and independent recomputation of the decoder-transfer scores",
        columns=["Quantity", "Rebuilt", "Paper", "Source (results/)"],
    )
    src = f"{AA}/decoder_transfer/summary.json"
    _row(t, "Hidden-value checks passed", summary["hidden_pressure_checks"], 96, 0, src)
    _row(t, "Recomputation agrees within (x1e-16, rounded up)",
         _ceil(summary["native_metric_max_disagreement"], -16, 1), 2.0, 1, src)
    extra = summary["extra_checks"]
    extra_passed = sum(c["passed"] for c in extra)
    primary = summary["hidden_pressure_checks"] - extra_passed
    t.notes.append(f"{summary['hidden_pressure_checks']} = {primary} checks on the {summary['models'] - len(extra)} "
                   f"primary models + {extra_passed} on the {len(extra)} decoder-control models "
                   f"({sum(c['failed'] for c in extra)} failed among the latter).")
    t.notes.append(f"Largest disagreement {summary['native_metric_max_disagreement']:.4g}, printed as an upper bound. "
                   "These checks used the harness default tolerance (1e-5 of the field spread), stricter than the 1e-3 "
                   "stated in Appendix H.")
    return t


def table_a1(summary: dict, grouped: dict, reserved: dict) -> Table:
    t = Table(
        key="Table A.1 (rows for these studies)",
        title="Experiment-map entries for decoder transfer, coordinate interface, normal targets, density shift and "
        "finite tokens",
        columns=["Row", "Quantity", "Rebuilt", "Paper"],
    )
    contrast = load_json(EXTRA, "coordinate_interface", "combined_interface_contrast.json")
    density = load_json(AA, "density_shift", "independent_grouped.json")
    finite = load_json(AA, "finite_tokens", "summary.json")
    entries = [
        ("Decoder transfer and head initialisation", "training cars", summary["train_cars"], 380),
        ("Decoder transfer and head initialisation", "withheld test cars", summary["test_cars"], 40),
        ("Decoder transfer and head initialisation", "seeds", len(summary["means"]["isotropic_reference"]["per_seed"]), 3),
        ("Coordinate interface", "reserved test cars", contrast["cars"], 48),
        ("Coordinate interface", "seeds", len(contrast["combined_interface_comparison"]["per_seed_differences"]), 3),
        ("Normal targets", "held-out objects", grouped["test_objects"], 128),
        ("Normal targets", "seeds", len([s for s in grouped["seeds"] if s["arm"] == "reference"]), 3),
        ("Density shift", "meshes", density["meshes"], 32),
        ("Density shift", "seeds", density["models"] // len(density["summaries"]), 3),
        ("Finite tokens", "meshes", finite["objects"], 32),
    ]
    for row, quantity, value, paper in entries:
        t.check(f"A.1 {row}: {quantity}", value, paper, 0)
        t.rows.append([row, quantity, fmt(value, 0), str(paper)])
    t.notes.append("Not stored as a number in results/: the Finite tokens row's single seed (one fixed production "
                   "checkpoint, set in finite_token.py).")
    return t


def tables() -> list[Table]:
    summary = load_json(AA, "decoder_transfer", "summary.json")
    audit = load_json(AA, "decoder_transfer", "encoder_parameter_audit.json")
    grouped = load_json(AA, "normal_targets", "grouped.json")
    reserved = load_json("reserved_cars", "summary.json")
    return [
        table_e1(summary, audit),
        table_decoder_text(summary, audit),
        table_normal_lines(grouped),
        table_interface(reserved),
        table_isotropic(summary, grouped),
        table_density(),
        table_g1(),
        table_attention(),
        table_averaging(),
        table_controls(summary),
        table_a1(summary, grouped, reserved),
    ]
