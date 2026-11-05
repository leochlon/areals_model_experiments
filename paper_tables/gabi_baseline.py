"""GABI baseline: training budgets, graph ablation and integrity checks (Tables D.3 and D.5, Appendices E and H).

Rebuilt from
  results/sensor_completion/gabi/        three-seed GABI runs on straight-line and along-mesh graphs
                                         (test_comparison.json), their config.json and export.json, and the
                                         independent recomputation of the 20-sensor predictions
                                         (independent_verification.json);
  results/sensor_completion/nine_angle/  AREALS three-seed results with the GABI - AREALS contrast
                                         (test_comparison.json), the single-model geometry-only and true-angle
                                         references (references_test_grouped.json) and the seed receipts;
  results/gabi_baseline_extra/           the frozen GABI and AREALS protocols, the pilot parity check of the
                                         three-seed GABI runs and the leak check of the single-seed GABI run.
Code: experiments/07_gabi_baseline/.
"""
from __future__ import annotations

import math
import re

import numpy as np

from paper_tables._core import RESULTS, Table, fmt, load_json

GABI = ("sensor_completion", "gabi")
NINE = ("sensor_completion", "nine_angle")
EXTRA = ("gabi_baseline_extra",)

# Table D.5 rows in paper order: (label, setting key written by field_benchmark.py, paper values).
# Paper values: geometry only, measurements (mean, s.d.), true angle, straight line, along mesh (mean, s.d.),
# along mesh - straight line (difference, low, high), GABI - AREALS (difference, low, high).
D5_ROWS = [
    ("Nothing", "unobserved",
     (0.302, 0.300, 0.001, 0.137, 0.914, 0.858, 0.037, -0.056, -0.088, -0.027, 0.558, 0.422, 0.696)),
    ("5 sensors", "sensors-5",
     (0.302, 0.211, 0.004, 0.137, 0.834, 0.745, 0.104, -0.089, -0.160, -0.015, 0.534, 0.447, 0.617)),
    ("20 sensors", "sensors-20",
     (0.302, 0.164, 0.003, 0.137, 0.736, 0.642, 0.155, -0.094, -0.156, -0.027, 0.479, 0.402, 0.556)),
    ("100 sensors", "sensors-100",
     (0.304, 0.143, 0.001, 0.138, 0.648, 0.537, 0.117, -0.112, -0.146, -0.077, 0.393, 0.342, 0.442)),
    ("500 sensors", "sensors-500",
     (0.310, 0.141, 0.002, 0.143, 0.623, 0.515, 0.113, -0.108, -0.140, -0.076, 0.374, 0.322, 0.424)),
    ("25% hidden", "patch-64-0.25",
     (0.295, 0.134, 0.003, 0.137, 0.631, 0.573, 0.121, -0.058, -0.100, -0.014, 0.440, 0.370, 0.511)),
    ("60% hidden", "patch-64-0.60",
     (0.293, 0.127, 0.002, 0.126, 0.651, 0.577, 0.122, -0.074, -0.114, -0.032, 0.450, 0.389, 0.512)),
    ("90% hidden", "patch-64-0.90",
     (0.302, 0.153, 0.003, 0.138, 0.702, 0.627, 0.141, -0.076, -0.120, -0.027, 0.474, 0.399, 0.550)),
]

CHECK_LINE = re.compile(r"^(PASS|FAIL) (\S+) (\S+) (\S+): largest change ([0-9.eE+-]+) x field spread; "
                        r"observed values \+1 spread moves the hidden prediction ([+-][0-9.]+) spreads$")


def _arg(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def _single(values) -> float:
    """The common value of a collection that must be constant (NaN if it is not, so every check fails)."""
    values = set(values)
    return float(values.pop()) if len(values) == 1 else float("nan")


def _ceil_sig(value: float, exponent: int, digits: int) -> float:
    """`value` in units of 10**exponent, rounded up to `digits` decimals: how the paper prints an upper bound."""
    scale = 10 ** digits
    return math.ceil(value / 10 ** exponent * scale - 1e-9) / scale


def _signed(value: float, decimals: int = 3) -> str:
    return fmt(value, decimals).replace("-", "−")


def tables() -> list[Table]:
    comparison = load_json(*GABI, "test_comparison.json")
    config = load_json(*GABI, "config.json")
    export = load_json(*GABI, "export.json")
    verification = load_json(*GABI, "independent_verification.json")
    nine = load_json(*NINE, "test_comparison.json")
    references = {s["setting"]: s for s in load_json(*NINE, "references_test_grouped.json")["settings"]}
    areals_protocol = load_json("sensor_completion_extra", "nine_angle", "protocol.json")
    receipts = [load_json(*NINE, f"seed{seed}_training_complete.json") for seed in areals_protocol["seeds"]]
    gabi_protocol = load_json(*EXTRA, "gabi_graph_ablation", "protocol.json")
    pilot = load_json(*EXTRA, "gabi_graph_ablation", "pilot_verification.json")
    check_log = (RESULTS.joinpath(*EXTRA, "gabi_single_seed", "check.log")).read_text(encoding="utf-8")

    # ---------------- shared budget quantities ----------------
    areals_args = list(areals_protocol["training_args"].values())
    areals_epochs = _single([r["epochs"] for r in receipts] + [areals_protocol["epochs"]])
    areals_updates = _single([r["updates"] for r in receipts] + [areals_protocol["updates"]])
    areals_batch = _single(int(_arg(a, "--batch-size")) for a in areals_args)
    areals_queries = _single(int(_arg(a, "--queries")) for a in areals_args)
    areals_cases = areals_protocol["counts"]["train"]
    areals_exposures = areals_cases * areals_epochs
    areals_seeds = len({r["seed"] for r in receipts if r["status"] == "PASS"} & set(nine["seeds"]))

    hyper = config["hyperparameters"]
    gabi_epochs, gabi_batch = hyper["epochs"], hyper["batch_size"]
    gabi_cases = export["cars"]["train"]
    gabi_steps = math.ceil(gabi_cases / gabi_batch)
    gabi_updates = export["iterations"]
    gabi_exposures = gabi_cases * gabi_epochs
    gabi_seeds_per_graph = _single(len(comparison["results"][s]["nrmse"]["paired_seed_results"])
                                   for s in comparison["results"])
    gabi_nodes = export["nodes_per_car"]["max"]

    # ---------------- Table D.3 ----------------
    d3 = Table(
        key="Table D.3",
        title="Training budgets on the nine-angle DoubleDelta split (AREALS per model; GABI's published schedule)",
        columns=["Budget", "AREALS, per model", "GABI, published schedule"],
    )
    paper_d3 = [
        ("Epochs", areals_epochs, gabi_epochs, 40, 455),
        ("Batch size", areals_batch, gabi_batch, 8, 100),
        ("Optimiser updates", areals_updates, gabi_updates, 10800, 10010),
        ("Field-example exposures", areals_exposures, gabi_exposures, 86400, 982800),
    ]
    for label, a, g, pa, pg in paper_d3:
        d3.check(f"{label}, AREALS", a, pa, 0)
        d3.check(f"{label}, GABI", g, pg, 0)
        d3.rows.append([label, f"{a:,.0f}", f"{g:,.0f}"])
    d3.check("Training seeds, AREALS", areals_seeds, 3, 0)
    d3.check("Training seeds per graph, GABI", gabi_seeds_per_graph, 3, 0)
    d3.rows.append(["Training seeds", f"{areals_seeds}", f"{gabi_seeds_per_graph:.0f} per graph"])
    ratio = d3.check("caption: GABI / AREALS field-example exposures", gabi_exposures / areals_exposures, 11.375, 3)
    d3.check("caption: GABI field points reconstructed", gabi_nodes, 32000, 0)
    d3.check("caption: AREALS queries per masked example", areals_queries, 2048, 0)
    d3.check("caption: GABI batch size of the published schedule", gabi_batch, 100, 0)
    # The published 10,000-update target is an export setting, not a stored result. What is stored is the schedule
    # it produced: the whole number of epochs that reaches the target. Check that the printed target gives it.
    d3.check("caption: 10,000 published updates -> whole epochs at the stored batches per epoch (stored epochs)",
             gabi_epochs, math.ceil(10000 / gabi_steps), 0)
    d3.check("caption: here GABI epochs", gabi_epochs, 455, 0)
    d3.rows.append(["Exposure ratio (caption)", "", f"{ratio:.3f}x"])
    d3.notes.append(
        f"AREALS: epochs and updates from the three seed receipts (nine_angle/seed*_training_complete.json) and the "
        f"protocol; batch size and queries from its training_args; exposures = {areals_cases:,} training "
        f"cases x epochs. GABI: epochs and batch size from gabi/config.json; updates from gabi/export.json "
        f"(= {gabi_epochs} x ceil({gabi_cases:,} / {gabi_batch}) = "
        f"{gabi_epochs * gabi_steps:,}); exposures = {gabi_cases:,} x epochs.")
    d3.notes.append(
        f"Caption's 10,000 updates: the target of GABI's published schedule is the export default "
        f"(--iterations 10000 in prepare_gabi.py and gabi_adapter.py export) and is not stored in results/. The stored schedule is its "
        f"rounding up to whole epochs ({gabi_epochs} epochs x {gabi_steps} batches = {gabi_updates:,} updates), so "
        f"the check confirms consistency: any target from {(gabi_epochs - 1) * gabi_steps + 1:,} to "
        f"{gabi_epochs * gabi_steps:,} gives the stored {gabi_epochs} epochs.")

    # ---------------- Table D.5 ----------------
    d5 = Table(
        key="Table D.5",
        title="GABI and AREALS on identical test observation sets: hidden-region nRMSE, means of three seeds "
        "(± s.d.); 95% intervals resample the 16 physical wings",
        columns=["Observed", "Geometry only", "Measurements", "True angle", "Straight line", "Along mesh",
                 "Along mesh − straight line", "GABI − AREALS"],
    )
    improved = 0
    for label, key, paper in D5_ROWS:
        gabi = comparison["results"][key]["nrmse"]
        contrast = nine["results"][key]["nrmse"]
        reference = references[key]["nrmse"]
        along_sd = float(np.std([p["surface"] for p in gabi["paired_seed_results"]], ddof=1))
        values = (
            reference["geometry"]["mean"],
            contrast["areals"]["mean"], contrast["areals"]["seed_sd"],
            reference["conditioned"]["mean"],
            gabi["euclidean"], gabi["surface"], along_sd,
            gabi["difference"], *gabi["geometry_bootstrap_difference_95ci"],
            contrast["gabi_minus_areals"], *contrast["geometry_bootstrap_difference_95ci"],
        )
        names = ("geometry only", "measurements", "measurements s.d.", "true angle", "straight line",
                 "along mesh", "along mesh s.d.", "along - straight", "along - straight, low",
                 "along - straight, high", "GABI - AREALS", "GABI - AREALS, low", "GABI - AREALS, high")
        for name, value, printed in zip(names, values, paper, strict=True):
            d5.check(f"{label}: {name}", value, printed, 3)
        improved += gabi["difference"] < 0
        v = values
        d5.rows.append([
            label, fmt(v[0]), f"{fmt(v[1])} ± {fmt(v[2])}", fmt(v[3]), fmt(v[4]), f"{fmt(v[5])} ± {fmt(v[6])}",
            f"{_signed(v[7])} [{_signed(v[8])}, {_signed(v[9])}]", f"{_signed(v[10])} [{_signed(v[11])}, {_signed(v[12])}]",
        ])
    d5.check("caption: physical wings resampled", comparison["geometries"], 16, 0)
    d5.check("caption: patch cells", _single(int(k.split("-")[1]) for k in comparison["results"] if k.startswith("patch")),
             64, 0)
    d5.check("caption: seeds averaged (GABI and AREALS)", _single([len(comparison["seeds"]), len(nine["seeds"])]), 3, 0)
    d5.check("caption: nearest vertices joined to each point (both graphs)",
             int(re.search(r"(\d+)-neighbour graphs", gabi_protocol["objective"])[1]), 8, 0)
    interval_keys = {k for s in comparison["results"].values() for k in s["nrmse"] if k.endswith("ci")}
    interval_keys |= {k for s in nine["results"].values() for k in s["nrmse"] if k.endswith("ci")}
    d5.check("caption: interval level (%)", _single(int(re.search(r"_(\d+)ci$", k)[1]) for k in interval_keys), 95, 0)
    # The along-mesh - straight-line intervals are rebuilt from the stored per-wing differences, as analyze.py does:
    # one 10,000 x 16 resampling of the wings (rng 20261028), 2.5% and 97.5% quantiles of the resampled means.
    deviation = 0.0
    for setting in comparison["results"].values():
        for metric in ("nrmse", "force_rel"):
            wings = setting[metric]["geometry_differences"]
            delta = np.asarray([wings[w] for w in sorted(wings)])
            draws = np.random.default_rng(20261028).integers(0, len(delta), (10000, len(delta)))
            rebuilt = np.quantile(delta[draws].mean(1), [0.025, 0.975])
            deviation = max(deviation, float(np.abs(rebuilt - setting[metric]["geometry_bootstrap_difference_95ci"]).max()),
                            abs(float(delta.mean()) - setting[metric]["difference"]))
    d5.notes.append(f"Along mesh − straight line: differences and intervals rebuilt from the stored per-wing differences "
                    f"(geometry_differences, rng 20261028, 10,000 draws) agree with the stored values to {deviation:.1e}.")
    d5.notes.append("Geometry only and true angle: geometry / conditioned means in nine_angle/references_test_grouped.json "
                    "(single models). Measurements: AREALS mean and seed s.d. in nine_angle/test_comparison.json. "
                    "Straight line / along mesh: euclidean / surface in gabi/test_comparison.json; the along-mesh s.d. "
                    "is the sample s.d. (ddof 1) of its paired per-seed means. Along mesh − straight line: its "
                    "difference and 10,000-draw wing bootstrap (rng 20261028). GABI − AREALS: gabi_minus_areals and "
                    "its 10,000-draw wing bootstrap (rng 20261029) in nine_angle/test_comparison.json.")

    # ---------------- Appendix E, GABI paragraph ----------------
    e = Table(
        key="Appendix E (GABI)",
        title="GABI architecture, training and inference settings, and the along-mesh graph's effect",
        columns=["Quantity", "Rebuilt", "Source"],
    )
    abc = next(item for item in gabi_protocol["fixed"] if item.startswith("ABC "))
    match = re.match(r"ABC ([\d,]+) normal latent draws, (\d+) accepted, noise sigma ([\d.]+)", abc)
    samples, accepted, sigma = float(match[1].replace(",", "")), float(match[2]), float(match[3])
    primary = comparison["results"]["sensors-20"]
    field_drop = -100 * primary["nrmse"]["relative_change"]
    force_drop = -100 * primary["force_rel"]["relative_change"]
    model = config["model_settings"]
    e_rows = [
        ("Graph-convolution layers", model["n_layers"], 6, 0, "gabi/config.json model_settings.n_layers"),
        ("Width", model["latent_dim"], 100, 0, "gabi/config.json model_settings.latent_dim"),
        ("Latent dimension", model["z_dim"], 100, 0, "gabi/config.json model_settings.z_dim"),
        ("Points per geometry", gabi_nodes, 32000, 0, "gabi/export.json nodes_per_car"),
        ("Nearest vertices per point", export["neighbours"], 8, 0, "gabi/export.json neighbours"),
        ("Seeds per graph", len(gabi_protocol["seeds"]), 3, 0, "gabi_baseline_extra/gabi_graph_ablation/protocol.json"),
        ("Epochs", gabi_epochs, 455, 0, "gabi/config.json"),
        ("Updates", gabi_updates, 10010, 0, "gabi/export.json iterations"),
        ("Batch size", gabi_batch, 100, 0, "gabi/config.json"),
        ("Latent samples per geometry", samples, 50000, 0, "protocol.json 'fixed' (ABC)"),
        ("Angle cases sharing a pool", verification["angles_per_geometry"], 9, 0,
         "gabi/independent_verification.json"),
        ("Accepted samples", accepted, 100, 0, "protocol.json 'fixed' (ABC)"),
        ("Observation noise (normalised units)", sigma, 0.01, 2, "protocol.json 'fixed' (ABC)"),
        ("Along-mesh nRMSE reduction at 20 sensors (%)", field_drop, 12.7, 1,
         "-relative_change, gabi/test_comparison.json"),
        ("Along-mesh force-error reduction at 20 sensors (%)", force_drop, 69, 0,
         "-relative_change of force_rel, gabi/test_comparison.json"),
        ("Settings where the along-mesh graph lowers nRMSE (paper: every one of Table D.5's 8)", improved, 8, 0,
         "sign of the difference, gabi/test_comparison.json"),
    ]
    for label, value, printed, decimals, source in e_rows:
        e.check(label, value, printed, decimals)
        e.rows.append([label, f"{value:,.{decimals}f}", source])

    # ---------------- Appendix H, GABI controls ----------------
    h = Table(
        key="Appendix H (GABI)",
        title="GABI benchmark controls: hidden-value perturbation, measurement response and independent recomputation",
        columns=["Control", "Rebuilt", "Source"],
    )
    lines = [CHECK_LINE.match(line) for line in check_log.splitlines() if line.startswith(("PASS ", "FAIL "))]
    assert all(lines), "unparsed check line"
    changes = [float(m[5]) for m in lines]
    passed = sum(m[1] == "PASS" for m in lines)
    summary = next(line for line in check_log.splitlines() if line.startswith("response to the observed values"))
    responses = dict(re.findall(r"(\S+) ([+-][0-9.]+)", summary.split(":", 1)[1]))
    response20 = float(responses["sensors-20"])
    per_check20 = float(np.mean([float(m[6]) for m in lines if m[4] == "sensors-20"]))
    max_difference = max(verification["maximum_difference_from_recorded_scores"].values())
    parity = pilot["real_geometry_abc_parity"]
    h.check("hidden-value checks recorded", len(lines), 6, 0)
    h.check("hidden-value checks passed", passed, 6, 0)
    h.check("largest change, x1e-6 field spread, rounded up (paper: below 8e-6)", _ceil_sig(max(changes), -6, 0), 8, 0)
    h.check("measurement response at 20 sensors (spreads)", response20, 0.10, 2)
    h.check("independent recomputation: predictions", verification["predictions_checked"], 864, 0)
    h.check("independent recomputation: runs", len(verification["verified_runs"]), 6, 0)
    h.check("independent recomputation: sensors", int(re.search(r"(\d+)-sensor", verification["scope"])[1]), 20, 0)
    h.check("independent recomputation: max |difference|, x1e-14, rounded up to 2 s.f. (paper: within 3.4e-14)",
            _ceil_sig(max_difference, -14, 1), 3.4, 1)
    wings = sorted({m[3] for m in lines})
    h.rows += [
        [f"Hidden-value perturbation (single-seed straight-line GABI, {len(wings)} wings x "
         f"{len(lines) // len(wings)} settings)",
         f"{passed}/{len(lines)} pass; largest change {max(changes):.2e} x field spread", "gabi_baseline_extra/gabi_single_seed/check.log"],
        ["Measurement response, +1 spread on observed values, 20 sensors",
         f"{response20:+.2f} spreads (per-check mean {per_check20:+.2f})", "gabi_baseline_extra/gabi_single_seed/check.log"],
        ["Pilot of the three-seed runs: our shared-pool ABC vs the authors' abc_inference, max difference",
         ", ".join(f"{k} {v['author_abc_max_difference']:.2e}" for k, v in parity.items()),
         "gabi_baseline_extra/gabi_graph_ablation/pilot_verification.json"],
        ["Pilot: hidden node values scrambled, max change in prediction",
         ", ".join(f"{k} {v['hidden_node_values_max_difference']:.2e}" for k, v in parity.items()),
         "gabi_baseline_extra/gabi_graph_ablation/pilot_verification.json"],
        ["Independent recomputation of the six runs' 20-sensor test scores",
         f"{verification['predictions_checked']} predictions; max abs. difference nRMSE "
         f"{verification['maximum_difference_from_recorded_scores']['nrmse']:.2e}, force "
         f"{verification['maximum_difference_from_recorded_scores']['force_rel']:.2e}",
         "gabi/independent_verification.json"],
    ]
    h.notes.append(f"The paper prints the two maxima as upper bounds; the checks round the stored maxima up "
                   f"({max(changes):.3g} -> {_ceil_sig(max(changes), -6, 0):g}e-6; "
                   f"{max_difference:.4g} -> {_ceil_sig(max_difference, -14, 1):g}e-14).")
    h.notes.append(f"Pilot status {pilot['status']}: one {pilot['epochs_per_variant']}-epoch run per graph at batch "
                   f"{pilot['full_batch_size']}, paired initial weights: {pilot['paired_initial_weights']}, paired "
                   f"example orders: {pilot['paired_example_orders']}.")
    return [d3, d5, e, h]
