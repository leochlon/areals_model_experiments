"""Expanded benchmarks: Table 2 (rows 2-3), Table C.1, Table C.5 (rows 1-3), the related text of
Sections 4.2 and 4.4, and the Appendix C field-training protocol.

Every value is read from results/expanded_benchmarks/ (written by the study's aggregate.py and protocol.py)
and results/expanded_benchmarks_extra/data_readiness.json (split counts written by the data preparation
and data_checks.py). Code: experiments/08_expanded_benchmarks/.
"""
from __future__ import annotations

import math
import re

from paper_tables._core import Table, fmt, load_json

SENSORS = [5, 20, 100, 500]
BENCHMARKS = {  # stored name -> label used in the paper
    "doubledelta": "DoubleDelta",
    "blendednet": "BlendedNet++",
    "drivaerml": "DrivAerML",
    "drivaernet": "DrivAerNet++",
}


def grouped(name: str) -> dict:
    return load_json("expanded_benchmarks", f"{name}_test_grouped.json")


def nrmse(g: dict, k: int, arm: str) -> float:
    return g["settings"][f"sensors-{k}"][arm]["nrmse"]


def interval(g: dict, k: int) -> list[float]:
    return g["settings"][f"sensors-{k}"]["pretrained_minus_scratch"]["geometry_grouped_95_interval"]


def reduction(g: dict, k: int) -> float:
    """Relative error reduction of AREALS against scratch, in percent."""
    return 100.0 * (1.0 - nrmse(g, k, "pretrained") / nrmse(g, k, "scratch"))


def gap_closed(g: dict, k: int) -> float:
    """Share of the shape-only to true-condition gap closed by the sensor model, in percent."""
    shape, cond = nrmse(g, k, "shape"), nrmse(g, k, "conditioned")
    return 100.0 * (shape - nrmse(g, k, "pretrained")) / (shape - cond)


def fmt_count(value: float, decimals: int) -> str:
    """Counts with thousands separators, other values as the paper prints them."""
    return f"{int(round(value)):,}" if decimals == 0 and float(value).is_integer() else fmt(value, decimals)


def fmt_interval(lo: float, hi: float, decimals: int = 3) -> str:
    return f"[{fmt(lo, decimals)}, {fmt(hi, decimals)}]".replace("-", "−")


def readiness(name: str) -> dict:
    return load_json("expanded_benchmarks_extra", "data_readiness.json")["datasets"][name]


# ---------------------------------------------------------------------------------------------------------------
# Table 2, rows 2-3
# ---------------------------------------------------------------------------------------------------------------
TABLE2_PAPER = {  # Scratch, AREALS, Shape, Condition at 20 sensors
    "doubledelta": ("DoubleDelta, 18–19°", (0.409, 0.206, 0.373, 0.193)),
    "blendednet": ("BlendedNet++", (0.212, 0.176, 0.743, 0.066)),
}


def table_2() -> Table:
    t = Table("Table 2 (rows 2–3)",
              "Sensor-conditioned comparison at 20 sensors with the condition unreported: hidden-region nRMSE",
              ["Benchmark", "GABI", "Scratch", "AREALS", "Shape", "Condition"])
    for name, (label, paper) in TABLE2_PAPER.items():
        g = grouped(name)
        row = [label, "—"]
        for arm, p in zip(["scratch", "pretrained", "shape", "conditioned"], paper):
            row.append(fmt(t.check(f"{label} {arm}", nrmse(g, 20, arm), p, 3)))
        t.rows.append(row)
    t.notes.append("Means of three seeds; GABI was run on the nine-angle split only (row 1, not part of this study).")
    return t


# ---------------------------------------------------------------------------------------------------------------
# Table C.1
# ---------------------------------------------------------------------------------------------------------------
C1_TEST_SET = {  # paper test-set size (printed in the row label)
    "doubledelta": 16, "blendednet": 1872, "drivaerml": 50, "drivaernet": 848,
}
C1_PAPER = {  # K -> (Shape only, Scratch, AREALS, True condition, interval low, interval high)
    "doubledelta": {5: (0.373, 0.409, 0.310, 0.193, -0.157, -0.046),
                    20: (0.373, 0.409, 0.206, 0.193, -0.235, -0.170),
                    100: (0.375, 0.411, 0.178, 0.194, -0.258, -0.210),
                    500: (0.383, 0.420, 0.187, 0.200, -0.258, -0.208)},
    "blendednet": {5: (0.742, 0.281, 0.240, 0.066, -0.044, -0.037),
                   20: (0.743, 0.212, 0.176, 0.066, -0.037, -0.034),
                   100: (0.745, 0.191, 0.145, 0.067, -0.046, -0.044),
                   500: (0.755, 0.189, 0.138, 0.068, -0.052, -0.050)},
    "drivaerml": {5: (0.240, 0.294, 0.241, 0.240, -0.056, -0.050),
                  20: (0.240, 0.295, 0.242, 0.241, -0.056, -0.050),
                  100: (0.242, 0.296, 0.243, 0.242, -0.056, -0.050),
                  500: (0.248, 0.303, 0.249, 0.248, -0.057, -0.051)},
    "drivaernet": {5: (0.204, 0.208, 0.204, 0.204, -0.004, -0.003),
                   20: (0.204, 0.208, 0.204, 0.204, -0.004, -0.003),
                   100: (0.204, 0.208, 0.204, 0.204, -0.004, -0.003),
                   500: (0.204, 0.207, 0.204, 0.204, -0.004, -0.003)},
}


def test_set_label(name: str, families: int) -> str:
    if name == "doubledelta":
        angles = load_json("expanded_benchmarks", "protocol.json")["experiments"]["doubledelta"]["test_angles"]
        return f"{families} wings, {min(angles)}–{max(angles)}°"
    if name == "blendednet":
        return f"{families:,} aircraft, high Mach"
    if name == "drivaerml":
        return f"{families} official test cars"
    return f"{families} test cars"


def table_c1() -> Table:
    t = Table("Table C.1", "Expanded benchmarks, three seeds: hidden-region nRMSE at each sensor count",
              ["Benchmark", "Test set", "K", "Shape only", "Scratch", "AREALS", "True condition", "AREALS − scratch"])
    for name, label in BENCHMARKS.items():
        g = grouped(name)
        families = t.check(f"{label} test geometries", g["independent_geometry_families"], C1_TEST_SET[name], 0)
        for i, k in enumerate(SENSORS):
            shape, scratch, areals, cond, lo, hi = C1_PAPER[name][k]
            cells = [fmt(t.check(f"{label} K={k} {arm}", nrmse(g, k, arm), p, 3))
                     for arm, p in zip(["shape", "scratch", "pretrained", "conditioned"], [shape, scratch, areals, cond])]
            ci_lo, ci_hi = interval(g, k)
            t.check(f"{label} K={k} interval low", ci_lo, lo, 3)
            t.check(f"{label} K={k} interval high", ci_hi, hi, 3)
            t.rows.append([label if i == 0 else "", test_set_label(name, int(families)) if i == 0 else "", str(k),
                           *cells, fmt_interval(ci_lo, ci_hi)])
    t.check("seeds per arm (all benchmarks)", min(len(grouped(n)["seeds"]) for n in BENCHMARKS), 3, 0)
    excluded = sum(interval(grouped(n), k)[1] < 0 for n in BENCHMARKS for k in SENSORS)
    t.notes.append(f"{excluded} of {len(BENCHMARKS) * len(SENSORS)} paired intervals lie entirely below zero.")
    constant = [BENCHMARKS[n] for n in BENCHMARKS
                 if readiness(n)["checks"]["condition_statistics"]["names"] == ["constant_flow"]]
    t.notes.append(f"{' and '.join(constant)} have a single inflow condition: the true-condition arm receives a "
                   "constant input and acts as a second shape-only fit.")
    return t


# ---------------------------------------------------------------------------------------------------------------
# Table C.5, rows 1-3
# ---------------------------------------------------------------------------------------------------------------
def train_size(name: str) -> int:
    protocol = load_json("expanded_benchmarks", "protocol.json")["experiments"]
    if name == "doubledelta":
        return protocol["doubledelta"]["geometry_families"][0]  # physical wing families (six angles each)
    if name == "drivaerml":
        return protocol["drivaerml"]["official_split"][0]
    return readiness(name)["data_ready"]["families"]["train"]


C5_PAPER = {  # Train, Scratch, AREALS, Reduction (%)
    "doubledelta": ("DoubleDelta, 18–19°", 228, 0.409, 0.206, 49.6),
    "blendednet": ("BlendedNet++, high Mach", 8736, 0.212, 0.176, 16.8),
    "drivaerml": ("DrivAerML, official", 400, 0.295, 0.242, 18.0),
}


def table_c5() -> Table:
    t = Table("Table C.5 (rows 1–3)",
              "Geometry pretraining against the same architecture from scratch at 20 sensors, three seeds",
              ["Benchmark", "Train", "Scratch", "AREALS", "Reduction"])
    for name, (label, train, scratch, areals, red) in C5_PAPER.items():
        g = grouped(name)
        n = t.check(f"{label} train", train_size(name), train, 0)
        s = t.check(f"{label} scratch", nrmse(g, 20, "scratch"), scratch, 3)
        a = t.check(f"{label} AREALS", nrmse(g, 20, "pretrained"), areals, 3)
        r = t.check(f"{label} reduction (%)", reduction(g, 20), red, 1)
        t.rows.append([label, f"{int(n):,}", fmt(s), fmt(a), f"{fmt(r, 1)}%"])
    stored = load_json("expanded_benchmarks", "latest_completed_results.json")
    for name in ["doubledelta", "drivaerml"]:
        value = stored[name]["settings"]["sensors-20"]["reduction_vs_scratch_percent"]
        t.notes.append(f"{BENCHMARKS[name]}: stored reduction_vs_scratch_percent {value:.3f} "
                       f"(rebuilt {reduction(grouped(name), 20):.3f}).")
    t.notes.append("Row 4 (DrivAerML, reserved) comes from a different study and is not rebuilt here.")
    return t


# ---------------------------------------------------------------------------------------------------------------
# Text of Sections 4.2 and 4.4
# ---------------------------------------------------------------------------------------------------------------
def table_text() -> Table:
    t = Table("Sections 4.2 and 4.4", "Numbers quoted in the text from the expanded benchmarks",
              ["Claim", "Rebuilt", "Paper"])

    def row(label, value, paper, decimals, unit=""):
        t.check(label, value, paper, decimals)
        shown = fmt(value, decimals) if unit else fmt_count(value, decimals)
        t.rows.append([label, f"{shown}{unit}", f"{fmt_count(paper, decimals)}{unit}"])

    ml, net = grouped("drivaerml"), grouped("drivaernet")
    dd, bn = grouped("doubledelta"), grouped("blendednet")
    # Section 4.2
    row("4.2 DrivAerML official training cars", train_size("drivaerml"), 400, 0)
    row("4.2 DrivAerML error reduction at 20 sensors", reduction(ml, 20), 18.0, 1, "%")
    row("4.2 DrivAerNet++ training cars", train_size("drivaernet"), 6492, 0)
    row("4.2 DrivAerNet++ error reduction at 20 sensors", reduction(net, 20), 2, 0, "%")
    # Section 4.4, DrivAerML: sensor model against the shape-only model
    gap = max(abs(nrmse(ml, k, "pretrained") - nrmse(ml, k, "shape")) for k in SENSORS)
    row("4.4 DrivAerML: AREALS within this nRMSE of shape-only at every K (largest gap, rounded up)",
        math.ceil(gap * 1000 - 1e-9) / 1000, 0.002, 3)
    t.notes.append(f"DrivAerML: the largest gap between AREALS and the shape-only model over the four sensor counts "
                   f"is {gap:.4f} nRMSE.")
    # Section 4.4, DoubleDelta beyond the training range
    row("4.4 DoubleDelta 18–19°: gap closed at 20 sensors", gap_closed(dd, 20), 93, 0, "%")
    scratch = [nrmse(dd, k, "scratch") for k in SENSORS]
    row("4.4 DoubleDelta scratch error, lowest over K", min(scratch), 0.41, 2)
    row("4.4 DoubleDelta scratch error, highest over K", max(scratch), 0.42, 2)
    row("4.4 DoubleDelta reduction at 20 sensors ('halves')", reduction(dd, 20), 50, 0, "%")
    row("4.4 DoubleDelta reduction at 100 sensors", reduction(dd, 100), 57, 0, "%")
    below = [k for k in SENSORS if nrmse(dd, k, "pretrained") < nrmse(dd, k, "conditioned")]
    # "from 100 sensors the error of AREALS is below that of the true-condition model": smallest K from which
    # AREALS stays below the true-condition model at every larger sensor count.
    below_from = min((k for k in SENSORS if all(j in below for j in SENSORS if j >= k)), default=float("nan"))
    row("4.4 DoubleDelta 18–19°: AREALS below the true-condition model from K =", below_from, 100, 0)
    above_shape = all(nrmse(dd, k, "scratch") > nrmse(dd, k, "shape") for k in SENSORS)
    t.notes.append(f"DoubleDelta: AREALS is below the true-condition model at K = {below}; scratch is above the "
                   f"shape-only model at every K: {above_shape}.")
    # Section 4.4, BlendedNet++
    row("4.4 BlendedNet++: gap closed at 20 sensors", gap_closed(bn, 20), 84, 0, "%")
    reductions = [reduction(bn, k) for k in SENSORS]
    row("4.4 BlendedNet++ reduction vs scratch, lowest over K = 5–500", min(reductions), 14, 0, "%")
    row("4.4 BlendedNet++ reduction vs scratch, highest over K = 5–500", max(reductions), 27, 0, "%")
    t.notes.append("BlendedNet++ reductions by K: "
                   + ", ".join(f"K={k} {r:.2f}%" for k, r in zip(SENSORS, reductions)) + ".")
    return t


# ---------------------------------------------------------------------------------------------------------------
# Appendix C protocol paragraph and splits
# ---------------------------------------------------------------------------------------------------------------
def table_protocol() -> Table:
    t = Table("Appendix C protocol", "Field-training protocol and splits of the expanded benchmarks (protocol.json)",
              ["Setting", "Stored", "Paper"])
    protocol = load_json("expanded_benchmarks", "protocol.json")
    ft, ex = protocol["field_training"], protocol["experiments"]

    def row(label, value, paper, decimals=0):
        t.check(label, value, paper, decimals)
        t.rows.append([label, fmt_count(value, decimals), fmt_count(paper, decimals)])

    row("surface field points per case", ft["field_points"], 32000)
    row("geometry samples per case", ft["geometry_samples"], 16384)
    row("queries per example", ft["queries"], 2048)
    row("batch size", ft["batch_size"], 8)
    row("epochs", ft["epochs"], 40)
    row("learning rate, all parameters (log10)", math.log10(ft["lr"]), -3)
    row("learning rate, new field parameters (log10)", math.log10(ft["new_lr"]), -3)
    row("weight decay", ft["weight_decay"], 0.05, 2)
    row("geometry tokens", ft["geometry_tokens"], 256)
    row("encoder points", ft["encoder_points"], 4096)
    row("examples observing nothing (%)", 100 * ft["sensor_training_mix"]["nothing_observed"], 10)
    row("examples observing scattered sensors (%)", 100 * ft["sensor_training_mix"]["scattered_sensors"], 90)
    row("sensor counts used for selection", len(ft["inference_sensors"]), 4)
    row("development cases for selection (max)", int(re.search(r"max (\d+)", ft["selection"]).group(1)), 64)
    low, high = (float(x) for x in re.findall(r"0\.\d+", ex["blendednet"]["condition_bands"]))
    row("BlendedNet++ training Mach band (percentile)", 100 * low, 70)
    row("BlendedNet++ validation Mach band (percentile)", 100 * high, 85)
    for split, part, paper in zip(["train", "val", "test"], ex["drivaernet"]["geometry_hash_folds"], [80, 10, 10]):
        row(f"DrivAerNet++ geometry-hash fold, {split} (%)", part, paper)
    for part, paper in zip(["train", "val", "test"], [400, 34, 50]):
        row(f"DrivAerML official split, {part}", readiness("drivaerml")["data_ready"]["counts"][part], paper)
    for part, paper in zip(["train", "val", "test"], [6492, 778, 848]):
        row(f"DrivAerNet++ split, {part}", readiness("drivaernet")["data_ready"]["counts"][part], paper)
    for part, paper in zip(["train", "test"], [8736, 1872]):
        row(f"BlendedNet++ aircraft, {part}", readiness("blendednet")["data_ready"]["families"][part], paper)
    angles = ex["doubledelta"]["test_angles"]
    row("DoubleDelta test angle, lowest (°)", min(angles), 18)
    row("DoubleDelta test angle, highest (°)", max(angles), 19)
    t.notes.append("Splits from the data-preparation receipts (results/expanded_benchmarks_extra/data_readiness.json); "
                   "every check there passed: " + str(all(readiness(n)["checks"]["passed"] for n in BENCHMARKS)) + ".")
    return t


# ---------------------------------------------------------------------------------------------------------------
# Table A.1, rows of the expanded benchmarks
# ---------------------------------------------------------------------------------------------------------------
A1_PAPER = {  # stored name -> (row label, training count, test count, seeds)
    "doubledelta": ("Extrapolated angles (Tables 2, C.1 and C.5)", 228, 16, 3),
    "blendednet": ("BlendedNet++ (Tables 2, C.1 and C.5)", 8736, 1872, 3),
    "drivaerml": ("DrivAerML official split (Tables C.1 and C.5)", 400, 50, 3),
    "drivaernet": ("DrivAerNet++ (Tables C.1 and C.5)", 6492, 848, 3),
}


def table_a1() -> Table:
    t = Table("Table A.1 (expanded-benchmark rows)", "Pretrained model, data, observations and seeds of each result",
              ["Experiment", "Pretrained", "Field training", "Test", "Observations", "Seeds"])
    ex = load_json("expanded_benchmarks", "protocol.json")["experiments"]
    for name, (label, train, test, seeds) in A1_PAPER.items():
        g = grouped(name)
        n_train = t.check(f"{label} training count", train_size(name), train, 0)
        n_test = t.check(f"{label} test count", g["independent_geometry_families"], test, 0)
        counts = sorted(int(s.split("-")[1]) for s in g["settings"] if s.startswith("sensors-"))
        lo = t.check(f"{label} fewest sensors", counts[0], 5, 0)
        hi = t.check(f"{label} most sensors", counts[-1], 500, 0)
        n_seeds = t.check(f"{label} seeds", len(g["seeds"]), seeds, 0)
        if name == "doubledelta":
            train_angles, test_angles = ex["doubledelta"]["train_angles"], ex["doubledelta"]["test_angles"]
            a0 = t.check(f"{label} training angle, lowest", min(train_angles), 11, 0)
            a1 = t.check(f"{label} training angle, highest", max(train_angles), 16, 0)
            b0 = t.check(f"{label} test angle, lowest", min(test_angles), 18, 0)
            b1 = t.check(f"{label} test angle, highest", max(test_angles), 19, 0)
            train_text = f"{int(n_train):,} wing families, {a0}–{a1}°"
            test_text = f"{int(n_test)} wings at {b0}–{b1}°"
        elif name == "blendednet":
            train_text, test_text = f"{int(n_train):,} aircraft", f"{int(n_test):,} unseen aircraft, higher Mach"
        else:
            train_text, test_text = f"{int(n_train):,} cars", f"{int(n_test):,} test cars"
        t.rows.append([label, "Production vs. none", train_text, test_text, f"{lo}–{hi} sensors", str(n_seeds)])
    t.notes.append("Metric: hidden-region nRMSE in every row.")
    return t


def tables() -> list[Table]:
    return [table_2(), table_c1(), table_c5(), table_text(), table_protocol(), table_a1()]
