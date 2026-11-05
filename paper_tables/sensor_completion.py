"""Sensor completion with the operating condition unreported (Section 4.4, Appendices D, E and H).

Rebuilds the nine-angle row of Table 2, Tables D.1, D.2, D.4 and D.6, the Figure D.1 curves, the rows of
Table A.1 for these studies, and the numbers quoted in the abstract and Sections 1, 4.4 and 5, from the stored
result files of experiments/06_sensor_completion:

    results/sensor_completion/nine_angle/        three fine-tuning seeds, seed-0 references, from-scratch run
    results/sensor_completion/condition_inference/  explicit condition inference vs AREALS (FP32)
    results/sensor_completion/heldout_angles/    angles held out of training and development
    results/sensor_completion_extra/heldout_angles/  development-split summary and the measurement-response check
    results/sensor_completion_extra/nine_angle/  protocol of the three fine-tuning seeds (training arguments)

Some cells use files stored for other experiments: the true-angle and shape-only force errors and the scored
area at 500 sensors come from the per-problem scores in results/sensor_completion/gabi/; the BlendedNet++
cell of Table D.1 and the training set of the 18-19 deg true-condition model (Table A.1) come from
results/expanded_benchmarks/.
"""
from __future__ import annotations

import math

import numpy as np

from paper_tables._core import Table, as_printed, fmt, load_csv, load_json

SENSORS = [5, 20, 100, 500]


# ----------------------------------------------------------------------------------------------- loaders

def _nine_angle():
    """Three-seed AREALS and GABI comparison on the 144 nine-angle test cases."""
    return load_json("sensor_completion", "nine_angle", "test_comparison.json")


def _references():
    """Seed-0 geometry-only, measurement and true-angle models, by observation setting."""
    data = load_json("sensor_completion", "nine_angle", "references_test_grouped.json")
    return {s["setting"]: s for s in data["settings"]}


def _scratch():
    """Seed-0 model trained from random initialisation, by observation setting."""
    data = load_json("sensor_completion", "nine_angle", "scratch_test_grouped.json")
    return {s["setting"]: s for s in data["settings"]}


def _per_problem_mean(rows, method, setting, column):
    values = [float(r[column]) for r in rows if r["method"] == method and r["setting"] == setting]
    return float(np.mean(values)) if values else float("nan")


def _heldout(split):
    if split == "test":
        data = load_json("sensor_completion", "heldout_angles", "test_grouped.json")
    else:
        data = load_json("sensor_completion_extra", "heldout_angles", "val_grouped.json")
    return data, {s["setting"]: s for s in data["settings"]}


def _protocol():
    """Protocol of the three fine-tuning seeds (written by three_seeds/prepare.py)."""
    return load_json("sensor_completion_extra", "nine_angle", "protocol.json")


def _arg(args, flag):
    """Value of a command-line flag in a stored training argument list."""
    return float(args[args.index(flag) + 1])


def _sensor_counts(settings):
    """Sensor counts of the observation settings; the unobserved setting counts as 0 sensors."""
    return sorted(0 if s == "unobserved" else int(s.split("-")[1])
                  for s in settings if s == "unobserved" or s.startswith("sensors-"))


def _pct(value, decimals=1):
    return f"{fmt(value, decimals)}%"


# ----------------------------------------------------------------------------------------------- tables

def table_2() -> Table:
    nine = _nine_angle()["results"]["sensors-20"]["nrmse"]
    ref = _references()["sensors-20"]["nrmse"]
    scratch = _scratch()["sensors-20"]["nrmse"]["areal_scratch"]["mean"]
    t = Table("Table 2", "Sensor-conditioned comparison at 20 sensors, condition unreported "
              "(hidden-region nRMSE; nine-angle row)",
              ["Benchmark", "GABI", "Scratch", "AREALS", "Shape", "Condition"])
    gabi = t.check("Table 2 nine angles GABI", nine["gabi_surface"]["mean"], 0.642, 3)
    scr = t.check("Table 2 nine angles Scratch", scratch, 0.222, 3)
    areals = t.check("Table 2 nine angles AREALS", nine["areals"]["mean"], 0.164, 3)
    shape = t.check("Table 2 nine angles Shape", ref["geometry"]["mean"], 0.302, 3)
    cond = t.check("Table 2 nine angles Condition", ref["conditioned"]["mean"], 0.137, 3)
    t.rows.append(["DoubleDelta, nine angles", fmt(gabi), fmt(scr), fmt(areals), fmt(shape), fmt(cond)])
    t.notes.append("GABI and AREALS: means of three seeds; Scratch, Shape and Condition: one seed (seed 0). "
                   "The other rows of Table 2 are rebuilt by the expanded-benchmark module.")
    return t


def table_d1() -> Table:
    winning = load_json("sensor_completion", "condition_inference", "winning_comparison.json")["rows"]
    rows = {r["setting"]: r for r in winning if r["dataset"] == "doubledelta" and r["domain"] == "training_support"}
    fp32 = {(s["dataset"], s["setting"]): s
            for s in load_json("sensor_completion", "condition_inference", "matched_fp32_areals.json")["summaries"]}
    nine = _nine_angle()["results"]["sensors-20"]["nrmse"]["areals"]
    blended = load_json("expanded_benchmarks", "blendednet_test_grouped.json")["settings"]["sensors-20"]["pretrained"]

    t = Table("Table D.1", "Explicit condition inference against AREALS, condition unreported: hidden-region nRMSE, "
              "mean ± s.d. over three seeds, 95% bootstrap intervals, median time per case",
              ["Benchmark", "K", "AREALS nRMSE", "95% interval", "Explicit nRMSE", "95% interval",
               "Time per case (s) AREALS / explicit"])

    k5 = t.check("D.1 18-19 K=5 AREALS", fp32[("doubledelta", "sensors-5")]["nrmse"]["mean"], 0.310, 3)
    t.rows.append(["DoubleDelta, 18–19°", "5", fmt(k5), "—", "—", "—", "—"])

    paper = {
        20: dict(a=0.2060, asd=0.0035, alo=0.181, ahi=0.230, e=0.2999, esd=0.0007, elo=0.264, ehi=0.336,
                 ta=0.122, te=0.356),
        100: dict(a=0.1775, asd=0.0061, alo=0.154, ahi=0.202, e=0.2999, esd=0.0023, elo=0.263, ehi=0.337,
                  ta=0.123, te=0.355),
        500: dict(a=0.1872, asd=0.0086, alo=0.163, ahi=0.211, e=0.3066, esd=0.0002, elo=0.268, ehi=0.345,
                  ta=0.123, te=0.354),
    }
    for k, p in paper.items():
        r = rows[f"sensors-{k}"]
        a, e, times = r["areals_nrmse"], r["inference_nrmse"], r["median_seconds"]
        lab = f"D.1 18-19 K={k}"
        am = t.check(f"{lab} AREALS mean", a["mean"], p["a"], 4)
        asd = t.check(f"{lab} AREALS s.d.", a["sample_sd"], p["asd"], 4)
        alo = t.check(f"{lab} AREALS 95% low", a["bootstrap95"][0], p["alo"], 3)
        ahi = t.check(f"{lab} AREALS 95% high", a["bootstrap95"][1], p["ahi"], 3)
        em = t.check(f"{lab} explicit mean", e["mean"], p["e"], 4)
        esd = t.check(f"{lab} explicit s.d.", e["sample_sd"], p["esd"], 4)
        elo = t.check(f"{lab} explicit 95% low", e["bootstrap95"][0], p["elo"], 3)
        ehi = t.check(f"{lab} explicit 95% high", e["bootstrap95"][1], p["ehi"], 3)
        ta = t.check(f"{lab} AREALS time (s)", times["areals"], p["ta"], 3)
        te = t.check(f"{lab} explicit time (s)", times["inference"], p["te"], 3)
        t.rows.append(["", str(k), f"{fmt(am, 4)} ±{fmt(asd, 4)}", f"[{fmt(alo)}, {fmt(ahi)}]",
                       f"{fmt(em, 4)} ±{fmt(esd, 4)}", f"[{fmt(elo)}, {fmt(ehi)}]", f"{fmt(ta)} / {fmt(te)}"])

    nm = t.check("D.1 nine angles K=20 AREALS mean", nine["mean"], 0.1637, 4)
    nsd = t.check("D.1 nine angles K=20 AREALS s.d.", nine["seed_sd"], 0.0030, 4)
    t.rows.append(["DoubleDelta, nine angles", "20", f"{fmt(nm, 4)} ±{fmt(nsd, 4)}", "—", "—", "—", "—"])
    bl = t.check("D.1 BlendedNet++ K=20 AREALS", blended["nrmse"], 0.176, 3)
    t.rows.append(["BlendedNet++, high Mach", "20", fmt(bl), "—", "—", "—", "—"])
    t.notes.append("18–19° rows: FP32 re-evaluation of the three AREALS checkpoints of the 18–19° split "
                   "(matched_fp32_areals.json, winning_comparison.json); the stored comparison keeps the "
                   "training-support configurations in which the AREALS mean is below both inference estimates. "
                   "Nine-angle row: BF16 evaluation (nine_angle/test_comparison.json). BlendedNet++ cell: "
                   "results/expanded_benchmarks/blendednet_test_grouped.json.")
    return t


def table_d2() -> Table:
    nine = _nine_angle()
    ref = _references()
    t = Table("Table D.2", "DoubleDelta, 144 test cases (16 wings, nine angles): hidden-region nRMSE",
              ["Sensors K", "Geometry only", "Measurements", "True angle"])
    paper = {5: (0.3021, 0.2111, 0.1373), 20: (0.3023, 0.1637, 0.1374),
             100: (0.3036, 0.1434, 0.1383), 500: (0.3101, 0.1410, 0.1426)}
    for k, (pg, pm, pc) in paper.items():
        s = f"sensors-{k}"
        g = t.check(f"D.2 K={k} geometry only", ref[s]["nrmse"]["geometry"]["mean"], pg, 4)
        m = t.check(f"D.2 K={k} measurements (3 seeds)", nine["results"][s]["nrmse"]["areals"]["mean"], pm, 4)
        c = t.check(f"D.2 K={k} true angle", ref[s]["nrmse"]["conditioned"]["mean"], pc, 4)
        t.rows.append([str(k), fmt(g, 4), fmt(m, 4), fmt(c, 4)])
    sd = max(nine["results"][f"sensors-{k}"]["nrmse"]["areals"]["seed_sd"] for k in SENSORS)
    t.check("D.2 caption: largest seed s.d. of the measurement column", sd, 0.0043, 4)
    t.check("D.2 caption: test cases", nine["cases"], 144, 0)
    t.check("D.2 caption: physical wings", nine["geometries"], 16, 0)
    t.check("D.2 caption: angles per wing", nine["angles_per_geometry"], 9, 0)
    scores = load_csv("sensor_completion", "gabi", "test_per_problem.csv")
    observed_area = 1 - _per_problem_mean(scores, "measurements", "sensors-500", "hidden_fraction")
    t.check("D.2 caption: area removed from scoring by 500 sensors (%)", 100 * observed_area, 10, 0)
    t.notes.append("Measurements: mean of three fine-tuning seeds (BF16); geometry-only and true-angle "
                   "references: seed-0 single models. Scored-area fraction from the per-problem scores "
                   "(hidden_fraction) in results/sensor_completion/gabi/test_per_problem.csv.")
    return t


def table_d4() -> Table:
    nine = _nine_angle()["results"]
    t = Table("Table D.4", "GABI and AREALS on the same 144 DoubleDelta test cases, angle unreported: "
              "hidden-region nRMSE and integrated-force error, means of three seeds",
              ["Sensors K", "nRMSE AREALS", "nRMSE GABI", "Force error (%) AREALS", "Force error (%) GABI"])
    paper = {0: (0.300, 0.858, 15.5, 25.3), 5: (0.211, 0.745, 8.6, 21.4), 20: (0.164, 0.642, 3.7, 14.7),
             100: (0.143, 0.537, 2.2, 12.2), 500: (0.141, 0.515, 1.5, 10.6)}
    for k, (pa, pg, pfa, pfg) in paper.items():
        s = "unobserved" if k == 0 else f"sensors-{k}"
        r = nine[s]
        a = t.check(f"D.4 K={k} AREALS nRMSE", r["nrmse"]["areals"]["mean"], pa, 3)
        g = t.check(f"D.4 K={k} GABI nRMSE", r["nrmse"]["gabi_surface"]["mean"], pg, 3)
        fa = t.check(f"D.4 K={k} AREALS force (%)", 100 * r["force_rel"]["areals"]["mean"], pfa, 1)
        fg = t.check(f"D.4 K={k} GABI force (%)", 100 * r["force_rel"]["gabi_surface"]["mean"], pfg, 1)
        t.rows.append([str(k), fmt(a), fmt(g), fmt(fa, 1), fmt(fg, 1)])

    scores = load_csv("sensor_completion", "gabi", "test_per_problem.csv")
    t.check("D.4 caption: true-angle force error at 20 sensors (%)",
            100 * _per_problem_mean(scores, "conditioned", "sensors-20", "force_rel"), 1.9, 1)
    t.check("D.4 caption: true-angle force error at 500 sensors (%)",
            100 * _per_problem_mean(scores, "conditioned", "sensors-500", "force_rel"), 1.8, 1)
    s20 = nine["sensors-20"]
    t.check("D.4 caption: AREALS nRMSE at 20, mean", s20["nrmse"]["areals"]["mean"], 0.164, 3)
    t.check("D.4 caption: AREALS nRMSE at 20, s.d.", s20["nrmse"]["areals"]["seed_sd"], 0.003, 3)
    t.check("D.4 caption: AREALS force at 20 (%), mean", 100 * s20["force_rel"]["areals"]["mean"], 3.7, 1)
    t.check("D.4 caption: AREALS force at 20 (%), s.d.", 100 * s20["force_rel"]["areals"]["seed_sd"], 0.4, 1)
    t.check("D.4 caption: GABI nRMSE at 20, mean", s20["nrmse"]["gabi_surface"]["mean"], 0.642, 3)
    t.check("D.4 caption: GABI nRMSE at 20, s.d.", s20["nrmse"]["gabi_surface"]["seed_sd"], 0.155, 3)
    t.check("D.4 caption: GABI force at 20 (%), mean", 100 * s20["force_rel"]["gabi_surface"]["mean"], 14.7, 1)
    t.check("D.4 caption: GABI force at 20 (%), s.d.", 100 * s20["force_rel"]["gabi_surface"]["seed_sd"], 1.7, 1)
    blind = nine["unobserved"]["nrmse"]
    t.check("D.4 caption: GABI / AREALS error with nothing observed",
            blind["gabi_surface"]["mean"] / blind["areals"]["mean"], 2.9, 1)
    t.check("D.4 caption: share of GABI's error removed by 20 sensors (%)",
            100 * (1 - s20["nrmse"]["gabi_surface"]["mean"] / blind["gabi_surface"]["mean"]), 25, 0)
    t.check("D.4 caption: share of AREALS's error removed by 20 sensors (%)",
            100 * (1 - s20["nrmse"]["areals"]["mean"] / blind["areals"]["mean"]), 45, 0)
    t.notes.append("K = 0 is the unobserved setting. True-angle force errors: seed-0 reference, from "
                   "results/sensor_completion/gabi/test_per_problem.csv (method 'conditioned').")
    return t


def table_d6() -> Table:
    data, grouped = _heldout("test")
    t = Table("Table D.6", "Angles held out from training and development (train 11, 13, 15, 17, 19°; "
              "test 12, 14, 16, 18° on 16 unseen wings): hidden-region nRMSE and gap closure G",
              ["Sensors K", "Geometry only", "Measurements", "True angle", "G [95% interval]"])
    paper = {5: (0.2832, 0.1947, 0.1333, 63.1, 41.3, 83.0), 20: (0.2834, 0.1550, 0.1334, 91.7, 86.6, 98.6),
             100: (0.2846, 0.1464, 0.1343, 96.4, 90.3, 104.4), 500: (0.2908, 0.1433, 0.1384, 99.8, 94.5, 107.3)}
    for k, (pg, pm, pc, pG, plo, phi) in paper.items():
        s = grouped[f"sensors-{k}"]
        g = t.check(f"D.6 K={k} geometry only", s["nrmse"]["geometry"]["mean"], pg, 4)
        m = t.check(f"D.6 K={k} measurements", s["nrmse"]["measurements"]["mean"], pm, 4)
        c = t.check(f"D.6 K={k} true angle", s["nrmse"]["conditioned"]["mean"], pc, 4)
        G = t.check(f"D.6 K={k} G (%)", 100 * s["gap_closed"], pG, 1)
        lo = t.check(f"D.6 K={k} G 95% low (%)", 100 * s["gap_closed_95"][0], plo, 1)
        hi = t.check(f"D.6 K={k} G 95% high (%)", 100 * s["gap_closed_95"][1], phi, 1)
        t.rows.append([str(k), fmt(g, 4), fmt(m, 4), fmt(c, 4), f"{fmt(G, 1)}% [{fmt(lo, 1)}, {fmt(hi, 1)}%]"])
    t.check("D.6 caption: unseen wing geometries", data["physical_geometries"], 16, 0)
    t.check("D.6 caption: test cases", data["cases"], 64, 0)
    audit = load_json("sensor_completion", "heldout_angles", "control_audit.json")
    for i, paper_angle in enumerate([11, 13, 15, 17, 19]):
        t.check(f"D.6 caption: training angle {i + 1} (deg)", audit["train_angles"][i], paper_angle, 0)
    t.check("D.6 caption: number of training angles", len(audit["train_angles"]), 5, 0)
    for i, paper_angle in enumerate([12, 14, 16, 18]):
        t.check(f"D.6 caption: test angle {i + 1} (deg)", audit["test_angles"][i], paper_angle, 0)
    t.check("D.6 caption: conditions per wing grouped in the intervals", len(audit["test_angles"]), 4, 0)
    t.notes.append("One seed. G is computed from casewise normalised MSE; intervals resample the 16 wings, "
                   "keeping each wing's four test angles together.")
    return t


def figure_d1() -> Table:
    nine = _nine_angle()["results"]
    ref = _references()
    t = Table("Figure D.1", "Plotted values: hidden-region nRMSE on the nine-angle test split against sensors K",
              ["Curve", "K=5", "K=20", "K=100", "K=500"])
    curves = [
        ("geometry only", lambda s: ref[s]["nrmse"]["geometry"]["mean"], (0.302, 0.302, 0.304, 0.310), 3),
        ("sensors (AREALS)", lambda s: nine[s]["nrmse"]["areals"]["mean"], (0.2111, 0.1637, 0.1434, 0.1410), 4),
        ("true angle", lambda s: ref[s]["nrmse"]["conditioned"]["mean"], (0.137, 0.137, 0.138, 0.143), 3),
        ("GABI", lambda s: nine[s]["nrmse"]["gabi_surface"]["mean"],
         (0.7451882254, 0.6422355529, 0.5366860914, 0.5146918886), 10),
    ]
    for name, get, plotted, decimals in curves:
        cells = [name]
        for k, value in zip(SENSORS, plotted):
            v = t.check(f"Fig D.1 {name} K={k}", get(f"sensors-{k}"), value, decimals)
            cells.append(fmt(v, min(decimals, 4)))
        t.rows.append(cells)
    G = t.check("Fig D.1 label: G at K=20, first AREALS seed (%)", 100 * ref["sensors-20"]["fraction_gap_closed"],
                90.7, 1)
    t.rows.append(["G (MSE gap closed at K=20, seed 0)", "", _pct(G), "", ""])
    t.notes.append("Paper values are the coordinates written in the figure source; GABI is plotted at full "
                   "precision and checked to 10 decimals.")
    return t


def text_section_4_4() -> Table:
    nine = _nine_angle()["results"]
    ref = _references()
    scratch = _scratch()
    winning = load_json("sensor_completion", "condition_inference", "winning_comparison.json")["rows"]
    explicit = [r["inference_nrmse"]["mean"] for r in winning
                if r["dataset"] == "doubledelta" and r["domain"] == "training_support"]
    _, heldout = _heldout("test")
    scores = load_csv("sensor_completion", "gabi", "test_per_problem.csv")
    s20 = nine["sensors-20"]
    t = Table("Section 4.4", "Sensors in place of the operating condition (also the abstract, Section 1 and Section 5)",
              ["Claim", "Rebuilt", "Paper"])

    def row(label, value, paper, decimals, unit=""):
        v = t.check(label, value, paper, decimals)
        t.rows.append([label, f"{fmt(v, decimals)}{unit}", f"{fmt(paper, decimals)}{unit}"])

    ratio = s20["nrmse"]["gabi_over_areals"]
    force_ratio = s20["force_rel"]["gabi_over_areals"]
    row("GABI / AREALS field error at 20 sensors (Sec. 4.4)", ratio, 3.9, 1, "x")
    row("GABI / AREALS force error at 20 sensors (Sec. 4.4)", force_ratio, 4.0, 1, "x")
    row("GABI / AREALS field error at 20 sensors (abstract)", ratio, 3.9, 1, "x")
    row("GABI / AREALS force error at 20 sensors (abstract)", force_ratio, 4.0, 1, "x")
    row("GABI / AREALS field error at 20 sensors (Sec. 1)", ratio, 3.9, 1, "x")
    row("GABI / AREALS field error at 20 sensors (Sec. 5)", ratio, 3.9, 1, "x")
    gain = 100 * (1 - s20["nrmse"]["areals"]["mean"] / scratch["sensors-20"]["nrmse"]["areal_scratch"]["mean"])
    row("AREALS below the model from scratch at 20 sensors", gain, 26.2, 1, "%")
    ratios = [nine[f"sensors-{k}"]["nrmse"]["gabi_over_areals"] for k in SENSORS]
    row("GABI / AREALS error ratio, smallest over 5-500 sensors", min(ratios), 3.5, 1, "x")
    row("GABI / AREALS error ratio, largest over 5-500 sensors", max(ratios), 3.9, 1, "x")
    every = all(min(r["nrmse"]["gabi_surface"]["per_seed"].values()) > max(r["nrmse"]["areals"]["per_seed"].values())
                for r in nine.values())
    row("Every GABI seed worse than every AREALS seed at every setting (1 = true)", float(every), 1, 0)
    shape, cond = ref["sensors-20"]["nrmse"]["geometry"]["mean"], ref["sensors-20"]["nrmse"]["conditioned"]["mean"]
    closed = 100 * (shape - s20["nrmse"]["areals"]["mean"]) / (shape - cond)
    row("nRMSE gap closed by 20 sensors (Sec. 4.4)", closed, 84, 0, "%")
    row("nRMSE gap closed by 20 sensors (Sec. 1)", closed, 84, 0, "%")
    row("nRMSE gap closed by 20 sensors (Sec. 5)", closed, 84, 0, "%")
    G = ref["sensors-20"]
    row("Normalised-MSE gap closed at 20 sensors, seed 0", 100 * G["fraction_gap_closed"], 90.7, 1, "%")
    row("  95% interval, low", 100 * G["fraction_gap_closed_95"][0], 84.4, 1, "%")
    row("  95% interval, high", 100 * G["fraction_gap_closed_95"][1], 96.9, 1, "%")
    row("AREALS minus true-condition nRMSE at 100 sensors",
        nine["sensors-100"]["nrmse"]["areals"]["mean"] - ref["sensors-100"]["nrmse"]["conditioned"]["mean"],
        0.005, 3)
    row("Shape-only force error (nothing observed)",
        100 * _per_problem_mean(scores, "geometry", "unobserved", "force_rel"), 15.5, 1, "%")
    row("AREALS force error at 20 sensors", 100 * s20["force_rel"]["areals"]["mean"], 3.7, 1, "%")
    row("AREALS force error at 500 sensors", 100 * nine["sensors-500"]["force_rel"]["areals"]["mean"], 1.5, 1, "%")
    row("True-condition force error at 500 sensors",
        100 * _per_problem_mean(scores, "conditioned", "sensors-500", "force_rel"), 1.8, 1, "%")
    row("Explicit inference, 18-19 deg, lowest over 20-500 sensors", min(explicit), 0.300, 3)
    row("Explicit inference, 18-19 deg, highest over 20-500 sensors", max(explicit), 0.307, 3)
    row("Held-out angles: normalised-MSE gap closed at 20 sensors",
        100 * heldout["sensors-20"]["gap_closed"], 91.7, 1, "%")
    t.notes.append("Force errors of the shape-only and true-condition references (seed 0) come from "
                   "results/sensor_completion/gabi/test_per_problem.csv; AREALS values are three-seed means.")
    return t


def text_appendix_d_e() -> Table:
    nine = _nine_angle()
    results = nine["results"]
    scratch = _scratch()
    winning = load_json("sensor_completion", "condition_inference", "winning_comparison.json")["rows"]
    rows = {r["setting"]: r for r in winning if r["dataset"] == "doubledelta" and r["domain"] == "training_support"}
    audit = load_json("sensor_completion", "heldout_angles", "control_audit.json")
    test, _ = _heldout("test")
    val, _ = _heldout("val")
    t = Table("Appendices D and E", "From-scratch baseline, fine-tuning seeds, explicit inference, "
              "held-out-angle split", ["Claim", "Rebuilt", "Paper"])

    def row(label, value, paper, decimals, unit=""):
        v = t.check(label, value, paper, decimals)
        t.rows.append([label, f"{fmt(v, decimals)}{unit}", f"{fmt(paper, decimals)}{unit}"])

    for k, p_scratch, p_areals, p_gain in [(20, 0.2219, 0.1637, 26.2), (500, 0.2177, 0.1410, 35.2)]:
        s = f"sensors-{k}"
        sc = scratch[s]["nrmse"]["areal_scratch"]["mean"]
        ar = results[s]["nrmse"]["areals"]["mean"]
        row(f"From scratch, {k} sensors (seed 0)", sc, p_scratch, 4)
        row(f"AREALS, {k} sensors (three seeds)", ar, p_areals, 4)
        row(f"Error reduction from geometry pretraining, {k} sensors", 100 * (1 - ar / sc), p_gain, 1, "%")

    row("Fine-tuning seeds", len(nine["seeds"]), 3, 0)
    for seed in nine["seeds"]:
        done = load_json("sensor_completion", "nine_angle", f"seed{seed}_training_complete.json")
        row(f"Seed {seed}: epochs", done["epochs"], 40, 0)
        row(f"Seed {seed}: optimiser updates", done["updates"], 10800, 0)
    row("Seed spread (s.d.) of nRMSE at 20 sensors", results["sensors-20"]["nrmse"]["areals"]["seed_sd"], 0.003, 3)

    # Training arguments of the added seeds, written before they ran; seed 0 is the original run, whose arguments
    # are in the single_seed_study commands of the 06 README and whose checkpoint metadata is stored in the protocol.
    protocol = _protocol()
    seed0 = protocol["seed0_metadata"]
    for seed, args in sorted(protocol["training_args"].items()):
        row(f"Seed {seed}: batch size", _arg(args, "--batch-size"), 8, 0)
        row(f"Seed {seed}: development cases used for checkpoint selection", _arg(args, "--select-cars"), 36, 0)
        row(f"Seed {seed}: epochs (training arguments)", _arg(args, "--epochs"), 40, 0)
        row(f"Seed {seed}: queries per example (App. E)", _arg(args, "--queries"), 2048, 0)
        row(f"Seed {seed}: backbone learning rate, log10 (App. E)", math.log10(_arg(args, "--lr")), -4, 0)
        row(f"Seed {seed}: new-parameter learning rate, log10 (App. E)", math.log10(_arg(args, "--new-lr")), -3, 0)
    row("Seed 0: queries per example (App. E)", seed0["queries"], 2048, 0)
    row("Seed 0: backbone learning rate, log10 (App. E)", math.log10(seed0["lr"]), -4, 0)
    row("Seed 0: new-parameter learning rate, log10 (App. E)", math.log10(seed0["new_lr"]), -3, 0)
    row("Optimiser updates per model (protocol)", protocol["updates"], 10800, 0)

    explicit = [rows[f"sensors-{k}"]["inference_nrmse"]["mean"] for k in (20, 100, 500)]
    row("Explicit inference, lowest over 20-500 sensors", min(explicit), 0.300, 3)
    row("Explicit inference, highest over 20-500 sensors", max(explicit), 0.307, 3)
    for k, paper in [(20, 0.206), (100, 0.178), (500, 0.187)]:
        # The text quotes the Table D.1 entries (printed to 4 dp) at 3 dp.
        row(f"AREALS at 18-19 deg, {k} sensors (from Table D.1)",
            as_printed(rows[f"sensors-{k}"]["areals_nrmse"]["mean"], 4), paper, 3)
    for k in (20, 100, 500):
        r = rows[f"sensors-{k}"]
        agree = abs(round(r["inference_nrmse"]["mean"], 5) - round(r["plugin_nrmse"]["mean"], 5))
        row(f"Mixture minus plug-in at five decimals, {k} sensors", agree, 0, 5)
    apart = all(rows[f"sensors-{k}"]["areals_nrmse"]["bootstrap95"][1]
                < rows[f"sensors-{k}"]["inference_nrmse"]["bootstrap95"][0] for k in (20, 100, 500))
    row("95% intervals of AREALS and explicit inference do not overlap (1 = true)", float(apart), 1, 0)
    samples = {rows[f"sensors-{k}"]["timing_samples"] for k in (20, 100, 500)}
    row("Timing samples per row (3 seeds x 4 wings x 3 repeats)",
        samples.pop() if len(samples) == 1 else float("nan"), 3 * 4 * 3, 0)

    cases = {r["training_cases"] for r in audit["roles"].values()}
    row("Held-out angles: training cases", cases.pop() if len(cases) == 1 else float("nan"), 1200, 0)
    row("Held-out angles: development cases", val["cases"], 80, 0)
    row("Held-out angles: test cases", test["cases"], 64, 0)
    t.notes.append("The from-scratch model and the references are seed 0; AREALS is the mean of three "
                   "fine-tuning seeds that share one geometry checkpoint.")
    return t


def text_appendix_h() -> Table:
    nine = _nine_angle()
    verification = load_json("sensor_completion", "nine_angle", "verification.json")
    audit = load_json("sensor_completion", "heldout_angles", "control_audit.json")
    response = load_json("sensor_completion_extra", "heldout_angles", "no_angle_search.json")
    test, _ = _heldout("test")
    val, _ = _heldout("val")
    t = Table("Appendix H", "Benchmark controls: nine-angle and held-out-angle studies", ["Claim", "Rebuilt", "Paper"])

    def row(label, value, paper, decimals, shown=None, paper_shown=None):
        v = t.check(label, value, paper, decimals)
        t.rows.append([label, shown if shown is not None else fmt(v, decimals),
                       paper_shown if paper_shown is not None else fmt(paper, decimals)])

    row("Held-out angles: hidden-value perturbation checks passed",
        sum(r["pass_lines"] for r in audit["roles"].values()), 36, 0)
    worst = max(test["independent_maximum_difference"], val["independent_maximum_difference"])
    row("Held-out angles: recomputed vs recorded scores agree within the stated 1.2e-12 (1 = true)",
        float(worst <= 1.2e-12), 1, 0, shown=f"1 (max {worst:.1e})", paper_shown="within 1.2e-12")
    row("Held-out angles: condition/search artifacts read", response["forbidden_file_reads"], 0, 0)
    row("Held-out angles: condition-construction calls", response["condition_input_calls"], 0, 0)
    row("Held-out angles: encoder invocations", response["encoder_calls"], 1, 0)
    row("Held-out angles: output values", response["predicted_surface_points"], 32000, 0)
    row("Nine angles: AREALS predictions recomputed", verification["recomputed_predictions"], 6912, 0)
    row("Nine angles: seeds", len(nine["seeds"]), 3, 0)
    row("Nine angles: observation settings", len(nine["results"]), 8, 0)
    gap = max(verification["maximum_difference_from_native_metrics"].values())
    row("Nine angles: recomputed vs recorded scores, max difference (units of 1e-14)", gap / 1e-14, 4, 0,
        shown=f"{gap:.2e}", paper_shown="4e-14")
    t.notes.append("The held-out-angle recomputation (summarize_unseen.py) agrees to 1.1e-16 on test and "
                   "1.7e-16 on development, inside the 1e-12 tolerance recorded in control_audit.json.")
    return t


def table_a1() -> Table:
    nine = _nine_angle()
    protocol = _protocol()
    audit = load_json("sensor_completion", "heldout_angles", "control_audit.json")
    heldout, heldout_settings = _heldout("test")
    winning = [r for r in load_json("sensor_completion", "condition_inference", "winning_comparison.json")["rows"]
               if r["dataset"] == "doubledelta" and r["domain"] == "training_support"]
    extrapolation = load_json("expanded_benchmarks", "protocol.json")["experiments"]["doubledelta"]
    t = Table("Table A.1 (rows for these studies)", "Field training, evaluation, observations and seeds",
              ["Result", "Field training", "Evaluation", "Observations", "Seeds"])

    # Unreported angle (Figure D.1, Tables 2, D.2, D.4 and D.5)
    train = t.check("A.1 unreported angle: training cases", protocol["counts"]["train"], 2160, 0)
    t.check("A.1 unreported angle: training cases of the seed-0 model", protocol["seed0_metadata"]["train_cars"], 2160, 0)
    epochs = t.check("A.1 unreported angle: epochs per model", protocol["epochs"], 40, 0)
    cases = t.check("A.1 unreported angle: test cases", nine["cases"], 144, 0)
    wings = t.check("A.1 unreported angle: test wings", nine["geometries"], 16, 0)
    counts = _sensor_counts(nine["results"])
    lo = t.check("A.1 unreported angle: fewest sensors", counts[0], 0, 0)
    hi = t.check("A.1 unreported angle: most sensors", counts[-1], 500, 0)
    patches = [s for s in nine["results"] if s.startswith("patch-")]
    seeds = t.check("A.1 unreported angle: seeds", len(nine["seeds"]), 3, 0)
    t.rows.append(["Unreported angle", f"{train:,.0f} cases, {epochs:.0f} epochs per model",
                   f"{cases:.0f} test cases ({wings:.0f} wings)",
                   f"{lo:.0f}–{hi:.0f} sensors; {len(patches)} patch settings", f"{seeds:.0f}; scratch and references: see note"])

    # Held-out angles (Table D.6)
    cases = {r["training_cases"] for r in audit["roles"].values()}
    train = t.check("A.1 held-out angles: training cases", cases.pop() if len(cases) == 1 else float("nan"), 1200, 0)
    odd = t.check("A.1 held-out angles: training angles all odd (1 = true)",
                  float(all(a % 2 == 1 for a in audit["train_angles"])), 1, 0)
    test = t.check("A.1 held-out angles: test cases", heldout["cases"], 64, 0)
    wings = t.check("A.1 held-out angles: test wings", heldout["physical_geometries"], 16, 0)
    even = t.check("A.1 held-out angles: test angles all even (1 = true)",
                   float(all(a % 2 == 0 for a in audit["test_angles"])), 1, 0)
    counts = [c for c in _sensor_counts(heldout_settings) if c > 0]
    lo = t.check("A.1 held-out angles: fewest sensors", counts[0], 5, 0)
    hi = t.check("A.1 held-out angles: most sensors", counts[-1], 500, 0)
    t.rows.append(["Held-out angles", f"{train:,.0f} cases ({'odd' if odd else '?'} angles)",
                   f"{test:.0f} cases ({wings:.0f} wings, {'even' if even else '?'} angles)",
                   f"{lo:.0f}–{hi:.0f} sensors", "see note"])

    # Explicit condition inference (Table D.1)
    families = t.check("A.1 explicit inference: wing families of the true-condition model",
                       extrapolation["geometry_families"][0], 228, 0)
    a_lo = t.check("A.1 explicit inference: lowest training angle (deg)", min(extrapolation["train_angles"]), 11, 0)
    a_hi = t.check("A.1 explicit inference: highest training angle (deg)", max(extrapolation["train_angles"]), 16, 0)
    geometries = {r[m]["geometries"] for r in winning for m in ("areals_nrmse", "inference_nrmse")}
    wings = t.check("A.1 explicit inference: test wings",
                    geometries.pop() if len(geometries) == 1 else float("nan"), 16, 0)
    e_lo = t.check("A.1 explicit inference: lowest test angle (deg)", min(extrapolation["test_angles"]), 18, 0)
    e_hi = t.check("A.1 explicit inference: highest test angle (deg)", max(extrapolation["test_angles"]), 19, 0)
    counts = _sensor_counts(r["setting"] for r in winning)
    lo = t.check("A.1 explicit inference: fewest sensors", counts[0], 20, 0)
    hi = t.check("A.1 explicit inference: most sensors", counts[-1], 500, 0)
    run_seeds = {r[m]["seeds"] for r in winning for m in ("areals_nrmse", "inference_nrmse")}
    seeds = t.check("A.1 explicit inference: seeds", run_seeds.pop() if len(run_seeds) == 1 else float("nan"), 3, 0)
    t.rows.append(["Explicit condition inference",
                   f"True-condition model, {families:.0f} wing families, {a_lo:.0f}–{a_hi:.0f}°",
                   f"{wings:.0f} wings at {e_lo:.0f}–{e_hi:.0f}°; timing wings: see note",
                   f"{lo:.0f}–{hi:.0f} sensors", f"{seeds:.0f}"])

    t.notes.append("Not stored as numbers, so not checked here: the paper's single seed (1) of the from-scratch "
                   "model, the references and the held-out-angle models (one trained model each; see the scope "
                   "fields of the stored files), and the 4 timing wings (stated in winning_comparison.json "
                   "'timing_scope'; the 36 = 3 x 4 x 3 timing samples are checked under Appendices D and E).")
    t.notes.append("True-condition training set and angles: results/expanded_benchmarks/protocol.json.")
    return t


def tables() -> list[Table]:
    return [table_2(), table_d1(), table_d2(), table_d4(), table_d6(), figure_d1(),
            text_section_4_4(), text_appendix_d_e(), text_appendix_h(), table_a1()]
