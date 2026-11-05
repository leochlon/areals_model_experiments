"""Simulation efficiency: Figure 1, Section 4.2 and the Appendix C "Simulation efficiency" paragraph.

Sources (all under results/):
  simulation_efficiency/windsor_drivaernet/summary.json      verified_results: per-dataset analyse.py summaries (WindsorML, DrivAerNet++)
  simulation_efficiency/windsor_drivaernet/fixed_accuracy.csv the nRMSE 0.25 endpoints of the same summaries
  simulation_efficiency/drivaerml/{summary,thresholds,replay}.json  DrivAerML analyse.py and replay.py outputs
  simulation_efficiency_extra/{windsor,drivaernet}/replay.json      replay.py receipts for WindsorML and DrivAerNet++
"""
from __future__ import annotations

import re

from paper_tables._core import Table, fmt, load_csv, load_json

ORDER = ("WindsorML", "DrivAerML", "DrivAerNet++")

# Figure 1 is drawn by the paper source (figures/efficiency.tex) from tables of
# budget, scratch mean / 95% band, AREALS mean / 95% band, printed there at five decimals.
FIGURE_1_SOURCE = {
    "WindsorML": {
        18: (0.44783, 0.41559, 0.48301, 0.27479, 0.24758, 0.30220),
        35: (0.34900, 0.31890, 0.37842, 0.22266, 0.20029, 0.24510),
        69: (0.26989, 0.24185, 0.29744, 0.19246, 0.16986, 0.21774),
        137: (0.21173, 0.18988, 0.23635, 0.16570, 0.14315, 0.19238),
        274: (0.18732, 0.16331, 0.21402, 0.14605, 0.12587, 0.16905),
    },
    "DrivAerML": {
        25: (0.48944, 0.47332, 0.50561, 0.36870, 0.36031, 0.37704),
        50: (0.41597, 0.40183, 0.43079, 0.30812, 0.30015, 0.31558),
        100: (0.34679, 0.33244, 0.36203, 0.25339, 0.24663, 0.26034),
        200: (0.26909, 0.25913, 0.27878, 0.20937, 0.20468, 0.21430),
        400: (0.21935, 0.21323, 0.22574, 0.18089, 0.17685, 0.18511),
    },
    "DrivAerNet++": {
        406: (0.30346, 0.29783, 0.30939, 0.27712, 0.27291, 0.28139),
        812: (0.27650, 0.27048, 0.28273, 0.25493, 0.25168, 0.25843),
        1623: (0.25637, 0.25215, 0.26061, 0.23860, 0.23568, 0.24184),
        3246: (0.24746, 0.24421, 0.25091, 0.23257, 0.22965, 0.23581),
        6492: (0.24603, 0.24193, 0.25052, 0.23026, 0.22742, 0.23341),
    },
}
# Tick labels of Figure 1 (training simulations per panel).
FIGURE_1_TICKS = {
    "WindsorML": (18, 35, 69, 137, 274),
    "DrivAerML": (25, 50, 100, 200, 400),
    "DrivAerNet++": (406, 812, 1623, 3246, 6492),
}


def _studies() -> dict[str, dict]:
    """Bring the three studies into one shape: curves, thresholds, paired differences and run diagnostics."""
    suite = {r["dataset"]: r for r in load_json("simulation_efficiency", "windsor_drivaernet", "summary.json")["verified_results"]}
    fixed = load_csv("simulation_efficiency", "windsor_drivaernet", "fixed_accuracy.csv")
    dml = load_json("simulation_efficiency", "drivaerml", "summary.json")
    out = {}
    for name, key in (("WindsorML", "windsor"), ("DrivAerNet++", "drivaernet")):
        s = suite[key]
        replay = load_json("simulation_efficiency_extra", key, "replay.json")
        out[name] = dict(
            rows=s["summary"], thresholds=s["thresholds"], differences=s["paired_differences"],
            flagged=int(s["optimisation_budget_limited_runs"]["0"]), runs=int(s["completed_runs"]),
            updates=int(s["total_field_updates"]), uncertainty=s["uncertainty"], replay=replay,
            split=(int(s["split"]["train"]), int(s["split"]["val"]), int(s["split"]["test"])),
            fixed={(int(r["sensors"]), float(r["target"])): r for r in fixed if r["dataset"] == key},
            gpu_hours=float(s["summed_fit_wall_gpu_hours"]),
        )
    replay = load_json("simulation_efficiency", "drivaerml", "replay.json")
    rows = dml["summary"]
    budgets = sorted({r["training_simulations"] for r in rows})
    validation = {r["including_validation"] - r["training_simulations"] for r in rows}
    assert len(validation) == 1
    sensor_settings = len({r["sensors"] for r in rows})
    out["DrivAerML"] = dict(
        rows=rows, thresholds=load_json("simulation_efficiency", "drivaerml", "thresholds.json"),
        differences=dml["paired_differences"], flagged=int(dml["optimisation_budget_limited_runs"]),
        runs=int(dml["completed_runs"]), updates=int(dml["total_field_updates"]), uncertainty=dml["uncertainty"],
        replay=replay,
        # The full training pool is the largest budget; every scored row is one (fit, test case, sensor setting).
        split=(max(budgets), validation.pop(), replay["scored_rows"] // (sensor_settings * int(dml["completed_runs"]))),
        fixed={}, gpu_hours=float(dml["summed_fit_wall_gpu_hours"]),
    )
    return out


def _curve(study: dict, arm: str, sensors: int = 0) -> dict[int, dict]:
    return {r["training_simulations"]: r for r in study["rows"] if r["arm"] == arm and r["sensors"] == sensors}


def _threshold(study: dict, target: float, sensors: int = 0) -> dict:
    (row,) = [t for t in study["thresholds"] if t["sensors"] == sensors and abs(t["target"] - target) < 1e-9]
    return row


def _gain(study: dict, budget: int) -> float:
    """Relative error reduction of AREALS over scratch at one budget, zero sensors, in percent."""
    p, s = _curve(study, "pretrained")[budget]["mean_nrmse"], _curve(study, "scratch")[budget]["mean_nrmse"]
    return 100 * (s - p) / s


def _n(value) -> str:
    return "not reached" if value is None else f"{int(value):,}"


def figure_1(studies: dict) -> Table:
    t = Table("Figure 1", "Test nRMSE without sensors against training simulations (mean of three seeds [pointwise 95% bootstrap band])",
              ["Dataset", "Training simulations", "From scratch", "AREALS"])
    for name in ORDER:
        study = studies[name]
        scratch, areals = _curve(study, "scratch"), _curve(study, "pretrained")
        budgets = sorted(scratch)
        for budget, tick in zip(budgets, FIGURE_1_TICKS[name], strict=True):
            t.check(f"{name}: tick label {tick}", budget, tick, 0)
        for budget in budgets:
            paper = FIGURE_1_SOURCE[name][budget]
            values = []
            for i, (arm, row) in enumerate((("scratch", scratch[budget]), ("AREALS", areals[budget]))):
                for j, (part, key) in enumerate((("mean", "mean_nrmse"), ("band low", "ci95_low"), ("band high", "ci95_high"))):
                    values.append(t.check(f"{name} {budget} {arm} {part}", row[key], paper[3 * i + j], 5))
            t.rows.append([name, f"{budget:,}", f"{fmt(values[0], 5)} [{fmt(values[1], 5)}, {fmt(values[2], 5)}]",
                           f"{fmt(values[3], 5)} [{fmt(values[4], 5)}, {fmt(values[5], 5)}]"])
        target = _threshold(study, 0.25)
        saving = 100 * (1 - target["pretrained_minimum_tested_budget"] / target["scratch_minimum_tested_budget"])
        label = {"WindsorML": 74.5, "DrivAerML": 50, "DrivAerNet++": 50}[name]
        t.check(f"{name}: panel label '% fewer simulations'", saving, label, 1 if name == "WindsorML" else 0)
        t.rows.append([name, "arrow at nRMSE 0.25", f"{target['scratch_minimum_tested_budget']:,}",
                       f"{target['pretrained_minimum_tested_budget']:,} ({fmt(saving, 1)}% fewer)"])
    t.notes.append("Paper values for the curves are the five-decimal tables embedded in the paper source of Figure 1 "
                   "(figures/efficiency.tex); the arrows and labels use the 0.25 thresholds below.")
    return t


def thresholds(studies: dict) -> Table:
    t = Table("Section 4.2 and Appendix C", "Smallest tested training budget reaching a fixed test nRMSE (no interpolation)",
              ["Dataset", "Sensors", "Target", "AREALS", "From scratch", "Fewer simulations", "Fewer incl. validation",
               "Bootstrap draws with AREALS fewer"])
    paper_025 = {"WindsorML": (35, 137, 74.5, 1, 59.3), "DrivAerML": (200, 400, 50, 0, 46.1),
                 "DrivAerNet++": (1623, 3246, 50, 0, 40.3)}
    paper_030 = {"WindsorML": (18, 69), "DrivAerML": (100, 200), "DrivAerNet++": (406, 812)}
    paper_fewer = {"WindsorML": (100, 2), "DrivAerML": (100, 2), "DrivAerNet++": (99.9, 1)}  # "every draw", "99.9%"
    training, with_validation = [], []
    for name in ORDER:
        study = studies[name]
        smallest = min(_curve(study, "pretrained"))
        for sensors in (0, 20):
            for target in (0.2, 0.25, 0.3):
                row = _threshold(study, target, sensors)
                p, s = row["pretrained_minimum_tested_budget"], row["scratch_minimum_tested_budget"]
                if target == 0.25 and study["fixed"]:
                    # WindsorML and DrivAerNet++: the 0.25 endpoints are also tabulated in fixed_accuracy.csv.
                    f = study["fixed"][(sensors, 0.25)]
                    assert (int(f["pretrained_N"]), int(f["scratch_N"])) == (p, s)
                    reduction, incl = 100 * float(f["training_reduction"]), 100 * float(f["including_validation_reduction"])
                    fewer = 100 * float(f["bootstrap_fewer_fraction"])
                else:
                    reduction = None if row.get("training_reduction") is None else 100 * row["training_reduction"]
                    incl = None if row.get("reduction_including_validation") is None else 100 * row["reduction_including_validation"]
                    fewer = 100 * row["bootstrap_pretrained_needs_fewer_fraction"]
                tag = f"{name}, {sensors} sensors, target {target}"
                if target == 0.25:
                    pp, ps, pr, dec, pv = paper_025[name]
                    t.check(f"{tag}: AREALS budget", p, pp, 0)
                    t.check(f"{tag}: scratch budget", s, ps, 0)
                    if sensors == 0:
                        t.check(f"{tag}: % fewer simulations", reduction, pr, dec)
                        t.check(f"{tag}: % fewer including validation", incl, pv, 1)
                        t.check(f"{tag}: % bootstrap draws with AREALS fewer", fewer, *paper_fewer[name])
                        training.append(reduction)
                        with_validation.append(incl)
                elif target == 0.3 and sensors == 0:
                    t.check(f"{tag}: AREALS budget", p, paper_030[name][0], 0)
                    t.check(f"{tag}: scratch budget", s, paper_030[name][1], 0)
                    if name != "DrivAerML":  # "AREALS's being the smallest tested on WindsorML and DrivAerNet++"
                        t.check(f"{tag}: AREALS budget is the smallest tested (1 = yes)", int(p == smallest), 1, 0)
                elif target == 0.2 and sensors == 0:
                    if name == "WindsorML":
                        t.check(f"{tag}: AREALS budget", p, 69, 0)
                        t.check(f"{tag}: scratch budget", s, 274, 0)
                    elif name == "DrivAerML":  # "on DrivAerML only AREALS reaches it"
                        t.check(f"{tag}: AREALS reaches target (1 = yes)", int(p is not None), 1, 0)
                        t.check(f"{tag}: scratch reaches target (1 = yes)", int(s is not None), 0, 0)
                    else:  # "on DrivAerNet++ neither does"
                        t.check(f"{tag}: AREALS reaches target (1 = yes)", int(p is not None), 0, 0)
                        t.check(f"{tag}: scratch reaches target (1 = yes)", int(s is not None), 0, 0)
                t.rows.append([name, sensors, fmt(target, 2), _n(p), _n(s),
                               "—" if reduction is None else f"{fmt(reduction, 1)}%",
                               "—" if incl is None else f"{fmt(incl, 1)}%", f"{fmt(fewer, 2)}%"])
    # Section 1, 4.2 and 6 summarise the three zero-sensor savings as ranges.
    t.check("Range of % fewer simulations: low (\"50–74.5%\")", min(training), 50, 0)
    t.check("Range of % fewer simulations: high (\"50–74.5%\")", max(training), 74.5, 1)
    t.check("Range of % fewer including validation: low (\"40–59%\")", min(with_validation), 40, 0)
    t.check("Range of % fewer including validation: high (\"40–59%\")", max(with_validation), 59, 0)
    t.notes.append("Validation savings add each dataset's validation simulations to both budgets; test simulations "
                   "and geometry-pretraining compute are excluded, as in Appendix C.")
    return t


def gains(studies: dict) -> Table:
    t = Table("Appendix C (gains)", "Error reduction of AREALS over scratch at each budget, no sensors, 8,000 updates per fit",
              ["Dataset", "Training simulations", "From scratch", "AREALS", "Reduction", "Paired difference [95% CI]"])
    for name in ORDER:
        study = studies[name]
        scratch, areals = _curve(study, "scratch"), _curve(study, "pretrained")
        diffs = {(d["budget"], d["sensors"]): d for d in study["differences"]}
        for budget in sorted(scratch):
            d = diffs[(budget, 0)]
            t.rows.append([name, f"{budget:,}", fmt(scratch[budget]["mean_nrmse"], 4), fmt(areals[budget]["mean_nrmse"], 4),
                           f"{fmt(_gain(study, budget), 1)}%",
                           f"{d['pretrained_minus_scratch']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"])
        # Section 4.2: "It has the lower error at every budget"; Figure 1 and Appendix C: every paired
        # 95% interval of the difference excludes zero (both sensor settings in Appendix C).
        lower = [areals[b]["mean_nrmse"] < scratch[b]["mean_nrmse"] for b in scratch]
        t.check(f"{name}: % of budgets where AREALS has the lower error (\"every budget\")", 100 * sum(lower) / len(lower), 100, 0)
        excludes = [d["ci95"][1] < 0 or d["ci95"][0] > 0 for d in study["differences"]]
        t.check(f"{name}: % of paired 95% intervals excluding zero (\"every\")", 100 * sum(excludes) / len(excludes), 100, 0)
    w, m, n = (studies[k] for k in ORDER)
    t.check("WindsorML: gain with 18-35 simulations, low (\"36–39%\")", min(_gain(w, b) for b in (18, 35)), 36, 0)
    t.check("WindsorML: gain with 18-35 simulations, high (\"36–39%\")", max(_gain(w, b) for b in (18, 35)), 39, 0)
    t.check("DrivAerML: gain with up to 100 simulations, low (\"25–27%\")", min(_gain(m, b) for b in (25, 50, 100)), 25, 0)
    t.check("DrivAerML: gain with up to 100 simulations, high (\"25–27%\")", max(_gain(m, b) for b in (25, 50, 100)), 27, 0)
    t.check("DrivAerNet++: gain with 406 simulations (%)", _gain(n, 406), 8.7, 1)
    t.check("WindsorML: gain with the full training set (%)", _gain(w, max(_curve(w, "scratch"))), 22, 0)
    t.check("DrivAerML: gain with the full training set (%)", _gain(m, max(_curve(m, "scratch"))), 17.5, 1)
    t.check("DrivAerNet++: gain with the full training set (%)", _gain(n, max(_curve(n, "scratch"))), 6.4, 1)
    full = max(_curve(m, "scratch"))
    t.check("DrivAerML, 400 cars: AREALS nRMSE (Section 4.2)", _curve(m, "pretrained")[full]["mean_nrmse"], 0.181, 3)
    t.check("DrivAerML, 400 cars: scratch nRMSE (Section 4.2)", _curve(m, "scratch")[full]["mean_nrmse"], 0.219, 3)
    t.check("DrivAerML full training set is 400 cars", full, 400, 0)
    return t


def protocol(studies: dict) -> Table:
    t = Table("Appendix C (protocol)", "Splits, fits, late-improvement flags and independent replay",
              ["Dataset", "Train / validation / test", "Fits", "Updates per fit", "Seeds", "Bootstrap draws",
               "Still improving at 8,000 updates", "Replay: rescored rows, max abs. difference", "Summed fit GPU-hours"])
    paper_split = {"WindsorML": (274, 35, 35), "DrivAerML": (400, 34, 50), "DrivAerNet++": (6492, 778, 848)}
    paper_flags = {"WindsorML": 0, "DrivAerML": 6, "DrivAerNet++": 17}
    for name in ORDER:
        study = studies[name]
        for part, value, paper in zip(("train", "validation", "test"), study["split"], paper_split[name], strict=True):
            t.check(f"{name}: {part} simulations (Appendix C, Table A.1)", value, paper, 0)
        t.check(f"{name}: fits", study["runs"], 30, 0)
        per_fit = study["updates"] / study["runs"]
        t.check(f"{name}: updates per fit", per_fit, 8000, 0)
        seeds = sum(1 for k in study["rows"][0] if re.fullmatch(r"seed_\d+", k))
        t.check(f"{name}: seeds", seeds, 3, 0)
        draws = int(re.search(r"([\d,]+) draws", study["uncertainty"]).group(1).replace(",", ""))
        t.check(f"{name}: bootstrap draws", draws, 5000, 0)
        t.check(f"{name}: fits still improving late in training", study["flagged"], paper_flags[name], 0)
        replay = study["replay"]
        t.check(f"{name}: independent replay passed (1 = yes)", int(bool(replay["passed"])), 1, 0)
        t.rows.append([name, " / ".join(f"{v:,}" for v in study["split"]), study["runs"], f"{per_fit:,.0f}", seeds,
                       f"{draws:,}", f"{study['flagged']} of {study['runs']}",
                       f"{replay['scored_rows']:,}, {replay['max_metric_difference']:.1e}", fmt(study["gpu_hours"], 1)])
    t.notes.append("DrivAerML test count = replay rows / (sensor settings x fits); its validation count = "
                   "budget including validation minus training budget.")
    t.notes.append("Not checked here: 'four times' and 'a quarter of' the 40-epoch schedule (Section 4.2) refer to the "
                   "schedule of Table C.5; the 1,668-mesh prior reaching 0.25 with 35 WindsorML cars belongs to Table M.2.")
    return t


def tables() -> list[Table]:
    studies = _studies()
    return [figure_1(studies), thresholds(studies), gains(studies), protocol(studies)]
