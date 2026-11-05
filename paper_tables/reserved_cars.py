"""Reserved-car completion: Table C.5 (row 4), the reserved-split text of Sections 4.1 and 4.2 and Appendix E,
the Table A.1 row, and the 48-car part of the Appendix H benchmark controls.

Every value is read from results/reserved_cars/summary.json (written by the study's score.py) and from
results/reserved_cars_extra/: data_split.json (the 420/16/48 split that registry.py reads),
split_and_coordinate_audit.json (written by setup_evaluation.py) and raw_csv_and_log_audit.json (written by
audit_complete.py). Code: experiments/09_reserved_cars/.
"""
from __future__ import annotations

import math

import numpy as np

from paper_tables._core import Table, fmt, load_json

ALL25 = "all-25-original-settings"  # equal-weight average of the 25 original observation settings
UNOBSERVED = "unobserved"  # separate no-observation diagnostic, excluded from ALL25


def summary() -> dict:
    return load_json("reserved_cars", "summary.json")


def split() -> dict:
    return load_json("reserved_cars_extra", "data_split.json")


def split_audit() -> dict:
    return load_json("reserved_cars_extra", "split_and_coordinate_audit.json")


def log_audit() -> dict:
    return load_json("reserved_cars_extra", "raw_csv_and_log_audit.json")


def nrmse(s: dict, arm: str, setting: str = ALL25) -> float:
    return s["summaries"]["replication"][arm][setting]["nrmse"]["mean"]


def primary(s: dict) -> dict:
    """The pretrained-minus-scratch completion contrast."""
    return next(c for c in s["comparisons"] if c["family"] == "replication" and c["primary"])


def original_settings(s: dict) -> list[str]:
    """The 25 observation settings that enter the primary average."""
    return sorted(k for k in s["summaries"]["replication"]["pretrained"] if k not in (ALL25, UNOBSERVED))


def seed_means(s: dict, arm: str, setting: str = ALL25) -> list[float]:
    return next(r["means"] for r in s["seeds"] if r["family"] == "replication" and r["group"] == arm
                and r["setting"] == setting and r["metric"] == "nrmse")


def reduction(s: dict) -> float:
    """Relative mean-error reduction of AREALS against scratch, in percent."""
    return 100.0 * (1.0 - nrmse(s, "pretrained") / nrmse(s, "scratch"))


def split_counts() -> tuple[int, int, int]:
    d = split()
    return len(d["train"]), len(d["validation"]), len(d["benchmark_test_excluded"])


def _ceil_sig(value: float, exponent: int, digits: int) -> float:
    """`value` in units of 10**exponent, rounded up to `digits` decimals: how the paper prints an upper bound."""
    scale = 10 ** digits
    return math.ceil(value / 10 ** exponent * scale - 1e-9) / scale


def _signed(value: float, decimals: int = 4) -> str:
    return fmt(value, decimals).replace("-", "−")


# ---------------------------------------------------------------------------------------------------------------
# Table C.5, row 4
# ---------------------------------------------------------------------------------------------------------------
def table_c5() -> Table:
    s = summary()
    t = Table("Table C.5 (row 4)",
              "Geometry pretraining against the same architecture from scratch on the reserved DrivAerML cars: "
              "hidden-region nRMSE, three seeds",
              ["Benchmark", "Train", "Scratch", "AREALS", "Reduction"])
    train, _, test = split_counts()
    n = t.check("train cars", train, 420, 0)
    scratch = t.check("scratch nRMSE", nrmse(s, "scratch"), 0.284, 3)
    areals = t.check("AREALS nRMSE", nrmse(s, "pretrained"), 0.226, 3)
    red = t.check("reduction (%)", reduction(s), 20.2, 1)
    t.rows.append(["DrivAerML, reserved*", f"{int(n)}", fmt(scratch), fmt(areals), f"{fmt(red, 1)}%"])

    settings = original_settings(s)
    t.check("footnote: reserved test cars", test, 48, 0)
    t.check("footnote: reserved test cars scored", s["test_cars"], 48, 0)
    t.check("footnote: observation settings averaged", len(settings), 25, 0)
    c = primary(s)
    t.check("caption: seeds", len(c["per_seed_differences"]), 3, 0)

    d = c["first_minus_second"]
    lo, hi = c["relative_reduction_95"]
    t.notes.append(f"* {test} reserved cars, averaged over all {len(settings)} observation settings. AREALS − scratch "
                   f"{_signed(d['mean'])} [{_signed(d['low'])}, {_signed(d['high'])}] (car bootstrap, 4,000 draws, "
                   f"seeds kept within car); the interval excludes zero: {d['high'] < 0}. Relative reduction "
                   f"95% interval [{100 * lo:.1f}%, {100 * hi:.1f}%].")
    t.notes.append("Per-seed AREALS − scratch: " + ", ".join(_signed(x) for x in c["per_seed_differences"])
                   + "; per-seed AREALS " + ", ".join(fmt(x, 4) for x in c["per_seed_means_first"])
                   + "; scratch " + ", ".join(fmt(x, 4) for x in c["per_seed_means_second"]) + ".")
    # The primary score is an equal-weight average of the 25 settings within each car; since every car has all
    # 25 settings, it equals the average of the 25 per-setting means.
    gaps = [abs(float(np.mean([nrmse(s, arm, k) for k in settings])) - nrmse(s, arm)) for arm in ("pretrained", "scratch")]
    seed_gap = max(abs(float(np.mean(seed_means(s, arm))) - nrmse(s, arm)) for arm in ("pretrained", "scratch"))
    stored = 100.0 * c["relative_mean_reduction"]
    csv = log_audit()["means_from_csv"]
    t.notes.append(f"Consistency: the mean of the 25 per-setting means matches the stored average within "
                   f"{max(gaps):.1e}; the mean of the three seed means within {seed_gap:.1e}; the stored relative "
                   f"reduction is {stored:.3f}% (rebuilt {reduction(s):.3f}%); the independent CSV reduction gives "
                   f"{csv['replication/pretrained']:.6f} and {csv['replication/scratch']:.6f}.")
    unobs = [nrmse(s, arm, UNOBSERVED) for arm in ("scratch", "pretrained")]
    t.notes.append(f"No-observation diagnostic (reported separately, not in the average): scratch {fmt(unobs[0])}, "
                   f"AREALS {fmt(unobs[1])}.")
    return t


# ---------------------------------------------------------------------------------------------------------------
# Sections 4.1 and 4.2, Appendix E and Table A.1
# ---------------------------------------------------------------------------------------------------------------
def table_text() -> Table:
    s = summary()
    t = Table("Sections 4.1, 4.2, Appendix E and Table A.1",
              "Numbers quoted in the text about the reserved-car split and protocol",
              ["Claim", "Rebuilt", "Paper"])

    def row(label, value, paper, decimals, unit=""):
        t.check(label, value, paper, decimals)
        t.rows.append([label, f"{fmt(value, decimals)}{unit}", f"{fmt(paper, decimals)}{unit}"])

    train, val, test = split_counts()
    settings = len(original_settings(s))
    seeds = len(primary(s)["per_seed_differences"])
    row("4.1 DrivAerML 420/16/48 split: training cars", train, 420, 0)
    row("4.1 DrivAerML 420/16/48 split: development cars", val, 16, 0)
    row("4.1 DrivAerML 420/16/48 split: test (reserved) cars", test, 48, 0)
    row("4.2 error reduction on the 420-car split", reduction(s), 20.2, 1, "%")
    row("4.2 training cars of that split", train, 420, 0)
    row("E reserved-car completion: training cars", train, 420, 0)
    row("E reserved-car completion: development cars", val, 16, 0)
    row("E reserved-car completion: test cars", test, 48, 0)
    row("E settings averaged within each car", settings, 25, 0)
    row("E seeds", seeds, 3, 0)
    row("A.1 reserved-car completion: training data (cars)", train, 420, 0)
    row("A.1 reserved-car completion: evaluation (reserved cars)", s["test_cars"], 48, 0)
    row("A.1 reserved-car completion: observation settings", settings, 25, 0)
    row("A.1 reserved-car completion: seeds", seeds, 3, 0)

    d = split()
    tr, va, te = (set(d[k]) for k in ("train", "validation", "benchmark_test_excluded"))
    audit = split_audit()
    t.notes.append(f"Split disjoint (train/development/test): {not (tr & va or tr & te or va & te)}. The split audit "
                   f"counts {audit['counts']} in the evaluation manifest, finds "
                   f"{len(audit['duplicate_test_coordinate_sets'])} test records whose surface-coordinate set duplicates "
                   f"another record, and passed: {audit['passed']}.")
    c = primary(s)
    t.notes.append(f"4.2 'every paired interval excluding zero': the reserved-car interval is "
                   f"[{_signed(c['first_minus_second']['low'])}, {_signed(c['first_minus_second']['high'])}].")
    t.notes.append(f"A.1 'None' observation setting: the no-observation diagnostic is stored as '{UNOBSERVED}' "
                   f"({UNOBSERVED in s['summaries']['replication']['pretrained']}) and scored separately.")
    return t


# ---------------------------------------------------------------------------------------------------------------
# Appendix H, 48-car study
# ---------------------------------------------------------------------------------------------------------------
def table_h() -> Table:
    s = summary()
    audit = log_audit()
    t = Table("Appendix H (48-car study)", "Benchmark controls on the reserved cars",
              ["Control", "Result", "Source"])
    checks = sum(a["hidden_value_checks"] for a in s["artifacts"])
    maximum = max(a["independent_maximum_difference"] for a in s["artifacts"])
    worst = max(s["artifacts"], key=lambda a: a["independent_maximum_difference"])["model"]
    t.check("hidden-value perturbation: study size (cars)", s["test_cars"], 48, 0)
    t.check("hidden-value perturbation: checks passed", checks, 264, 0)
    t.check("hidden-value perturbation: checks passed (log audit)", audit["hidden_pressure_checks_passed"], 264, 0)
    t.check("independent recomputation: max |difference|, x1e-16, rounded up to 2 s.f. (paper: within 3.4e-16)",
            _ceil_sig(maximum, -16, 1), 3.4, 1)

    families: dict[str, list[int]] = {}
    for r in audit["check_counts"]:
        families.setdefault(r["family"], []).append(r["passed"])
    breakdown = ", ".join(f"{f} {len(v)} x {v[0]}" if len(set(v)) == 1 else f"{f} {sum(v)}"
                          for f, v in sorted(families.items()))
    t.rows += [
        ["Hidden-value perturbation (all 48 models)",
         f"{checks} pass ({breakdown}); {audit['hidden_pressure_checks_failed']} fail",
         "reserved_cars/summary.json; reserved_cars_extra/raw_csv_and_log_audit.json"],
        ["Independent recomputation of every score from saved predictions, targets and area weights",
         f"max abs. difference {maximum:.3g} ({worst}) over {len(s['artifacts'])} models",
         "reserved_cars/summary.json"],
        ["Independent CSV reduction of the summary means",
         f"max disagreement {audit['maximum_mean_disagreement']:.2g} over {len(audit['means_from_csv'])} groups",
         "reserved_cars_extra/raw_csv_and_log_audit.json"],
    ]
    t.notes.append(f"The paper prints the maximum as an upper bound; the check rounds the stored maximum up "
                   f"({maximum:.4g} -> {_ceil_sig(maximum, -16, 1):g}e-16).")
    frozen = len(families.get("frozen", []))
    t.notes.append(f"The {s['models'] - frozen} field-completion and geometry-only models were checked with the "
                   "harness's default tolerance (1e-5 of the field spread; run_task.py passes no --tolerance), stricter "
                   f"than the 10^-3 stated in Appendix H; the four checks per frozen-surrogate model ({frozen} models) "
                   "perturb the target pressures and require an exact zero change.")
    return t


def tables() -> list[Table]:
    return [table_c5(), table_text(), table_h()]
