"""Masked against unmasked geometry pretraining, transferred to pressure (Table C.4, Section 4.3, Table A.1).

Rebuilt from results/masked_vs_unmasked/ (results.json written by the scoring step, and the two per-problem
score tables it was computed from) and results/masked_vs_unmasked_extra/ (pins.json, which records the six
geometry checkpoints and dataset splits the fits used, and protocol.json).
Code: experiments/04_masked_vs_unmasked/.

Cells are hidden-region nRMSE averaged over three field seeds (each paired with geometry seed 17, 18 or 19)
and over test geometry families. Intervals are the stored joint seed-and-family bootstrap intervals
(5,000 draws; 97.5% at the pre-specified 20-sensor endpoint, 95% elsewhere). The bootstrap resamples
geometry families, whose labels are not stored in the per-problem tables, so the intervals are read, not
recomputed; the arm means and differences are cross-checked against the per-problem tables.
"""
from __future__ import annotations

import re

import numpy as np

from paper_tables._core import Table, fmt, load_csv, load_json

BENCHMARKS = [("doubledelta", "DoubleDelta, 18–19°"), ("drivaerml", "DrivAerML, official")]
# Observed sensor count K and the setting names written by the field benchmark (K = 0 observes nothing).
SETTINGS = [(0, "unobserved"), (5, "sensors-5"), (20, "sensors-20"), (100, "sensors-100"), (500, "sensors-500")]

# Table C.4 as printed: masked, unmasked, masked minus unmasked, interval low, interval high.
PAPER = {
    "doubledelta": {
        0: (0.4306, 0.4211, 0.0095, -0.0043, 0.0277),
        5: (0.3119, 0.3036, 0.0083, -0.0059, 0.0223),
        20: (0.2081, 0.2057, 0.0025, -0.0086, 0.0139),
        100: (0.1782, 0.1801, -0.0018, -0.0117, 0.0072),
        500: (0.1867, 0.1847, 0.0020, -0.0130, 0.0156),
    },
    "drivaerml": {
        0: (0.2403, 0.2412, -0.0009, -0.0041, 0.0028),
        5: (0.2403, 0.2413, -0.0010, -0.0042, 0.0028),
        20: (0.2406, 0.2415, -0.0009, -0.0044, 0.0034),
        100: (0.2418, 0.2428, -0.0010, -0.0042, 0.0027),
        500: (0.2477, 0.2487, -0.0010, -0.0042, 0.0027),
    },
}


def _csv_means(dataset: str) -> dict:
    """Per-seed arm means of hidden nRMSE from the per-problem score table, keyed by (arm, setting)."""
    rows = load_csv("masked_vs_unmasked", f"{dataset}_test_per_problem.csv")
    values: dict = {}
    for row in rows:
        arm, seed = row["method"].rsplit("_seed", 1)
        values.setdefault((arm, row["setting"], int(seed)), []).append(float(row["nrmse"]))
    seeds = sorted({k[2] for k in values})
    return {(arm, setting): np.array([np.mean(values[arm, setting, s]) for s in seeds])
            for arm, setting, _ in values}


def _single(values: set) -> float:
    """The one value shared by every entry, or NaN (a failed check) if they disagree."""
    return float(values.pop()) if len(values) == 1 else float("nan")


def table_c4(results: dict) -> Table:
    t = Table(
        key="Table C.4",
        title="Masked against unmasked geometry pretraining, transferred to pressure: hidden-region nRMSE "
        "(lower is better), three geometry seeds (17, 18, 19) at epoch 11, each fine-tuned once; joint "
        "seed-and-family bootstrap intervals (5,000 draws; 97.5% at K = 20, 95% otherwise)",
        columns=["Benchmark", "K", "Masked", "Unmasked", "Masked − unmasked [interval]"],
    )
    deviation = 0.0
    for dataset, label in BENCHMARKS:
        stored = results["datasets"][dataset]["settings"]
        csv = _csv_means(dataset)
        for k, setting in SETTINGS:
            cell = stored[setting]
            masked, unmasked = cell["masked_nrmse"], cell["unmasked_nrmse"]
            delta = cell["masked_minus_unmasked"]
            low, high = cell["joint_seed_and_geometry_interval"]
            p = PAPER[dataset][k]
            name = f"{label} K={k}"
            t.check(f"{name} masked", masked, p[0], 4)
            t.check(f"{name} unmasked", unmasked, p[1], 4)
            t.check(f"{name} masked − unmasked", delta, p[2], 4)
            t.check(f"{name} interval low", low, p[3], 4)
            t.check(f"{name} interval high", high, p[4], 4)
            level = f" ({100 * cell['interval_level']:g}%)" if setting == results["primary_setting"] else ""
            t.rows.append([label if k == 0 else "", str(k), fmt(masked, 4), fmt(unmasked, 4),
                           f"{delta:+.4f} [{low:+.4f}, {high:+.4f}]{level}"])
            # Cross-check: the stored means against the per-problem score tables.
            m, u = csv["masked", setting], csv["unmasked", setting]
            deviation = max(deviation,
                            abs(m.mean() - masked), abs(u.mean() - unmasked), abs((m - u).mean() - delta),
                            float(np.max(np.abs(m - np.asarray(cell["per_seed"]["masked"])))),
                            float(np.max(np.abs(u - np.asarray(cell["per_seed"]["unmasked"])))))
    t.notes.append(f"Arm means, per-seed means and differences recomputed from the per-problem score tables agree "
                   f"with results.json to {deviation:.1e} (largest absolute deviation).")
    if deviation > 1e-9:
        t.notes.append("MISMATCH between the per-problem score tables and results.json.")
    t.notes.append("Intervals are read from results.json; they resample geometry families, whose labels are not "
                   "stored in the per-problem tables.")
    return t


def table_protocol(results: dict) -> Table:
    t = Table(
        key="Table C.4 caption, Table A.1 row 'Masked vs. unmasked' and Section 4.3",
        title="Protocol numbers and the Section 4.3 claim",
        columns=["Quantity", "Rebuilt", "Paper", "Source (results/)"],
    )
    pins = load_json("masked_vs_unmasked_extra", "pins.json")
    protocol = load_json("masked_vs_unmasked_extra", "protocol.json")
    models = pins["models"].values()

    geometry_seeds = sorted({m["geometry_seed"] for m in models})
    epochs = {int(re.search(r"epoch_(\d+)\.pt$", m["path"]).group(1)) for m in models}
    updates = {m["updates"] for m in models}
    draws = int(re.search(r"(\d+) bootstrap draws", protocol["uncertainty"]).group(1))
    primary = int(re.fullmatch(r"sensors-(\d+)", results["primary_setting"]).group(1))
    settings = [s for d in results["datasets"].values() for s in d["settings"].items()]
    primary_level = _single({100 * c["interval_level"] for n, c in settings if n == results["primary_setting"]})
    other_level = _single({100 * c["interval_level"] for n, c in settings if n != results["primary_setting"]})

    doubledelta = next(d for d in protocol["data"] if d.startswith("Existing DoubleDelta"))
    train_angles = [int(a) for a in re.search(r"train(\d+)-(\d+)", doubledelta).groups()]
    test_angles = [int(a) for a in re.search(r"test(\d+)-(\d+)", doubledelta).groups()]
    counts = {name: d["official_test_counts"] for name, d in pins["datasets"].items()}
    # One grid per family and training angle (11-16 degrees), so families = training cases / training angles.
    train_families = counts["doubledelta"]["train"] / (train_angles[1] - train_angles[0] + 1)
    names = list(results["datasets"]["doubledelta"]["settings"])
    sensors = sorted(0 if n == "unobserved" else int(n.split("-")[1]) for n in names)
    field_seeds = _single({len(d["seeds"]) for d in results["datasets"].values()})

    excluding_zero = sum(not (c["joint_seed_and_geometry_interval"][0] <= 0 <= c["joint_seed_and_geometry_interval"][1])
                         for _, c in settings)

    rows = [
        ("C.4/A.1: first geometry seed", geometry_seeds[0], 17, 0, "masked_vs_unmasked_extra/pins.json"),
        ("C.4: second geometry seed", geometry_seeds[1], 18, 0, "masked_vs_unmasked_extra/pins.json"),
        ("C.4/A.1: last geometry seed", geometry_seeds[-1], 19, 0, "masked_vs_unmasked_extra/pins.json"),
        ("C.4: number of geometry seeds", len(geometry_seeds), 3, 0, "masked_vs_unmasked_extra/pins.json"),
        ("C.4/A.1: geometry checkpoint epoch", _single(epochs), 11, 0,
         "masked_vs_unmasked_extra/pins.json (checkpoint paths)"),
        ("C.4: geometry updates per checkpoint", _single(updates), 10626, 0, "masked_vs_unmasked_extra/pins.json"),
        ("C.4: bootstrap draws", draws, 5000, 0, "masked_vs_unmasked_extra/protocol.json"),
        ("C.4: pre-specified endpoint K", primary, 20, 0, "masked_vs_unmasked/results.json"),
        ("C.4: interval level at K = 20 (%)", primary_level, 97.5, 1, "masked_vs_unmasked/results.json"),
        ("C.4: interval level elsewhere (%)", other_level, 95, 0, "masked_vs_unmasked/results.json"),
        ("C.4/A.1: DoubleDelta first test angle (deg)", test_angles[0], 18, 0,
         "masked_vs_unmasked_extra/protocol.json"),
        ("C.4/A.1: DoubleDelta last test angle (deg)", test_angles[1], 19, 0,
         "masked_vs_unmasked_extra/protocol.json"),
        ("A.1: DoubleDelta training wing families", train_families, 228, 0,
         "masked_vs_unmasked_extra/pins.json, protocol.json"),
        ("A.1: DrivAerML training cars", counts["drivaerml"]["train"], 400, 0, "masked_vs_unmasked_extra/pins.json"),
        ("A.1: DoubleDelta test wings", results["datasets"]["doubledelta"]["test_geometries"], 16, 0,
         "masked_vs_unmasked/results.json"),
        ("A.1: DrivAerML test cars", results["datasets"]["drivaerml"]["test_geometries"], 50, 0,
         "masked_vs_unmasked/results.json"),
        ("A.1: fewest sensors", sensors[0], 0, 0, "masked_vs_unmasked/results.json (setting names)"),
        ("A.1: most sensors", sensors[-1], 500, 0, "masked_vs_unmasked/results.json (setting names)"),
        ("A.1: seeds", field_seeds, 3, 0, "masked_vs_unmasked/results.json"),
        ("4.3: intervals excluding zero ('every interval ... contains zero')", excluding_zero, 0, 0,
         "masked_vs_unmasked/results.json"),
    ]
    for label, value, paper, decimals, source in rows:
        rebuilt = t.check(label, value, paper, decimals)
        shown = fmt(rebuilt, decimals) if decimals else (f"{int(round(rebuilt)):,}" if np.isfinite(rebuilt) else "nan")
        t.rows.append([label, shown, f"{paper:,}", source])
    t.notes.append(f"DoubleDelta training cases {counts['doubledelta']['train']:,} over angles "
                   f"{train_angles[0]}–{train_angles[1]}°; test cases {counts['doubledelta']['test']} "
                   f"({results['datasets']['doubledelta']['test_geometries']} families x "
                   f"{test_angles[1] - test_angles[0] + 1} angles). "
                   f"DrivAerML official split {counts['drivaerml']['train']}/{counts['drivaerml']['val']}/"
                   f"{counts['drivaerml']['test']}.")
    return t


def tables() -> list[Table]:
    results = load_json("masked_vs_unmasked", "results.json")
    return [table_c4(results), table_protocol(results)]
