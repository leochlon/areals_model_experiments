"""Gaussian fields: where the posterior is exact (Section 4.4, Tables C.2 and C.3, Appendix C, Table A.1).

Rebuilt from results/gaussian_fields/ (the six-arm readout behind Table C.2 and the frozen-trunk
summary behind Table C.3) and results/gaussian_fields_extra/frozen_trunk_readout.json (the
frozen-trunk readout, which records the field-adaptation budget). Code: experiments/05_gaussian_fields/.

Efficiency is the share of the exact posterior mean's improvement over the prior that a network
recovers, (R_prior - R_model) / (R_prior - R_Bayes), averaged with equal weight over five test
observation settings (5, 20, 100 and 500 sensors; 32 patches with 60% of the area hidden).
"""
from __future__ import annotations

import re

import numpy as np

from paper_tables._core import Table, as_printed, fmt, load_json

ARMS = {
    "random": ("random_frozen", "random_full"),
    "areals": ("masked_native_frozen", "masked_native_full"),
}
# Paper row order of Table C.3, keyed by the setting names written by field_benchmark.py.
SETTINGS = [
    ("5 sensors", "sensors-5"),
    ("20 sensors", "sensors-20"),
    ("100 sensors", "sensors-100"),
    ("500 sensors", "sensors-500"),
    ("60% hidden patches", "patch-32-0.60"),
]


def _contrast(contrasts: list[dict], first: str, second: str) -> dict:
    return next(c for c in contrasts if c["first"] == first and c["second"] == second
                and c["setting"] == "equal_weight_all_settings")


def tables() -> list[Table]:
    readout = load_json("gaussian_fields", "mechanism_native_readout.json")
    summary = load_json("gaussian_fields", "frozen_trunk_summary.json")["test"]
    frozen_readout = load_json("gaussian_fields_extra", "frozen_trunk_readout.json")

    percent = readout["efficiency_percent"]
    gain = _contrast(readout["contrasts"], "masked_native_frozen", "random_frozen")
    gain_point = 100 * gain["efficiency_difference"]
    gain_low, gain_high = (100 * v for v in gain["seed_and_geometry_interval"])

    # Per-setting test efficiencies (%) of the frozen arms, in paper row order.
    per_setting = {
        arm: np.array([100 * summary["summary"][arm][key]["efficiency"] for _, key in SETTINGS])
        for arm in ("random_frozen", "masked_native_frozen")
    }

    # ---------------- Table C.2 ----------------
    c2 = Table(
        key="Table C.2",
        title="Gaussian fields: share of the exact posterior mean's gain over the prior recovered (%, higher is "
        "better), equal-weight mean over 5, 20, 100 and 500 sensors and 60% hidden patches; 16 held-out "
        "families, eight fields each, three seeds",
        columns=["Geometry trunk", "Frozen", "Fine-tuned"],
    )
    paper_c2 = {"random": (4.7, 25.4), "areals": (30.4, 82.6)}
    labels = {"random": "Random initialisation", "areals": "AREALS pretraining"}
    for key in ("random", "areals"):
        frozen_arm, full_arm = ARMS[key]
        frozen = c2.check(f"{labels[key]}, frozen", percent[frozen_arm], paper_c2[key][0], 1)
        full = c2.check(f"{labels[key]}, fine-tuned", percent[full_arm], paper_c2[key][1], 1)
        c2.rows.append([labels[key], fmt(frozen, 1), fmt(full, 1)])
    c2.rows.append([
        "Frozen: AREALS minus random (caption), 95% interval",
        f"{fmt(c2.check('caption: frozen pretraining gain (points)', gain_point, 25.7, 1), 1)} "
        f"({fmt(c2.check('caption: gain interval, lower', gain_low, 21.6, 1), 1)}–"
        f"{fmt(c2.check('caption: gain interval, upper', gain_high, 29.5, 1), 1)})",
        "",
    ])
    c2.notes.append("Cells: efficiency_percent in gaussian_fields/mechanism_native_readout.json (100 x the mean of the "
                    "five per-setting test efficiencies). Caption contrast: its masked_native_frozen minus random_frozen "
                    "entry; the interval jointly resamples the 16 geometry families and the three seeds "
                    "(10,000 draws, seed 91318).")
    # The frozen-only summary was produced separately; its frozen arms must agree with the six-arm readout.
    for arm in ("random_frozen", "masked_native_frozen"):
        same = abs(per_setting[arm].mean() - percent[arm]) < 1e-9
        c2.notes.append(f"{arm}: mean of gaussian_fields/frozen_trunk_summary.json = {per_setting[arm].mean():.6f}% "
                        f"({'identical to' if same else 'DIFFERS from'} the readout).")
    frozen_gain = _contrast(summary["contrasts"], "masked_native_frozen", "random_frozen")
    same = (abs(frozen_gain["efficiency_difference"] - gain["efficiency_difference"]) < 1e-12
            and np.allclose(frozen_gain["seed_and_geometry_interval"], gain["seed_and_geometry_interval"],
                            rtol=0, atol=1e-12))
    c2.notes.append("Frozen-trunk summary gives the same gain and interval: "
                    f"{100 * frozen_gain['efficiency_difference']:.4f} "
                    f"[{100 * frozen_gain['seed_and_geometry_interval'][0]:.4f}, "
                    f"{100 * frozen_gain['seed_and_geometry_interval'][1]:.4f}] "
                    f"({'identical' if same else 'DIFFERENT'}).")

    # ---------------- Table C.3 ----------------
    c3 = Table(
        key="Table C.3",
        title="Gaussian-field efficiency by observation setting with a frozen geometry trunk (%, higher is better)",
        columns=["Observations", "Random trunk", "AREALS pretraining"],
    )
    paper_c3 = {
        "5 sensors": (-0.2, 6.9),
        "20 sensors": (0.8, 18.5),
        "100 sensors": (2.8, 35.2),
        "500 sensors": (9.3, 54.8),
        "60% hidden patches": (10.7, 36.3),
    }
    for i, (label, _) in enumerate(SETTINGS):
        random_value = c3.check(f"{label}, random trunk", per_setting["random_frozen"][i], paper_c3[label][0], 1)
        areals_value = c3.check(f"{label}, AREALS", per_setting["masked_native_frozen"][i], paper_c3[label][1], 1)
        c3.rows.append([label, fmt(random_value, 1), fmt(areals_value, 1)])
    c3.rows.append([
        "Mean",
        fmt(c3.check("mean, random trunk", per_setting["random_frozen"].mean(), 4.7, 1), 1),
        fmt(c3.check("mean, AREALS", per_setting["masked_native_frozen"].mean(), 30.4, 1), 1),
    ])
    c3.notes.append("Source: test.summary.<arm>.<setting>.efficiency in gaussian_fields/frozen_trunk_summary.json; "
                    "each value pools 16 families x 8 fields x 3 seeds before taking the efficiency ratio.")

    # ---------------- Section 4.4 text ----------------
    s44 = Table(
        key="Section 4.4 'Where the posterior is exact'",
        title="Numbers quoted in the text",
        columns=["Claim", "Rebuilt", "Paper"],
    )
    for label, value, paper in [
        ("Fine-tuned AREALS recovers (%)", percent["masked_native_full"], 82.6),
        ("From scratch (random initialisation, fine-tuned) (%)", percent["random_full"], 25.4),
        ("Frozen AREALS trunk (%)", percent["masked_native_frozen"], 30.4),
        ("Frozen random trunk (%)", percent["random_frozen"], 4.7),
    ]:
        s44.rows.append([label, fmt(s44.check(f"Sec 4.4: {label}", value, paper, 1), 1), fmt(paper, 1)])
    # The text compares the two printed Table C.2 entries.
    ratio = as_printed(percent["masked_native_frozen"], 1) / as_printed(percent["random_frozen"], 1)
    s44.rows.append(["Frozen AREALS / frozen random ('six times')",
                     fmt(s44.check("Sec 4.4: frozen ratio ('six times')", ratio, 6, 0), 2), "6"])
    s44.rows.append(["Held-out shape families", str(summary["geometries"]), "16"])
    s44.check("Sec 4.4: held-out shape families", summary["geometries"], 16, 0)
    s44.notes.append("The ratio is of the printed Table C.2 entries; the unrounded means give "
                     f"{percent['masked_native_frozen'] / percent['random_frozen']:.2f}.")

    # ---------------- Protocol: Appendix C and Table A.1 ----------------
    protocol = Table(
        key="Appendix C 'Gaussian fields' and Table A.1",
        title="Protocol numbers",
        columns=["Quantity", "Rebuilt", "Paper", "Source (results/)"],
    )
    names = list(summary["summary"]["masked_native_frozen"])
    sensors = sorted(int(m.group(1)) for m in (re.fullmatch(r"sensors-(\d+)", n) for n in names) if m)
    hidden = [100 * float(m.group(2)) for m in (re.fullmatch(r"patch-(\d+)-([0-9.]+)", n) for n in names) if m]
    rows = [
        ("Held-out geometry families", summary["geometries"], 16, 0, "gaussian_fields/frozen_trunk_summary.json"),
        ("Held-out geometry families (six-arm readout)", readout["geometry_families"], 16, 0,
         "gaussian_fields/mechanism_native_readout.json"),
        ("Fresh fields per geometry", summary["fields_per_geometry"], 8, 0, "gaussian_fields/frozen_trunk_summary.json"),
        ("Fresh fields per geometry (six-arm readout)", readout["fresh_fields_per_geometry"], 8, 0,
         "gaussian_fields/mechanism_native_readout.json"),
        ("Seeds", summary["seeds"], 3, 0, "gaussian_fields/frozen_trunk_summary.json"),
        ("Seeds (six-arm readout)", len(readout["seeds"]), 3, 0, "gaussian_fields/mechanism_native_readout.json"),
        ("Field-adaptation updates", frozen_readout["field_adaptation_updates"], 9120, 0,
         "gaussian_fields_extra/frozen_trunk_readout.json"),
        ("Observation settings averaged with equal weight", len(names), 5, 0,
         "gaussian_fields/frozen_trunk_summary.json"),
        ("Fewest sensors", sensors[0], 5, 0, "gaussian_fields/frozen_trunk_summary.json (setting names)"),
        ("Second sensor count", sensors[1], 20, 0, "gaussian_fields/frozen_trunk_summary.json (setting names)"),
        ("Third sensor count", sensors[2], 100, 0, "gaussian_fields/frozen_trunk_summary.json (setting names)"),
        ("Most sensors", sensors[-1], 500, 0, "gaussian_fields/frozen_trunk_summary.json (setting names)"),
        ("Hidden share of patch setting (%)", hidden[0], 60, 0,
         "gaussian_fields/frozen_trunk_summary.json (setting names)"),
    ]
    for label, value, paper, decimals, source in rows:
        rebuilt = protocol.check(label, value, paper, decimals)
        protocol.rows.append([label, f"{int(round(rebuilt)):,}", f"{paper:,}", source])

    return [c2, c3, s44, protocol]
