"""Distance tubes: the distance target reproduces topology (Section 4.6, Appendix J, Tables J.1 and J.2).

Rebuilt from results/distance_tubes/. Code: experiments/13_distance_tubes/.

complete_inputs/  summary.json (tube maxima per group, field, grid and radius), fields_summary.json (largest
                  over- and underestimate of the disk distance on the 97^3 grid, per shape) and
                  persistence.csv (bottleneck distances to the exact distance's diagrams). Field "disk" is
                  the decoder's output for a complete input (the native distance to the 2,048 token disks,
                  selected by complete = 1, see decoder.patch); field "full" is the learned distance the
                  decoder returned for complete inputs before that routing, which the paper does not report.
partial_inputs/   persistence.csv for 60% of the area hidden, where "full" is the learned distance (AREALS)
                  and "disk" the disks built from the visible surface, and surface_queries/<shape>.npz: the
                  16,384 surface samples of each shape with the visible mask, area weights, and both
                  distances in units of the complete shape's RMS radius.

results/distance_tubes_extra/ holds the records behind the Appendix J protocol and the Table A.1 row:
complete_inputs/protocol.json and metrics_protocol.json (the complete-input protocol and persistence cap),
complete_inputs/metadata/<shape>.json (per-shape grid spacing, scale, query count and analytic dimensions, written by
sample_fields.py), partial_inputs/protocol.json (mask and persistence cap) and partial_inputs/cases.json (realised
encoder points and disk centres per masked shape, written by run_fields.py).

Table J.2, the hidden-area medians and the visible-area share are recomputed here from surface_queries/*.npz:
no script for this aggregation was saved. The rule is the area-weighted share of the hidden (or visible)
samples whose distance falls strictly below the threshold.

Prose upper bounds ("at most", "within", "never ... by more than", "beyond") are checked as the stored
maximum rounded up at the printed precision, and the lower bounds "at least" as the stored minimum rounded
down; table cells, ranges and bottleneck distances are checked rounded to nearest.
"""
from __future__ import annotations

import math

import numpy as np

from paper_tables._core import RESULTS, Table, fmt, load_csv, load_json

COMPLETE = ("distance_tubes", "complete_inputs")
PARTIAL = ("distance_tubes", "partial_inputs")
EXTRA_COMPLETE = ("distance_tubes_extra", "complete_inputs")
EXTRA_PARTIAL = ("distance_tubes_extra", "partial_inputs")
ANALYTIC_LABELS = {"sphere": "Sphere", "torus": "Torus", "two_spheres": "Two spheres"}
# Coverage thresholds of Table J.2, in RMS radii (protocol parameters, not results).
THRESHOLDS = (0.05, 0.1)


def _up(value: float, decimals: int) -> float:
    """Round up at `decimals`: how the paper prints an upper bound."""
    return math.ceil(value * 10 ** decimals - 1e-9) / 10 ** decimals


def _down(value: float, decimals: int) -> float:
    """Round down at `decimals`: how the paper prints a lower bound."""
    return math.floor(value * 10 ** decimals + 1e-9) / 10 ** decimals


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def _share_below(values: np.ndarray, weights: np.ndarray, threshold: float) -> float:
    """Area-weighted share (%) of the samples whose distance is below `threshold`."""
    return float(100 * weights[values < threshold].sum() / weights.sum())


def _surface_coverage(groups: dict[str, str]) -> dict[str, dict]:
    """Per shape: hidden-area coverage, hidden medians and visible-area share, from surface_queries/*.npz."""
    out = {}
    for shape in groups:
        z = np.load(RESULTS.joinpath(*PARTIAL, "surface_queries", f"{shape}.npz"))
        weights = z["weights"].astype(np.float64)
        visible = z["visible"].astype(bool)
        hidden = ~visible
        row = {
            "samples": len(weights),
            "hidden_share": float(100 * weights[hidden].sum() / weights.sum()),
            "visible_below_first": _share_below(z["predicted"][visible].astype(np.float64), weights[visible],
                                                THRESHOLDS[0]),
        }
        for name, key in (("areals", "predicted"), ("disks", "disk")):
            values = z[key][hidden].astype(np.float64)
            for t in THRESHOLDS:
                row[(name, t)] = _share_below(values, weights[hidden], t)
            row[(name, "median")] = _weighted_median(values, weights[hidden])
        out[shape] = row
    return out


def _bottleneck_maxima(rows: list[dict], grid: str, field: str, groups: set[str]) -> list[float]:
    """Largest capped bottleneck distance over shapes, per homology dimension 0, 1, 2."""
    return [max(float(r["bottleneck_capped"]) for r in rows if r["grid_n"] == grid and r["field"] == field
                and r["group"] in groups and r["dimension"] == str(k)) for k in range(3)]


def _first_radius_from_which(rows: list[dict], ok) -> float:
    """Smallest radius r such that `ok(row)` holds at r and at every larger radius."""
    rows = sorted(rows, key=lambda r: r["radius"])
    start = None
    for r in rows:
        if ok(r):
            start = r["radius"] if start is None else start
        else:
            start = None
    return start


def tables() -> list[Table]:
    summary = load_json(*COMPLETE, "summary.json")
    fields_summary = load_json(*COMPLETE, "fields_summary.json")
    complete_persistence = load_csv(*COMPLETE, "persistence.csv")
    partial_persistence = load_csv(*PARTIAL, "persistence.csv")

    fine_grid = max(int(r["grid_n"]) for r in summary)
    radii = sorted({r["radius"] for r in summary})

    def tubes(group: str) -> list[dict]:
        return sorted((r for r in summary if r["group"] == group and r["field"] == "disk"
                       and r["grid_n"] == fine_grid), key=lambda r: r["radius"])

    analytic, meshes = tubes("control"), tubes("heldout")
    by_radius = {r["radius"]: r for r in meshes}

    def path_percent(r: dict) -> float:
        # A lost landmark pair makes the path error infinite.
        return math.inf if r["path_error_infinite"] else 100 * r["max_finite_path_relative_error"]

    # The analytic tubes have exact Betti numbers from this radius on; Table J.1 lists those radii.
    exact_from = _first_radius_from_which(analytic, lambda r: r["topology_matching_meshes"] == r["shapes"])
    connected_from = _first_radius_from_which(meshes, lambda r: r["lost_pairs"] == 0)

    # ---------------- Table J.1 ----------------
    j1 = Table(
        key="Table J.1",
        title="Complete inputs: tubes of the decoder's distance against the exact distance's tubes at the same "
        "radius, maxima over shapes (analytic shapes: sphere, torus, two spheres; meshes: the eight held-out "
        "meshes)",
        columns=["Radius r", "Analytic: exact Betti numbers", "Analytic: area error (%)",
                 "Analytic: path error (%)", "Meshes: area error (%)"],
    )
    paper_j1 = [  # r, Betti matches, of, analytic area, analytic path, mesh area
        (0.10, 3, 3, 1.3, 2.4, 7.1),
        (0.14, 3, 3, 0.8, 0.4, 5.6),
        (0.20, 3, 3, 0.3, 0.0, 4.9),
        (0.28, 3, 3, 0.1, 0.1, 3.5),
        (0.40, 3, 3, 0.1, 0.5, 2.9),
    ]
    rows_j1 = [r for r in analytic if r["radius"] >= exact_from]
    if len(rows_j1) != len(paper_j1):
        j1.notes.append(f"MISMATCH: {len(rows_j1)} radii with exact analytic Betti numbers, paper lists {len(paper_j1)}")
    for row, (pr, pm, pn, pa, pp, pmesh) in zip(rows_j1, paper_j1):
        r = row["radius"]
        mesh = by_radius[r]
        j1.rows.append([
            fmt(j1.check(f"r = {pr:.2f}: radius", r, pr, 2), 2),
            f"{j1.check(f'r = {pr:.2f}: analytic shapes with exact Betti numbers', row['topology_matching_meshes'], pm, 0):.0f}"
            f" of {j1.check(f'r = {pr:.2f}: analytic shapes', row['shapes'], pn, 0):.0f}",
            fmt(j1.check(f"r = {pr:.2f}: analytic area error (%)", 100 * row["max_area_relative_error"], pa, 1), 1),
            fmt(j1.check(f"r = {pr:.2f}: analytic path error (%)", path_percent(row), pp, 1), 1),
            fmt(j1.check(f"r = {pr:.2f}: mesh area error (%)", 100 * mesh["max_area_relative_error"], pmesh, 1), 1),
        ])
    j1.notes.append(f"Cells: complete_inputs/summary.json, field disk, grid {fine_grid}; groups control (analytic) and "
                    f"heldout (meshes). Area error is the relative error of the tube-volume area estimate "
                    f"Vol{{d <= r}}/(2r); path error is the largest relative error of the 26-connected landmark paths "
                    f"over reference-connected pairs. The rows are the radii from which every analytic tube has the "
                    f"exact Betti numbers (r >= {exact_from:g}).")
    smaller = [r for r in analytic if r["radius"] < exact_from]
    j1.notes.append("Smaller radii, analytic (area %, path %, shapes with exact Betti numbers): " + "; ".join(
        f"r={r['radius']:g}: {100 * r['max_area_relative_error']:.1f}, {path_percent(r):.1f}, "
        f"{r['topology_matching_meshes']}/{r['shapes']}" for r in smaller) + ".")

    # ---------------- Appendix J, complete inputs (text) ----------------
    ct = Table(
        key="Appendix J (complete inputs)",
        title="Complete inputs: distance to the 2,048 token disks against the exact distance",
        columns=["Claim", "Rebuilt", "Paper"],
    )

    def claim(table: Table, text: str, value: float, paper: float, decimals: int, shown: str | None = None):
        table.check(text, value, paper, decimals)
        table.rows.append([text, shown if shown is not None else fmt(value, decimals), fmt(paper, decimals)])

    groups = {r["id"]: r["group"] for r in fields_summary}
    over_analytic = max(r["disk_max_overestimate"] for r in fields_summary if r["group"] == "control")
    under_analytic = max(r["disk_max_underestimate"] for r in fields_summary if r["group"] == "control")
    under_meshes = max(r["disk_max_underestimate"] for r in fields_summary if r["group"] == "heldout")
    after = [r for r in analytic if r["radius"] >= exact_from]
    claim(ct, "Analytic: disk distance exceeds the exact one by at most (RMS radii, max rounded up)",
          _up(over_analytic, 2), 0.07, 2, f"{_up(over_analytic, 2):.2f} (max {over_analytic:.4f})")
    claim(ct, "Analytic: never falls below the exact one by more than (max rounded up)",
          _up(under_analytic, 3), 0.001, 3, f"{_up(under_analytic, 3):.3f} (max {under_analytic:.5f})")
    claim(ct, "Analytic: every tube has the exact Betti numbers from r =", exact_from, 0.1, 1)
    area_after = 100 * max(r["max_area_relative_error"] for r in after)
    path_after = max(path_percent(r) for r in after)
    claim(ct, "Analytic, from that radius: area within (%, max rounded up)", _up(area_after, 1), 1.4, 1,
          f"{_up(area_after, 1):.1f} (max {area_after:.3f})")
    claim(ct, "Analytic, from that radius: path lengths within (%, max rounded up)", _up(path_after, 1), 2.4, 1,
          f"{_up(path_after, 1):.1f} (max {path_after:.3f})")
    claim(ct, "Meshes: disk distance never falls below the exact one by more than (max rounded up)",
          _up(under_meshes, 3), 0.045, 3, f"{_up(under_meshes, 3):.3f} (max {under_meshes:.4f})")
    for start, paper in ((0.1, 7.1), (0.28, 3.5)):
        worst = 100 * max(r["max_area_relative_error"] for r in meshes if r["radius"] >= start)
        claim(ct, f"Meshes: tube volume within (%, max rounded up) from r = {start:g}", _up(worst, 1), paper, 1,
              f"{_up(worst, 1):.1f} (max {worst:.3f})")
    claim(ct, "Meshes: every landmark pair stays connected from r =", connected_from, 0.14, 2)
    worst_path = max(path_percent(r) for r in meshes if r["radius"] >= 0.2)
    claim(ct, "Meshes: path lengths agree within (%, max rounded up) from r = 0.2", _up(worst_path, 1), 3.6, 1,
          f"{_up(worst_path, 1):.1f} (max {worst_path:.3f})")
    complete_protocol = load_json(*EXTRA_COMPLETE, "protocol.json")
    claim(ct, "Complete input: token disks per shape (surfels)", complete_protocol["surfels"], 2048, 0,
          f"{complete_protocol['surfels']:,}")
    complete_shapes = {r["id"] for r in complete_persistence}
    claim(ct, "Bottleneck comparison: shapes", len(complete_shapes), 11, 0)
    bottleneck = _bottleneck_maxima(complete_persistence, str(fine_grid), "disk", {"control", "heldout"})
    for k, paper in enumerate((0.063, 0.051, 0.101)):
        claim(ct, f"All eleven shapes: bottleneck distance, H{k}", bottleneck[k], paper, 3, f"{bottleneck[k]:.4f}")
    epsilon = max(bottleneck)
    claim(ct, "Features persisting beyond 2 x max bottleneck (rounded up) are reproduced", _up(2 * epsilon, 2), 0.21, 2,
          f"{_up(2 * epsilon, 2):.2f} (2 x {epsilon:.4f})")
    claim(ct, "... with birth and death within max bottleneck (rounded up)", _up(epsilon, 2), 0.11, 2,
          f"{_up(epsilon, 2):.2f} ({epsilon:.4f})")
    claim(ct, "No added feature persists beyond 2 x max bottleneck (rounded up)", _up(2 * epsilon, 2), 0.21, 2,
          f"{_up(2 * epsilon, 2):.2f}")
    ct.notes.append("Over- and underestimates: complete_inputs/fields_summary.json (disk_max_overestimate, "
                    f"disk_max_underestimate) on the {fine_grid}^3 grid. Bottleneck distances: complete_inputs/"
                    f"persistence.csv, field disk, grid {fine_grid}, column bottleneck_capped (sublevel filtrations up "
                    "to r = 0.4), maximum over all eleven shapes; the uncapped column gives the same maxima. The "
                    "0.21 and 0.11 follow from the stability theorem: every feature of persistence above 2 eps is "
                    "matched with birth and death within eps, where eps is the largest bottleneck distance.")
    ct.notes.append("The prose bounds are upper bounds on the stored maxima. Rounded to nearest, three of them would "
                    f"print lower: area {area_after:.2f}% (Table J.1 prints 1.3), mesh underestimate {under_meshes:.4f} "
                    f"and mesh paths from r = 0.2 {worst_path:.2f}%.")

    # ---------------- Table J.2 ----------------
    partial_groups = {r["id"]: r["group"] for r in partial_persistence}
    coverage = _surface_coverage(partial_groups)
    mesh_ids = sorted(s for s, g in partial_groups.items() if g == "heldout")
    analytic_ids = [s for s in ANALYTIC_LABELS if partial_groups.get(s) == "control"]
    j2 = Table(
        key="Table J.2",
        title="Partial inputs, 60% of the area hidden: share of the hidden area (%) over which the distance falls "
        "below 0.05 and 0.1 RMS radii, for AREALS and for disks built from the visible surface",
        columns=["Shape", "Below 0.05: AREALS", "Below 0.05: visible disks", "Below 0.1: AREALS",
                 "Below 0.1: visible disks"],
    )
    paper_j2 = {
        "Mesh 1": (96.1, 9.1, 99.5, 23.9),
        "Mesh 2": (92.2, 11.5, 95.2, 30.3),
        "Mesh 3": (78.9, 9.2, 83.1, 23.4),
        "Mesh 4": (97.6, 8.9, 99.0, 20.3),
        "Mesh 5": (93.7, 16.4, 97.6, 37.2),
        "Mesh 6": (97.1, 8.5, 99.4, 23.1),
        "Mesh 7": (86.3, 12.5, 93.9, 31.9),
        "Mesh 8": (94.7, 7.8, 97.1, 18.9),
        "Sphere": (97.6, 10.8, 100.0, 23.4),
        "Torus": (100.0, 12.1, 100.0, 27.1),
        "Two spheres": (100.0, 11.9, 100.0, 25.9),
    }
    labelled = [(f"Mesh {i + 1}", s) for i, s in enumerate(mesh_ids)] + [(ANALYTIC_LABELS[s], s) for s in analytic_ids]
    columns = [("areals", THRESHOLDS[0]), ("disks", THRESHOLDS[0]), ("areals", THRESHOLDS[1]), ("disks", THRESHOLDS[1])]
    for label, shape in labelled:
        cells = [fmt(j2.check(f"{label}: {name} below {t:g}", coverage[shape][(name, t)], paper, 1), 1)
                 for (name, t), paper in zip(columns, paper_j2[label])]
        j2.rows.append([label] + cells)
    j2.notes.append("Recomputed from partial_inputs/surface_queries/<shape>.npz (no aggregation script was saved): "
                    "area-weighted share of the hidden samples (~visible, weights) whose distance (predicted for "
                    "AREALS, disk for the visible disks) is below the threshold. Meshes 1-8 are the held-out "
                    "meshes in identifier order: " + ", ".join(s[:8] for s in mesh_ids) + ".")

    # ---------------- Appendix J, partial inputs (text) ----------------
    pt = Table(
        key="Appendix J (partial inputs)",
        title="Partial inputs, 60% of the area hidden: coverage, medians and bottleneck distances",
        columns=["Claim", "Rebuilt", "Paper"],
    )

    def mesh_values(key):
        return [coverage[s][key] for s in mesh_ids]

    a05, a10 = mesh_values(("areals", THRESHOLDS[0])), mesh_values(("areals", THRESHOLDS[1]))
    d05, d10 = mesh_values(("disks", THRESHOLDS[0])), mesh_values(("disks", THRESHOLDS[1]))
    claim(pt, "Meshes: AREALS below 0.05, lowest share of hidden area (%)", min(a05), 79, 0, f"{min(a05):.1f}")
    claim(pt, "Meshes: AREALS below 0.05, highest share (%)", max(a05), 98, 0, f"{max(a05):.1f}")
    sixth = _down(sorted(a05, reverse=True)[5], 0)
    claim(pt, "Meshes: sixth-highest AREALS share below 0.05, rounded down (at least ...%)", sixth, 92, 0)
    claim(pt, "... number of meshes at or above it (on ... of the eight)", sum(v >= sixth for v in a05), 6, 0)
    claim(pt, "Meshes: AREALS below 0.1, lowest share (%)", min(a10), 83, 0, f"{min(a10):.1f}")
    claim(pt, "Meshes: AREALS below 0.1, highest share (%)", max(a10), 99.5, 1, f"{max(a10):.2f}")
    claim(pt, "Meshes: visible disks below 0.05, lowest share (%)", min(d05), 8, 0, f"{min(d05):.1f}")
    claim(pt, "Meshes: visible disks below 0.05, highest share (%)", max(d05), 16, 0, f"{max(d05):.1f}")
    claim(pt, "Meshes: visible disks below 0.1, lowest share (%)", min(d10), 19, 0, f"{min(d10):.1f}")
    claim(pt, "Meshes: visible disks below 0.1, highest share (%)", max(d10), 37, 0, f"{max(d10):.1f}")
    an05 = [coverage[s][("areals", THRESHOLDS[0])] for s in analytic_ids]
    claim(pt, "Analytic shapes: AREALS below 0.05, lowest share (%)", min(an05), 98, 0, f"{min(an05):.1f}")
    claim(pt, "Analytic shapes: AREALS below 0.05, highest share (%)", max(an05), 100, 0, f"{max(an05):.1f}")
    am, dm = mesh_values(("areals", "median")), mesh_values(("disks", "median"))
    claim(pt, "Meshes: AREALS median over the hidden surface, lowest", min(am), 0.018, 3, f"{min(am):.4f}")
    claim(pt, "Meshes: AREALS median over the hidden surface, highest", max(am), 0.025, 3, f"{max(am):.4f}")
    claim(pt, "Meshes: visible-disk median over the hidden surface, lowest", min(dm), 0.13, 2, f"{min(dm):.4f}")
    claim(pt, "Meshes: visible-disk median over the hidden surface, highest", max(dm), 0.24, 2, f"{max(dm):.4f}")
    visible = min(coverage[s]["visible_below_first"] for s in coverage)
    claim(pt, "All shapes: AREALS below 0.05 over at least (% of visible area, min rounded down)",
          _down(visible, 1), 99.8, 1, f"{_down(visible, 1):.1f} (min {visible:.3f})")
    held = _bottleneck_maxima(partial_persistence, str(fine_grid), "full", {"heldout"})
    disk = _bottleneck_maxima(partial_persistence, str(fine_grid), "disk", {"heldout"})
    for k, (pa, pd) in enumerate(((0.053, 0.200), (0.090, 0.110), (0.119, 0.190))):
        claim(pt, f"Meshes: bottleneck distance H{k}, AREALS", held[k], pa, 3, f"{held[k]:.4f}")
        claim(pt, f"Meshes: bottleneck distance H{k}, visible disks", disk[k], pd, 3, f"{disk[k]:.4f}")
    pt.notes.append("Coverage and medians: recomputed from partial_inputs/surface_queries/*.npz as for Table J.2; the "
                    "median is the area-weighted median over the hidden samples. Analytic-shape medians (not in the "
                    "paper's range): " + ", ".join(f"{ANALYTIC_LABELS[s]} {coverage[s][('areals', 'median')]:.3f} vs "
                                                   f"{coverage[s][('disks', 'median')]:.3f}" for s in analytic_ids)
                    + ".")
    pt.notes.append(f"Bottleneck distances: partial_inputs/persistence.csv, group heldout, grid {fine_grid}, column "
                    "bottleneck_capped (filtrations capped at r = 0.4), maximum over the eight meshes; field full is "
                    "the learned distance, field disk the visible disks.")

    # ---------------- Protocol and Section 4.6 ----------------
    sec = Table(
        key="Section 4.6 and Appendix J protocol",
        title="'The distance target reproduces topology' and the Appendix J protocol",
        columns=["Claim", "Rebuilt", "Paper"],
    )
    claim(sec, "Sec 4.6: features persisting beyond (RMS radii) are reproduced", _up(2 * epsilon, 2), 0.21, 2)
    claim(sec, "Sec 4.6: ... on all test shapes", len(complete_shapes), 11, 0)
    hidden_mean = float(np.mean([coverage[s]["hidden_share"] for s in coverage]))
    claim(sec, "Sec 4.6: area hidden (%, mean realised share over the eleven shapes)", hidden_mean, 60, 0,
          f"{hidden_mean:.1f}")
    claim(sec, "Sec 4.6: learned distance below 0.05, lowest mesh share (%)", min(a05), 79, 0, f"{min(a05):.1f}")
    claim(sec, "Sec 4.6: learned distance below 0.05, highest mesh share (%)", max(a05), 98, 0, f"{max(a05):.1f}")
    claim(sec, "Sec 4.6: visible disks below 0.05, lowest mesh share (%)", min(d05), 8, 0, f"{min(d05):.1f}")
    claim(sec, "Sec 4.6: visible disks below 0.05, highest mesh share (%)", max(d05), 16, 0, f"{max(d05):.1f}")
    claim(sec, "Protocol: held-out meshes", sum(g == "heldout" for g in groups.values()), 8, 0)
    claim(sec, "Protocol: analytic surfaces", sum(g == "control" for g in groups.values()), 3, 0)
    samples = {coverage[s]["samples"] for s in coverage}
    claim(sec, "Protocol: area-weighted surface samples per shape", max(samples), 16384, 0,
          f"{max(samples):,}" + ("" if len(samples) == 1 else " (varies)"))
    claim(sec, "Protocol: grid points per axis", fine_grid, 97, 0)
    claim(sec, "Protocol: radii", len(radii), 9, 0)
    claim(sec, "Protocol: smallest radius", min(radii), 0.025, 3)
    claim(sec, "Protocol: largest radius", max(radii), 0.4, 1)

    # Analytic surfaces, grid padding and landmarks, from the per-shape metadata written by sample_fields.py.
    metadata = {s: load_json(*EXTRA_COMPLETE, "metadata", f"{s}.json") for s in groups}
    sphere, torus, pair = metadata["sphere"], metadata["torus"], metadata["two_spheres"]
    claim(sec, "Protocol: unit sphere, radius", sphere["radius"], 1, 0, f"{sphere['radius']:g}")
    claim(sec, "Protocol: torus, major radius", torus["major_radius"], 1, 0, f"{torus['major_radius']:g}")
    claim(sec, "Protocol: torus, minor radius", torus["minor_radius"], 0.35, 2)
    claim(sec, "Protocol: two spheres, radius", pair["radius"], 0.55, 2)
    # true_feature_radius of the two spheres is half the gap between them (the radius at which their tubes merge).
    centre_distance = 2 * (pair["radius"] + pair["true_feature_radius"])
    claim(sec, "Protocol: two spheres, distance between centres", centre_distance, 1.3, 1)
    partial_protocol = load_json(*EXTRA_PARTIAL, "protocol.json")
    cases = load_json(*EXTRA_PARTIAL, "cases.json")
    claim(sec, "Protocol: encoder points, at most", complete_protocol["encoder_points"], 4096, 0,
          f"{complete_protocol['encoder_points']:,}")
    disk_centres = {c["disk_centres"] for c in cases}
    claim(sec, "Protocol: surfels per shape (60% hidden, realised)", max(disk_centres), 2048, 0,
          f"{max(disk_centres):,}" + ("" if len(disk_centres) == 1 else " (varies)"))
    claim(sec, "Protocol: area hidden, requested (%)", 100 * partial_protocol["mask"]["requested_fraction"], 60, 0)
    claim(sec, "Protocol: hidden patches", partial_protocol["mask"]["patches"], 32, 0)
    # sample_fields.py: grid width = longest normalised bounding-box side + 2 x padding. The analytic extents follow
    # from the stored dimensions (two spheres: centres at +-(radius + half gap)).
    extents = {"sphere": 2 * sphere["radius"], "torus": 2 * (torus["major_radius"] + torus["minor_radius"]),
               "two_spheres": 2 * (2 * pair["radius"] + pair["true_feature_radius"])}
    padding = {s: ((metadata[s]["grid_n"] - 1) * metadata[s]["h"] - extents[s] / metadata[s]["scale"]) / 2
               for s in extents}
    claim(sec, "Protocol: grid padding (RMS radii), analytic shapes", min(padding.values()), 0.6, 1,
          f"{min(padding.values()):.3f}-{max(padding.values()):.3f}")
    probes = sum(complete_protocol["additional_queries"][k] for k in ("uniform", "near_surface"))
    landmarks = {m["query_count"] - m["grid_n"] ** 3 - probes for m in metadata.values()}
    claim(sec, "Protocol: farthest-point landmarks per shape", max(landmarks), 6, 0,
          f"{max(landmarks)}" + ("" if len(landmarks) == 1 else " (varies)"))
    claim(sec, "Protocol: persistence filtrations capped at r (complete inputs)",
          load_json(*EXTRA_COMPLETE, "metrics_protocol.json")["secondary_filtration_cap"], 0.4, 1)
    claim(sec, "Protocol: persistence filtrations capped at r (partial inputs)",
          partial_protocol["evaluation"]["persistence_cap"], 0.4, 1)
    hidden_shares = [coverage[s]["hidden_share"] for s in coverage]
    encoder = [c["encoder_points"] for c in cases]
    sec.notes.append(f"The mask requests 60% of the area as 32 patches; the realised hidden share per shape is "
                     f"{min(hidden_shares):.1f}-{max(hidden_shares):.1f}% (partial_inputs/surface_queries/*.npz). "
                     "Shape counts: complete_inputs/fields_summary.json and persistence.csv; grid and radii: "
                     "complete_inputs/summary.json.")
    sec.notes.append("From results/distance_tubes_extra: analytic dimensions, grid spacing and query counts from "
                     "complete_inputs/metadata/<shape>.json; encoder-point cap, surfels and probe counts from "
                     "complete_inputs/protocol.json; mask and partial-input cap from partial_inputs/protocol.json; "
                     "complete-input cap from complete_inputs/metrics_protocol.json; realised disk centres from "
                     f"partial_inputs/cases.json. With 60% hidden the encoder reads {min(encoder):,}-{max(encoder):,} "
                     "visible points (cases.json). Landmarks per shape = query_count - grid_n^3 - (uniform + "
                     "near-surface probes), the count sample_fields.py records.")

    # ---------------- Table A.1, 'Distance tubes' row ----------------
    a1 = Table(
        key="Table A.1 (Distance tubes row)",
        title="Experiment map: evaluation set, observations and seeds of the distance-tube results",
        columns=["Claim", "Rebuilt", "Paper"],
    )
    claim(a1, "A.1: held-out meshes", sum(c["group"] == "heldout" for c in cases), 8, 0)
    claim(a1, "A.1: observations, area hidden (%)", 100 * partial_protocol["mask"]["requested_fraction"], 60, 0)
    claim(a1, "A.1: seeds (distinct seeds in the complete- and partial-input protocols)",
          len({complete_protocol["seed"], partial_protocol["seed"]}), 1, 0)
    a1.notes.append("Held-out meshes: partial_inputs/cases.json (group heldout); the analytic shapes are "
                    + ", ".join(ANALYTIC_LABELS[s] for s in analytic_ids).lower() + ". Area hidden and seed: "
                    "partial_inputs/protocol.json and complete_inputs/protocol.json in results/distance_tubes_extra.")

    return [j1, ct, j2, pt, sec, a1]
