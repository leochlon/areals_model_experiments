"""Surfel and symmetry readouts (Section 3.2, Appendix I, Table I.1, Figure I.1, Table A.1 row "Symmetry").

Table I.1 compares the package presets configs/surfels.json and configs/symmetry.json, trained on
170 meshes and evaluated on 41 held-out meshes with one seed. The outputs of that evaluation (mesh
manifest, evaluation reports, per-mesh errors and gate weights) are not part of this supplement, so
the numbers below are listed for reference and are not rebuilt or checked.
Code and instructions: experiments/14_symmetry_readouts/.
"""
from __future__ import annotations

from paper_tables._core import Table

NOTE = ("Listed for reference only: the 41-mesh evaluation outputs are not included in results/, so these "
        "numbers are not rebuilt. See experiments/14_symmetry_readouts/README.md for how to rerun the evaluation.")
NOT_REBUILT = "not included"


def table_i1() -> Table:
    t = Table(
        key="Table I.1",
        title="Symmetry maps on 41 held-out meshes, one seed: normal error in degrees (lower is better); "
        "All is pooled over points, Hidden is averaged over meshes",
        columns=["Prior input", "Full", "60% hidden: All", "60% hidden: Hidden", "60% hidden: Repeated parts"],
    )
    paper = {
        "Surfels, 8 per token": (5.7, 12.4, 17.6, 10.9),
        "+ symmetry maps": (5.7, 11.4, 15.7, 6.9),
    }
    for name, values in paper.items():
        t.rows.append([name] + [f"{v:.1f}" for v in values])
    t.notes.append("Values as printed in the paper. " + NOTE)
    return t


def table_text() -> Table:
    t = Table(
        key="Section 3.2 and Appendix I",
        title="Numbers quoted in the text (normal error in degrees; shares in percent)",
        columns=["Claim", "Rebuilt", "Paper"],
    )
    for label, paper in [
        ("Repeated-part normal error, surfels", "10.9"),
        ("Repeated-part normal error, + symmetry maps", "6.9"),
        ("Hidden normal error (mean over meshes), surfels", "17.6"),
        ("Hidden normal error (mean over meshes), + symmetry maps", "15.7"),
        ("Held-out meshes improved", "32 of 41"),
        ("Gate weight on symmetry maps at repeated-part points (%)", "61"),
        ("Hidden points mapped onto visible input, symmetries from complete shapes (%)", "63"),
    ]:
        t.rows.append([label, NOT_REBUILT, paper])
    t.notes.append(NOTE)
    return t


def table_a1() -> Table:
    t = Table(
        key="Table A.1 (row for this study)",
        title="Experiment-map entry \"Symmetry\" and the Table I.1 caption",
        columns=["Quantity", "Rebuilt", "Paper"],
    )
    for quantity, paper in [
        ("Training meshes", "170"),
        ("Held-out meshes", "41"),
        ("Seeds", "1"),
        ("Hidden fraction of the surface (%)", "60"),
        ("Surfels per token", "8"),
    ]:
        t.rows.append([quantity, NOT_REBUILT, paper])
    t.notes.append(NOTE)
    return t


def tables() -> list[Table]:
    return [table_i1(), table_text(), table_a1()]
