"""Shared helpers for rebuilding the paper's tables from the stored result files."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def load_json(*parts: str):
    """Load a JSON file under results/."""
    with open(RESULTS.joinpath(*parts), encoding="utf-8") as f:
        return json.load(f)


def load_csv(*parts: str) -> list[dict]:
    """Load a CSV file under results/ as a list of row dicts."""
    with open(RESULTS.joinpath(*parts), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_printed(value: float, decimals: int) -> float:
    """The value as a table prints it (round half up). Used where the text quotes a number that
    a table already prints at higher precision, e.g. "52%" for a table entry of 51.5."""
    from decimal import ROUND_HALF_UP, Decimal

    return float(Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP))


def fmt(value: float | None, decimals: int = 3) -> str:
    """Format a number the way the paper prints it."""
    if value is None:
        return "—"
    return f"{value:.{decimals}f}"


@dataclass
class Check:
    """One printed number in the paper, compared with the value rebuilt from results/."""

    label: str
    value: float
    paper: float
    decimals: int

    @property
    def passed(self) -> bool:
        # The paper prints `decimals` places; the rebuilt value must round to the printed one.
        return abs(self.value - self.paper) <= 0.5 * 10 ** (-self.decimals) + 1e-9


@dataclass
class Table:
    """A rebuilt table: what is printed, plus the checks against the paper."""

    key: str  # e.g. "Table 1"
    title: str
    columns: list[str]
    rows: list[list[str]] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def check(self, label: str, value: float, paper: float, decimals: int) -> float:
        """Record a check and return the value, so it can be used inline when building a row."""
        self.checks.append(Check(label, float(value), float(paper), decimals))
        return value

    def markdown(self) -> str:
        lines = [f"### {self.key}. {self.title}", ""]
        lines.append("| " + " | ".join(self.columns) + " |")
        lines.append("|" + "---|" * len(self.columns))
        for row in self.rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        failed = [c for c in self.checks if not c.passed]
        lines.append("")
        lines.append(f"{len(self.checks) - len(failed)}/{len(self.checks)} printed numbers reproduced.")
        for c in failed:
            lines.append(f"- MISMATCH {c.label}: rebuilt {c.value:.6g}, paper {c.paper} ({c.decimals} dp)")
        for note in self.notes:
            lines.append(f"- {note}")
        return "\n".join(lines)
