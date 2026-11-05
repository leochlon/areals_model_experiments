"""Every number printed in the paper's tables is rebuilt from results/."""
import importlib
import pkgutil

import pytest

import paper_tables

MODULES = sorted(m.name for m in pkgutil.iter_modules(paper_tables.__path__) if not m.name.startswith("_"))


@pytest.mark.parametrize("name", MODULES)
def test_tables_reproduce_paper(name):
    module = importlib.import_module(f"paper_tables.{name}")
    failures = [
        f"{t.key} {c.label}: rebuilt {c.value}, paper {c.paper}"
        for t in module.tables()
        for c in t.checks
        if not c.passed
    ]
    assert not failures, "\n".join(failures)
