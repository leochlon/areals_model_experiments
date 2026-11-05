"""Rebuild every table in the paper from results/ and check each printed number.

    python -m paper_tables                 print all tables as Markdown
    python -m paper_tables --check         also exit non-zero if any printed number is not reproduced
    python -m paper_tables --only gabi_baseline
"""
from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys

import paper_tables


def modules(only: list[str] | None):
    names = sorted(m.name for m in pkgutil.iter_modules(paper_tables.__path__) if not m.name.startswith("_"))
    for name in names:
        if only and name not in only:
            continue
        yield name, importlib.import_module(f"paper_tables.{name}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="exit 1 if any printed number is not reproduced")
    parser.add_argument("--only", nargs="*", help="experiment module names (see paper_tables/)")
    args = parser.parse_args(argv)

    total = failed = 0
    for name, module in modules(args.only):
        print(f"## {name}\n")
        for table in module.tables():
            print(table.markdown())
            print()
            total += len(table.checks)
            failed += sum(not c.passed for c in table.checks)
    print(f"Summary: {total - failed}/{total} printed numbers reproduced.")
    return 1 if (args.check and failed) else 0


if __name__ == "__main__":
    sys.exit(main())
