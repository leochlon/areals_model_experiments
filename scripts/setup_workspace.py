#!/usr/bin/env python3
"""Recreate the directory layout the experiments ran in.

Every study in experiments/ ran in place in a fixed directory under /workspace (runs/, home/ or local/),
and its scripts refer to those absolute paths. This tool copies each study folder, the shared harness
versions and the model package to the locations listed in scripts/workspace_layout.json. Several
studies share one directory; their files are merged (no two of them write the same file).

Files are copied rather than linked, because some scripts locate their siblings through
Path(__file__).resolve().

    python scripts/setup_workspace.py                         # copy into /workspace
    python scripts/setup_workspace.py --root /scratch/ws      # stage elsewhere, then bind-mount at /workspace
    python scripts/setup_workspace.py --python-env /opt/venv  # also point every expected .venv at one environment
    python scripts/setup_workspace.py --dry-run

Datasets, checkpoints and per-run outputs are not part of this repository; each experiment README lists
the inputs it expects.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")


def copy_tree(src: Path, dst: Path, dry: bool, ignore=SKIP) -> None:
    print(f"  {src.relative_to(REPO)}  ->  {dst}")
    if not dry:
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)


def copy_file(src: Path, dst: Path, dry: bool) -> None:
    print(f"  {src.relative_to(REPO)}  ->  {dst}")
    if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def place_package(entry: dict, root: Path, dry: bool) -> None:
    src = REPO / entry["repo"]
    dst = root / entry["workspace"]
    layout = entry["layout"]
    if layout == "package":  # the importable package directory itself
        copy_tree(src, dst, dry, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
    elif layout == "project":  # the package's project layout: src/, configs/, tests/, pyproject.toml, README.md
        copy_tree(src, dst / "src" / src.name, dry, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
        copy_tree(REPO / "configs", dst / "configs", dry)
        copy_tree(REPO / "tests", dst / "tests", dry, ignore=shutil.ignore_patterns("__pycache__", "test_paper_tables.py"))
        copy_file(REPO / "pyproject.toml", dst / "pyproject.toml", dry)
        copy_file(REPO / "docs" / "areal_tokens.md", dst / "README.md", dry)
    elif layout == "package_tests":  # the package's own tests (not the paper-table test)
        copy_tree(src, dst, dry, ignore=shutil.ignore_patterns("__pycache__", "test_paper_tables.py"))
    else:
        raise ValueError(f"unknown layout {layout!r}")
    if entry.get("overlay"):  # a study's modified copy: only the files that differ from src/ are stored
        copy_tree(REPO / entry["overlay"], dst, dry)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="/workspace", help="workspace root (default /workspace)")
    parser.add_argument("--python-env", help="existing Python environment to link at every expected .venv path")
    parser.add_argument("--dry-run", action="store_true", help="print what would be copied")
    args = parser.parse_args(argv)

    root = Path(args.root)
    layout = json.loads((REPO / "scripts" / "workspace_layout.json").read_text())
    if root != Path("/workspace"):
        print(f"Note: scripts refer to /workspace; mount or link {root} there before running them.")

    print("Studies:")
    for entry in layout["studies"]:
        copy_tree(REPO / entry["repo"], root / entry["workspace"], args.dry_run)
    print("Shared files (field-training harness versions and shared scripts):")
    for entry in layout["files"]:
        copy_file(REPO / entry["repo"], root / entry["workspace"], args.dry_run)
    print("Model package copies:")
    for entry in layout["packages"]:
        place_package(entry, root, args.dry_run)
    if args.python_env:
        env = Path(args.python_env).resolve()
        print("Python environments:")
        for rel in layout["python_environments"]:
            link = root / rel
            print(f"  {link} -> {env}")
            if not args.dry_run and not link.exists():
                link.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(env, link)
    else:
        print("Python environments the experiments expect (pass --python-env to link one environment):")
        for rel in layout["python_environments"]:
            print(f"  {root / rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
