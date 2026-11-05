"""Publish the expanded training pool: the combined manifest and its audit, in manifest order.

Run after finish_preparation.py has written results.json. Writes a new version under
/workspace/runs/objaverse_training_pool/versions/, points `current` at it, and records the counts in
pool.json and pool_ready.json.
"""

import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from areal_tokens.data import read_manifest
from areal_tokens.io import write_json
from areal_tokens.validation import load_audit

ROOT = Path(__file__).resolve().parent
POOL = Path("/workspace/runs/objaverse_training_pool")
VERSION = POOL / "versions" / "20260927T140652Z"


def read(path):
    return json.loads(Path(path).read_text())


def main():
    completed = read(ROOT / "results.json")
    assert completed["phase"] == "complete" and completed["additional_training_geometries"] == 30000
    entries, rejected = [], []
    for worker in range(24):
        rows = (ROOT / "records" / f"worker_{worker:02d}.jsonl").read_text().splitlines()
        entries.extend(json.loads(line) for line in rows if line.strip())
        failures = ROOT / "records" / f"failures_{worker:02d}.jsonl"
        if failures.exists():
            rejected.extend(json.loads(line) for line in failures.read_text().splitlines() if line.strip())
    qualification = read(ROOT / "qualification.json")
    backends = []  # GPU closest-triangle queries checked against the CPU (libigl) reference
    for entry in entries:
        if entry["record"]["cached_original"]:
            continue
        details = entry["preparation"]
        if details is None:  # the mesh prepared by qualify_gpu.py
            assert entry["record"] == qualification["record"]
            details = qualification["gpu_queries"]
        assert details["cpu_full_fallback"] or details["max_reference_error_over_extent"] <= 1e-5
        backends.append(details)

    VERSION.mkdir(parents=True, exist_ok=False)
    manifest = VERSION / "manifest.json"
    shutil.copyfile(ROOT / "combined_manifest.json", manifest)
    records, metadata = read_manifest(manifest)  # ID, family and reused-path checks
    splits = dict(Counter(r["split"] for r in records))
    assert splits == {"train": 61822, "val": 1844, "test": 1825}
    inventory = defaultdict(set)  # every referenced file exists: one directory listing per storage root
    for record in records:
        path = Path(record["path"])
        inventory[path.parent].add(path.name)
    for directory, required in inventory.items():
        present = {entry.name for entry in os.scandir(directory)}
        assert not (required - present), (str(directory), sorted(required - present)[:10])

    audit = load_audit(ROOT / "combined_audit.json", ROOT / "combined_manifest.json")
    shape_rows = {row["id"]: row for row in audit["shapes"]}
    audit.update(metadata)
    audit["shapes"] = [shape_rows[r["id"]] for r in records]
    write_json(VERSION / "audit.json", audit)

    generated = sum(not e["record"]["cached_original"] for e in entries)
    report = {
        "phase": "ready", "utc": datetime.now(timezone.utc).isoformat(),
        "additional_training_geometries": len(entries), "combined_splits": splits,
        "generated": generated, "reused": len(entries) - generated, "rejected_records": rejected,
        "distance_backend": {
            "new_meshes_checked_against_cpu_reference": len(backends),
            "full_cpu_fallbacks": sum(r["cpu_full_fallback"] for r in backends),
            "full_query_reference_checks": sum(r["cpu_reference_queries"] == 32768 for r in backends),
            "max_accepted_gpu_error_over_mesh_extent": max(r["max_reference_error_over_extent"]
                                                           for r in backends if not r["cpu_full_fallback"]),
            "tolerance": 1e-5,
        },
        "points_after_native_coalescing": [min(e["audit"]["points"] for e in entries),
                                           max(e["audit"]["points"] for e in entries)],
        "queries_per_added_mesh": 32768,
        "manifest": str(manifest), "audit": str(VERSION / "audit.json"),
    }
    write_json(VERSION / "verification.json", report)
    temporary = POOL / ".current.tmp"
    temporary.symlink_to(Path("versions") / VERSION.name, target_is_directory=True)
    temporary.replace(POOL / "current")
    ready = dict(report, current_manifest=str(POOL / "current/manifest.json"),
                 current_audit=str(POOL / "current/audit.json"))
    write_json(POOL / "pool.json", ready)
    write_json(ROOT / "pool_ready.json", ready)
    print(json.dumps(ready, indent=2), flush=True)


if __name__ == "__main__":
    main()
