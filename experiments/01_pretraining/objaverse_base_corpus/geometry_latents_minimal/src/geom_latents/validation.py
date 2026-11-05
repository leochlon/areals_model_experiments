"""Fail early on invalid geometry, missing normals, or split leakage."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np

from .data import SCHEMA, load_cloud_file, read_manifest
from .geometry import frame
from .io import write_json
from .parallel import map_jobs


def inspect_record(record: dict, require_targets: bool = True) -> dict:
    """Per-file statistics and a geometry fingerprint; a failure is returned, not raised, so a
    parallel audit can report every bad file at once."""
    try:
        cloud, targets = load_cloud_file(record["path"])
        if require_targets and not targets:
            raise ValueError("supplied query targets are required")
        center, scale, area = frame(cloud)
        # Detect exact copied geometry even when IDs, paths or query labels differ.
        geometry_hash = hashlib.sha256()
        for array in (cloud.points, cloud.normals, cloud.weights / cloud.weights.sum()):
            geometry_hash.update(np.ascontiguousarray(array).tobytes())
        return dict(
            id=record["id"],
            split=record["split"],
            points=len(cloud.points),
            queries=len(targets.get("queries", [])),
            area=area,
            scale=scale,
            center=center.tolist(),
            geometry_key=geometry_hash.hexdigest(),
        )
    except Exception as error:  # noqa: BLE001 - reported to the caller
        return dict(id=record["id"], error=f"{type(error).__name__}: {error}")


def validate_manifest(path, *, require_targets=True, workers=1, bad_records="raise") -> dict:
    """Audit every record. With bad_records="skip" unreadable files are listed
    under "failures" and left out instead of raising."""
    if bad_records not in ("raise", "skip"):
        raise ValueError("bad_records must be raise or skip")
    records, metadata = read_manifest(path)
    results = map_jobs(partial(inspect_record, require_targets=require_targets), records, workers)
    identities = {}
    rows, failures = [], []
    for record, result in zip(records, results, strict=True):
        if "error" in result:
            if bad_records == "raise":
                raise ValueError(f"{record['id']}: {result['error']}")
            failures.append(dict(id=record["id"], path=record["path"], error=result["error"]))
            continue
        previous = identities.get(result["geometry_key"])
        if previous and previous["split"] != record["split"]:
            raise ValueError(f"Exact geometry copy crosses splits: {previous['id']} / {record['id']}")
        identities[result["geometry_key"]] = record
        rows.append({key: result[key] for key in ("id", "split", "points", "queries", "area", "scale", "center")})
    if not rows:
        raise ValueError("No readable records in the manifest")
    return dict(
        **metadata,
        splits=dict(Counter(row["split"] for row in rows)),
        shapes=rows,
        failures=failures,
        leakage_check="family separation, reused paths, exact canonical geometry copies; not near-duplicates",
    )


def load_audit(path, manifest, *, require_targets=True) -> dict:
    """Reuse a saved validate report for this exact manifest instead of rereading every file."""
    report = json.loads(Path(path).read_text())
    if report.get("manifest") != str(Path(manifest).resolve()):
        raise ValueError("The audit was made from a different manifest; run validate again")
    if report.get("failures"):
        raise ValueError("The audit lists failed records; train on its filtered manifest instead")
    if require_targets and any(row["queries"] == 0 for row in report["shapes"]):
        raise ValueError("The audit contains records without supplied targets")
    return report


def write_filtered_manifest(source, report: dict, output) -> Path:
    """The source manifest minus the records the audit could not read, with paths
    rewritten relative to the new file so it can live anywhere."""
    output = Path(output).resolve()
    document = json.loads(Path(source).read_text())
    failed = {failure["id"] for failure in report.get("failures", [])}
    parent = Path(source).resolve().parent
    kept = []
    for record in document["records"]:
        if record["id"] in failed:
            continue
        absolute = (parent / record["path"]).resolve()
        kept.append(dict(record, path=os.path.relpath(absolute, output.parent)))
    if not kept:
        raise ValueError("Every record failed; nothing to write")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, dict(document, schema=SCHEMA, records=kept, filtered_out=sorted(failed)))
    return output
