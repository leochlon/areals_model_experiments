"""Wait for the 24 preparation workers, then write the combined manifest and its audit."""

import concurrent.futures
import json
import time
from collections import Counter
from pathlib import Path

from areal_tokens.data import read_manifest
from areal_tokens.io import write_json
from areal_tokens.validation import inspect_record

ROOT = Path(__file__).resolve().parent


def main():
    config = json.loads((ROOT / "config.json").read_text())
    while True:
        progress = []
        for worker in range(24):
            path = ROOT / f"progress_{worker:02d}.json"
            progress.append(json.loads(path.read_text()) if path.exists() else {})
        write_json(ROOT / "progress.json", dict(
            phase="preparing", completed=sum(r.get("completed", 0) for r in progress),
            target=30000, generated=sum(r.get("generated", 0) for r in progress),
            reused=sum(r.get("reused", 0) for r in progress),
            failures=sum(r.get("failures", 0) for r in progress), workers=24, gpus=3))
        if any(r.get("phase") == "failed" for r in progress):
            raise RuntimeError("A preparation worker failed")
        if all(r.get("phase") == "complete" for r in progress):
            break
        time.sleep(10)
    entries = []
    for worker in range(24):
        entries.extend(json.loads(line) for line in (ROOT / "records" / f"worker_{worker:02d}.jsonl").read_text().splitlines() if line.strip())
    assert len(entries) == 30000
    new_records = sorted([entry["record"] for entry in entries], key=lambda r: r["id"])
    prior, metadata = read_manifest(config["existing_manifest"])
    assert len({r["id"] for r in new_records}) == 30000
    assert all(r["split"] == "train" for r in new_records)
    assert not ({r["id"] for r in prior} & {r["id"] for r in new_records})
    # Existing data are reused. Check the held-out geometry fingerprints once
    # against the new records, using the package's original leakage criterion.
    held_out = [r for r in prior if r["split"] != "train"]
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        checks = list(pool.map(inspect_record, held_out, chunksize=8))
    assert not any("error" in row for row in checks)
    assert not ({r["geometry_key"] for r in checks} & {e["audit"]["geometry_key"] for e in entries})
    additions = dict(schema="geometry_latents_v1", units=metadata["units"], records=new_records,
                     target_source="point-to-triangle unsigned distance")
    combined = dict(additions, records=prior + new_records)
    write_json(ROOT / "additional_training.json", additions)
    write_json(ROOT / "combined_manifest.json", combined)
    old_audit = json.loads(Path(config["existing_audit"]).read_text())
    shape_fields = ("id", "split", "points", "queries", "area", "scale", "center")
    audit = dict(units=metadata["units"], manifest=str(ROOT / "combined_manifest.json"),
                 splits=dict(Counter(r["split"] for r in combined["records"])),
                 shapes=old_audit["shapes"] + [{k:e["audit"][k] for k in shape_fields} for e in entries],
                 failures=[], leakage_check="Original family/path separation and exact held-out geometry fingerprints")
    write_json(ROOT / "combined_audit.json", audit)
    result = dict(phase="complete", additional_training_geometries=30000,
                  generated=sum(not r["cached_original"] for r in new_records),
                  reused=sum(r["cached_original"] for r in new_records),
                  combined_splits=audit["splits"], workers=24, gpus=3,
                  additional_manifest=str(ROOT / "additional_training.json"),
                  combined_manifest=str(ROOT / "combined_manifest.json"))
    write_json(ROOT / "progress.json", result)
    write_json(ROOT / "results.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
