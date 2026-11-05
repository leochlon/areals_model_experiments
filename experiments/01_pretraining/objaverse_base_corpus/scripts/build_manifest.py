"""Build deterministic splits from every downloaded Objaverse metadata row."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.dataset / "metadata.csv"
    records, seen, excluded = [], set(), Counter()
    with source.open(newline="") as stream:
        for row in csv.DictReader(stream):
            identity, local = row["sha256"].lower(), row.get("local_path", "")
            if not local:
                excluded["not_downloaded"] += 1
                continue
            if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
                raise ValueError("Invalid asset identifier in metadata")
            relative = PurePosixPath(local)
            if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".glb":
                raise ValueError(f"Unexpected model path: {local}")
            if identity in seen:
                excluded["duplicate_asset"] += 1
                continue
            seen.add(identity)
            bucket = int.from_bytes(hashlib.sha256(("split-17:" + identity).encode()).digest()[:8], "big") % 10000
            split = "train" if bucket < 9000 else "val" if bucket < 9500 else "test"
            records.append(dict(id=identity, family=identity, split=split,
                                path=str(args.dataset / local), source_url=row["file_identifier"]))
    records.sort(key=lambda item: item["id"])
    counts = Counter(record["split"] for record in records)
    if set(counts) != {"train", "val", "test"}:
        raise ValueError("All three splits must be populated")
    document = dict(schema="geometry_latents_v1", units="source_units_unknown",
                    records=records, source_metadata=str(source),
                    split_policy="hash of the asset identifier with seed 17; 90/5/5 train/val/test",
                    deduplication="exact asset identifier only")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(document, indent=2) + "\n"
    if args.out.exists() and args.out.read_text() != serialized:
        raise ValueError("Existing run manifest differs; refusing to change this run's data")
    args.out.write_text(serialized)
    summary = dict(assets=len(records), splits=dict(counts), excluded=dict(excluded), manifest=str(args.out))
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
