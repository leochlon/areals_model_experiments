#!/usr/bin/env python3
"""Run supplied budget/seed manifests exactly; never invent a paper split.

Study JSON requires geometry_checkpoint, pretraining_cache, runs (each with
budget, seed, manifest), encoder_lrs, decoder_lrs and training_options.
The pretraining cache must hash to the source checkpoint's recorded data hash.
This runner evaluates the polynomial-patch arm only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from pressure_data import Cases
from train_mesh_geometry import sha256, json_write


def pretraining_provenance(checkpoint, cache):
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    digest = sha256(cache)
    if state.get("data_sha256") != digest:
        raise ValueError("Pretraining cache does not match the geometry checkpoint's recorded data hash")
    cache = Path(cache)
    if cache.suffix == ".json":
        document = json.loads(cache.read_text())
        meshes = [s["mesh"] for s in document["shards"]]
        for shard in document["shards"]:
            if sha256(cache.parent/shard["path"]) != shard["sha256"]:
                raise ValueError("Pretraining shard changed after geometry preparation")
    else:
        with np.load(cache, allow_pickle=False) as z:
            meshes = json.loads(str(z["metadata"]))["meshes"]
    training_hashes = sorted({m["sha256"] for m in meshes if m["split"] == "train"})
    if not training_hashes:
        raise ValueError("No geometry training meshes")
    corpus_hash = hashlib.sha256(json.dumps(training_hashes, separators=(",", ":")).encode()).hexdigest()
    return dict(checkpoint_sha256=sha256(checkpoint), checkpoint_step=state["step"],
                geometry_cache_sha256=digest, training_meshes=len(training_hashes),
                training_source_meshes_sha256=corpus_hash,
                geometry_masking="none: this patch encoder used complete-geometry pretraining")


def run(args):
    source = Path(args.study).resolve(); spec = json.loads(source.read_text()); base = source.parent
    if spec.get("version") != "patch-pressure-study-v1":
        raise ValueError("Expected patch-pressure-study-v1")
    checkpoint = (base/spec["geometry_checkpoint"]).resolve()
    provenance = pretraining_provenance(checkpoint, (base/spec["pretraining_cache"]).resolve())
    expected = spec.get("expected_training_source_meshes_sha256")
    if expected is not None and expected != provenance["training_source_meshes_sha256"]:
        raise ValueError("Pretraining corpus differs from the study's required source mesh set")
    records, seen = [], set()
    for entry in spec["runs"]:
        budget, seed = entry["budget"], entry["seed"]
        if not isinstance(budget, int) or budget <= 0 or not isinstance(seed, int) or seed < 0 or (budget, seed) in seen:
            raise ValueError("Runs need unique positive-budget/nonnegative-seed pairs")
        seen.add((budget, seed))
        manifest = (base/entry["manifest"]).resolve()
        data = Cases(manifest, weighting="area")
        if data.scope != "held_out_cars" or not data.by_split["test"]:
            raise ValueError("Study requires explicit train/validation/test cars")
        car_ids = sorted({data.entries[i]["car_id"] for i in data.by_split["train"]})
        if len(car_ids) != budget:
            raise ValueError(f"Budget {budget} contains {len(car_ids)} training cars")
        records.append(dict(budget=budget, seed=seed, manifest=str(manifest),
                            data_fingerprint=data.audit["fingerprint"], training_car_ids=car_ids,
                            validation_car_ids=sorted({data.entries[i]["car_id"] for i in data.by_split["val"]}),
                            test_car_ids=sorted({data.entries[i]["car_id"] for i in data.by_split["test"]})))
    if not records:
        raise ValueError("Provide the actual budget/seed manifests; no splits are generated")
    extra = spec.get("training_options", [])
    forbidden = {"--manifest", "--geometry-checkpoint", "--out", "--seeds", "--encoder-lrs", "--decoder-lrs", "--diagnostic-same-car", "--weighting", "--selection-metric"}
    if not all(isinstance(s, str) for s in extra) or any(s.split("=")[0] in forbidden for s in extra):
        raise ValueError("Training options cannot override study data, rates, scope or area/std metric")
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    audit = dict(study_sha256=sha256(source), model_family="polynomial_patch_encoder", pretraining=provenance,
                 corpus_matches_explicit_required_hash=expected is not None,
                 runs=records, original_areals_or_hessian_evaluated=False)
    audit_path = out/"study_audit.json"
    if audit_path.exists() and json.loads(audit_path.read_text()) != audit:
        raise ValueError("Study changed after an earlier audit")
    json_write(audit_path, audit)
    if args.audit_only:
        print(json.dumps(audit, indent=2)); return
    summaries = []
    for entry in records:
        dest = out/f"budget_{entry['budget']}"/f"seed_{entry['seed']}"
        command = [sys.executable, str(Path(__file__).with_name("tune_pressure.py")),
                   "--manifest", entry["manifest"], "--geometry-checkpoint", str(checkpoint),
                   "--out", str(dest), "--seeds", str(entry["seed"]),
                   "--encoder-lrs", *map(str, spec["encoder_lrs"]),
                   "--decoder-lrs", *map(str, spec["decoder_lrs"]),
                   "--weighting", "area", "--selection-metric", "nrmse_std", *extra]
        if args.resume: command.append("--resume")
        subprocess.run(command, check=True)
        summaries.append(dict(budget=entry["budget"], seed=entry["seed"],
                              result=json.loads((dest/"summary.json").read_text())))
    json_write(out/"study_results.json", dict(audit=audit, results=summaries))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True); p.add_argument("--out", required=True)
    p.add_argument("--audit-only", action="store_true"); p.add_argument("--resume", action="store_true")
    run(p.parse_args())
