#!/usr/bin/env python3
"""Nearest-neighbour pressure interpolation using ONLY training CFD labels."""
from __future__ import annotations
import argparse
import hashlib
import numpy as np
from scipy.spatial import cKDTree
from pressure_data import Cases, case_metrics, summarize_metrics
from train_mesh_geometry import json_write


def nearest_predictions(train_xyz, train_cp, train_weights, query_xyz):
    unique, inverse = np.unique(train_xyz, axis=0, return_inverse=True)
    weights = np.bincount(inverse, weights=train_weights)
    values = np.bincount(inverse, weights=train_cp*train_weights) / weights
    distances, nearest = cKDTree(unique).query(query_xyz)
    return values[nearest], distances


def run(args):
    data = Cases(args.manifest, weighting=args.weighting)
    if data.scope not in ("held_out_queries_same_car", "held_out_regions_same_car"):
        raise ValueError("This baseline tests within-car interpolation; it cannot use a held-out car's CFD labels as training data")
    stats = data.normalization()
    records = []
    for i in data.by_split[args.split]:
        e, f = data.entries[i], data.get(i)
        parts = [data.get(j) for j in data.by_split["train"] if data.entries[j]["car_id"] == e["car_id"]
                 and data.entries[j]["conditions"] == e["conditions"]]
        if not parts:
            raise ValueError(f"No same-car, same-condition training samples for {e['id']}")
        arrays = {k: np.concatenate([p[k] for p in parts]) for k in ("xyz", "cp", "weights")}
        pred, distance = nearest_predictions(arrays["xyz"], arrays["cp"], arrays["weights"], f["xyz"])
        scale = data.geometry(e)["scale"]
        records.append(dict(id=e["id"], car_id=e["car_id"],
                            weight_measure=f["metadata"].get("weight_measure", "unspecified"),
                            query_indices_sha256=hashlib.sha256(np.asarray(f["indices"], dtype="<i8").tobytes()).hexdigest(),
                            min_neighbor_distance_diagonal=float(distance.min()/scale),
                            median_neighbor_distance_diagonal=float(np.median(distance)/scale),
                            **case_metrics(pred, f["cp"], f["weights"], stats["cp_std"])))
    if not records:
        raise ValueError("No evaluation cases")
    report = summarize_metrics(records, data.scope, data.audit["fingerprint"])
    report.update(split=args.split, method="nearest training coordinate; duplicate coordinates use training area-weighted means",
                  uses_only_training_pressure=True)
    json_write(args.out, report)
    print(f"Nearest-neighbour area/std nRMSE: {report['case_mean_nrmse_reference_std']}")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True); p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--weighting", choices=["area", "provided"], default="area"); p.add_argument("--out", required=True)
    run(p.parse_args())
