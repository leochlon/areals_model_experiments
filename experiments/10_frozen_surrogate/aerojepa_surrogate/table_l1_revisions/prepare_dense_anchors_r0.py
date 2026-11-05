"""Extend the original 2,048 anchors into the common dense surface cloud."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from extend_anchors import add_anchor_features, extend_fps

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=10000)
    parser.add_argument("--source", type=Path, default=ROOT / "dense_context_16384")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.tokens <= 2048:
        parser.error("--tokens must exceed the 2,048 preserved anchors")
    out = ROOT / f"dense_anchors_{args.tokens}"
    out.mkdir(exist_ok=True)
    if (out / "ready.json").exists():
        cached = json.loads((out / "ready.json").read_text())
        if cached["source"] != str(args.source.resolve()) or cached["geometry_tokens"] != args.tokens:
            raise ValueError("Existing anchor cache has a different configuration")
        print("DENSE_ANCHORS_READY cached", flush=True)
        return
    ready = json.loads((args.source / "ready.json").read_text())
    started = time.time()
    # Compile once before starting the workers; the selection loop releases the GIL.
    extend_fps(np.zeros((1, 3), np.float64), np.array([-np.inf]), 1, 1)

    def prepare(row):
        name = row["id"]
        output, info = out / (name + ".npz"), out / (name + ".json")
        if output.exists() and info.exists():
            return json.loads(info.read_text())
        with np.load(ROOT / "data" / (name + ".npz"), allow_pickle=False) as p:
            original = {key: p[key] for key in ("context_pos", "context_feat")}
        add_anchor_features(original, 2048)
        rank = original["context_feat"][:, 6]
        inherited = np.argsort(rank, kind="stable")[-2048:]
        order = np.concatenate((inherited, np.flatnonzero(rank == 0)))
        old_points = original["context_pos"][order]
        old_normals = original["context_feat"][order, :3]
        old_weights = original["context_feat"][order, 3].astype(np.float64)
        with np.load(args.source / (name + ".npz"), allow_pickle=False) as p:
            dense_points, dense_normals = p["points"], p["normals"]
            dense_weights = p["point_weights"].astype(np.float64)
        # Both clouds estimate the same area measure. Combine their normalized
        # measures in proportion to sample count, then merge duplicate positions.
        old_weights *= len(old_points) / old_weights.sum()
        dense_weights *= len(dense_points) / dense_weights.sum()
        points = np.concatenate((old_points, dense_points))
        normals = np.concatenate((old_normals, dense_normals))
        mass = np.concatenate((old_weights, dense_weights))
        _, first, inverse = np.unique(points, axis=0, return_index=True, return_inverse=True)
        stable = np.argsort(first)
        pooled_mass = np.bincount(inverse, weights=mass)[stable]
        points, normals = points[first[stable]], normals[first[stable]]
        if not np.array_equal(points[:2048], old_points[:2048]):
            raise ValueError(f"{name}: original anchors are not distinct")
        if len(points) < args.tokens:
            raise ValueError(f"{name}: insufficient distinct anchor candidates")
        distance = cKDTree(points[:2048]).query(points, workers=1)[0] ** 2
        distance[:2048] = -np.inf
        chosen = extend_fps(points.astype(np.float64), distance, 2048, args.tokens)
        anchors, anchor_normals = points[chosen], normals[chosen]
        owners = cKDTree(anchors).query(points, workers=1)[1]
        weights = np.bincount(owners, weights=pooled_mass, minlength=args.tokens)
        if (weights <= 0).any() or len(np.unique(chosen)) != args.tokens:
            raise ValueError(f"{name}: empty or repeated anchor")
        weights = (weights / weights.sum()).astype(np.float32)
        np.savez(output, anchors=anchors, anchor_normals=anchor_normals, anchor_weights=weights)
        result = dict(id=name, split=row["split"], code=row["code"],
                      geometry_tokens=args.tokens, preserved_anchors=2048,
                      distinct_positions=len(np.unique(anchors, axis=0)),
                      positive_area_weights=int((weights > 0).sum()))
        info.write_text(json.dumps(result) + "\n")
        return result

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(prepare, row) for row in ready["records"]]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
            if len(records) == 1 or len(records) % 32 == 0 or len(records) == len(futures):
                progress = dict(complete=len(records), total=len(futures), seconds=time.time()-started)
                (out / "progress.json").write_text(json.dumps(progress) + "\n")
                print("DENSE_ANCHORS " + json.dumps(progress), flush=True)
    records.sort(key=lambda row: row["code"])
    report = dict(records=records, geometry_tokens=args.tokens, preserved_anchors=2048,
                  source=str(args.source.resolve()),
                  sampling="FPS extension of the original 2,048 anchors over the deduplicated union of the original and 16k dense geometry",
                  area_weights="Voronoi mass over the sample-count-weighted mixture of the two surface area measures",
                  seconds=time.time()-started)
    (out / "ready.json").write_text(json.dumps(report, indent=2) + "\n")
    print("DENSE_ANCHORS_READY", flush=True)


if __name__ == "__main__":
    main()
