"""Prepare denser geometry input for the same DrivAerML cars, without reading field labels."""
import concurrent.futures
import hashlib
import io
import json
import os
from pathlib import Path
import time

import botocore.config
import botocore.session
import h5py
import numpy as np
from areal_tokens.geometry import canonical_cloud

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "dense_context_16384"
COUNT = 16384


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / "ready.json").exists():
        print("Dense geometry cache already complete", flush=True)
        return
    ready = json.loads((ROOT / "data_ready.json").read_text())
    prefix = json.loads((ROOT.parent / "drivaerml_transfer_pilot_20260925T121525Z/config.json").read_text())["source_prefix"]
    # Any S3-compatible copy of the DrivAerML release: set S3_ENDPOINT_URL and the usual AWS credentials.
    client = botocore.session.get_session().create_client(
        "s3", region_name="us-east-1", endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
        config=botocore.config.Config(signature_version="s3v4", max_pool_connections=24,
                                      s3={"addressing_style": "virtual"}))
    started = time.time()

    def prepare(item):
        code, row = item
        name = row["id"]
        output = OUT / (name + ".npz")
        info = OUT / (name + ".json")
        if output.exists() and info.exists():
            return json.loads(info.read_text())
        response = client.get_object(Bucket="training-data", Key=prefix + name + "/fields.h5")
        body = response["Body"]
        try:
            raw = body.read()
        finally:
            body.close()
        with h5py.File(io.BytesIO(raw), "r") as packed:
            points = packed["mesh.verts"][:].astype(np.float64)
            normals = packed["mesh.verts_normals"][:].astype(np.float64)
            weights = packed["mass"][:].astype(np.float64).reshape(-1)
        del raw
        lengths = np.linalg.norm(normals, axis=1)
        good = (weights > 0) & (lengths > 1e-8)
        good &= np.isfinite(points).all(1) & np.isfinite(normals).all(1) & np.isfinite(weights)
        normals[good] /= lengths[good, None]
        cloud = canonical_cloud(points[good], normals[good], weights[good])
        # Uniform sampling without replacement gives equal inclusion probability;
        # original area weights therefore retain the surface integration measure.
        seed = int.from_bytes(hashlib.sha256((name + ":dense16384:0").encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        if len(cloud.points) < COUNT:
            raise ValueError(f"{name} has fewer than {COUNT} distinct usable surface samples")
        chosen = rng.choice(len(cloud.points), COUNT, replace=False)
        with np.load(ROOT / "data" / (name + ".npz"), allow_pickle=False) as previous:
            center, scale = previous["center"], previous["scale"]
        xyz = ((cloud.points[chosen] - center) / scale).astype(np.float32)
        nrm = cloud.normals[chosen].astype(np.float32)
        mass = cloud.weights[chosen]
        mass = (mass / mass.sum()).astype(np.float32)
        np.savez(output, points=xyz, normals=nrm, point_weights=mass, canonical_indices=chosen)
        result = dict(id=name, split=row["split"], code=code, points=COUNT,
                      source_vertices=len(points),
                      distinct_positions=len(np.unique(xyz, axis=0)),
                      source_bytes=response.get("ContentLength"))
        info.write_text(json.dumps(result)+"\n")
        return result

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(prepare, item) for item in enumerate(ready["records"])]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
            if len(records) == 1 or len(records) % 32 == 0 or len(records) == len(futures):
                progress = dict(complete=len(records), total=len(futures), seconds=time.time()-started)
                (OUT / "progress.json").write_text(json.dumps(progress)+"\n")
                print("DENSE_GEOMETRY " + json.dumps(progress), flush=True)
    records.sort(key=lambda r: r["code"])
    report = dict(records=records, geometry_points=COUNT, geometry_tokens=1024,
                  sampling="Uniform without replacement from the native canonical surface cloud; original area weights",
                  seconds=time.time()-started)
    (OUT / "ready.json").write_text(json.dumps(report, indent=2)+"\n")
    print("DENSE_GEOMETRY_READY", flush=True)


if __name__ == "__main__":
    main()
