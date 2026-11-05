"""Reuse the 420/16 surface benchmark and fetch only its operating-condition scalars."""
import concurrent.futures
import json
import os
from pathlib import Path
import time

import botocore.config
import botocore.session
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "areal_objaverse_060_61822_15ep_20260928T121633Z/field_utonia_compare_420"
DATA = Path("/workspace/runs/areal_drivaerml_420_epoch14_20260927T162928Z")
PILOT = ROOT.parent / "drivaerml_transfer_pilot_20260925T121525Z"


def main():
    if (ROOT / "data_ready.json").exists():
        print("Prepared DrivAerML data already complete", flush=True)
        return
    started = time.monotonic()
    source = json.loads((OLD / "cache_ready.json").read_text())
    records = json.loads((DATA / "manifest.json").read_text())["records"]
    assert len(records) == 436 and sum(x["split"] == "train" for x in records) == 420
    assert [x["id"] for x in records if x["split"] == "val"] == json.loads(
        (PILOT / "manifest.json").read_text())["validation"]
    # Any S3-compatible copy of the DrivAerML release: set S3_ENDPOINT_URL and the usual AWS credentials.
    client = botocore.session.get_session().create_client(
        "s3", region_name="us-east-1", endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
        config=botocore.config.Config(signature_version="s3v4", max_pool_connections=24,
                                      s3={"addressing_style": "virtual"}))
    prefix = json.loads((PILOT / "config.json").read_text())["source_prefix"]
    (ROOT / "scalars").mkdir(exist_ok=True)
    (ROOT / "data").mkdir(exist_ok=True)

    def prepare(record):
        name = record["id"]
        path = OLD / "cache" / (name + ".npz")
        with np.load(path, allow_pickle=False) as z:
            p = {key: z[key] for key in ["points", "normals", "point_weights", "anchors", "anchor_normals",
                                         "anchor_weights", "anchor_index", "values", "center", "scale", "area"]}
        anchors = p["anchor_index"].astype(np.int64)
        assert len(np.unique(anchors)) == 256
        np.testing.assert_array_equal(p["points"][anchors], p["anchors"])
        order = np.concatenate([anchors, np.setdiff1d(np.arange(len(p["points"])), anchors)])
        features = np.zeros((len(order), 6), np.float32)
        features[:, :3] = p["normals"][order]
        features[:, 3] = p["point_weights"][order]
        features[:256, 4] = p["anchor_weights"]
        features[:, 5] = p["area"].reshape(-1)[0]
        scalar = ROOT / "scalars" / (name + ".h5")
        if not scalar.exists():
            try:
                response = client.get_object(Bucket="training-data", Key=prefix + name + "/scalars.h5")
                body = response["Body"]
                try:
                    scalar.write_bytes(body.read())
                finally:
                    body.close()
            except Exception:
                raise RuntimeError("Could not read operating conditions for " + name) from None
        with h5py.File(scalar, "r") as z:
            conditions = np.array([float(z["upstream_velocity"][()]) / 40,
                                   float(z["reference_area"][()]) / 2.17], np.float32)
        arrays = dict(context_pos=p["points"][order], context_feat=features, pressure=p["values"][order],
                      center=p["center"], scale=p["scale"], gen_params_raw=conditions)
        if record["split"] == "val":
            with np.load(PILOT / "data" / (name + ".npz"), allow_pickle=False) as z:
                arrays.update(fixed_query_pos=((z["surface_points"] - p["center"]) / p["scale"]).astype(np.float32),
                              fixed_target=z["pressure_coeff"].reshape(-1, 1).astype(np.float32))
                np.testing.assert_allclose(conditions, z["conditions"], rtol=1e-6, atol=1e-6)
        assert all(np.isfinite(v).all() for v in arrays.values())
        output = ROOT / "data" / (name + ".npz")
        with output.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        return dict(id=name, split=record["split"], points=len(order), gen_params_raw=conditions.tolist())

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for row in pool.map(prepare, records):
            rows.append(row)
            if len(rows) % 64 == 0 or len(rows) == len(records):
                print(json.dumps(dict(phase="data_preparation", completed=len(rows), total=len(records),
                                      seconds=round(time.monotonic() - started, 1))), flush=True)
    gp = np.array([x["gen_params_raw"] for x in rows if x["split"] == "train"], np.float64)
    report = dict(records=rows, train_cars=420, validation_cars=16,
                  normalization=source["normalization"], normalization_fit_split="train",
                  gen_params_mean=gp.mean(0).tolist(), gen_params_std=np.maximum(gp.std(0), 1e-6).tolist(),
                  gen_params_names=["upstream_velocity_div40", "reference_area_div2.17"],
                  geometry_coordinates="Areal normalized coordinates; first 256 points are ordered anchors",
                  old_surface_cache=str(OLD / "cache_ready.json"), fixed_validation_queries=2048)
    (ROOT / "data_ready.json").write_text(json.dumps(report, indent=2) + "\n")
    print("DRIVAERML_READY", flush=True)


if __name__ == "__main__":
    main()
