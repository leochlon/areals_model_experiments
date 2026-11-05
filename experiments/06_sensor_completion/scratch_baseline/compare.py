import argparse
import csv
import json
from pathlib import Path

import numpy as np

import field_benchmark as fb

ROOT = Path(__file__).resolve().parent


def compare(split):
    bench = ROOT / "bench"
    config = fb.bench_config(bench)
    meta, indices = fb.read_observations(bench,split)
    methods = ["geometry","measurements","conditioned","areal_scratch"]
    for name in ["gabi_40ep","gabi"]:
        if (bench / "predictions" / name / f"{split}.npz").exists():
            methods.append(name)
    fb.cmd_score(argparse.Namespace(bench=str(bench),split=split,final=split=="test",
        methods=methods,reference="measurements",boot=4000))
    rows = list(csv.DictReader((bench / "scores" / f"{split}_per_problem.csv").open()))
    records = {r["id"]:r for r in fb.records_of(config,split)}
    families = sorted({r["family"] for r in records.values()})
    assert len(families) == 16 and len(records) == 144 and len(meta["entries"]) == 1152
    new_meta = json.loads((bench / "predictions/areal_scratch" / f"{split}.json").read_text())["method"]["benchmark"]
    old_meta = json.loads((bench / "predictions/measurements" / f"{split}.json").read_text())["method"]["benchmark"]
    assert new_meta["scratch"] is True and old_meta["scratch"] is False
    for key in ["lr","new_lr","epochs","seed","queries","normalise","floor","mix",
        "visible_weight","train_cars","protocol","mean","std","input_role"]:
        assert new_meta[key] == old_meta[key],(key,new_meta[key],old_meta[key])
    bootstrap = np.random.default_rng(20261023).integers(0,len(families),(4000,len(families)))
    settings = []
    for setting in dict.fromkeys(row["setting"] for row in rows):
        matrix = np.array([[np.mean([float(row["nrmse"]) for row in rows if row["setting"]==setting
            and row["method"]==method and records[row["car"]]["family"]==family])
            for method in methods] for family in families])
        assert np.isfinite(matrix).all()
        estimates = matrix[bootstrap].mean(1)
        item = dict(setting=setting,nrmse={},scratch_minus={})
        for i,method in enumerate(methods):
            item["nrmse"][method] = dict(mean=float(matrix[:,i].mean()),
                low=float(np.quantile(estimates[:,i],.025)),high=float(np.quantile(estimates[:,i],.975)))
            if method == "areal_scratch":
                continue
            differences = matrix[:,3]-matrix[:,i]
            draws = differences[bootstrap].mean(1)
            item["scratch_minus"][method] = dict(mean=float(differences.mean()),
                low=float(np.quantile(draws,.025)),high=float(np.quantile(draws,.975)),
                geometries_scratch_better=int((differences<0).sum()))
        settings.append(item)
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    fb.save_json(output / f"{split}_grouped.json",dict(split=split,settings=settings,
        methods=methods,independent_geometry_groups=16,conditions_per_geometry=9))
    # Independently verify every saved scratch prediction's area-weighted score.
    lookup = {(row["car"],row["setting"],int(row["repeat"])):row for row in rows if row["method"]=="areal_scratch"}
    maximum = 0.0
    with np.load(bench / "predictions/areal_scratch" / f"{split}.npz") as predictions:
        assert set(predictions.files) == {entry["key"] for entry in meta["entries"]}
        for car,entries in fb.by_car(meta["entries"]).items():
            field = fb.load_field(config["fields"],car)
            w,y = field.weights,field.values
            centre = (w*y).sum()/w.sum()
            spread = np.sqrt((w*(y-centre)**2).sum()/w.sum())
            for entry in entries:
                hidden = np.ones(len(y),dtype=bool)
                hidden[indices[entry["key"]]] = False
                residual = predictions[entry["key"]].astype(np.float64)-y
                value = np.sqrt((w[hidden]*residual[hidden]**2).sum()/w[hidden].sum())/spread
                expected = float(lookup[(car,entry["setting"],int(entry["repeat"]))]["nrmse"])
                maximum = max(maximum,abs(value-expected))
    assert maximum < 1e-10,maximum
    fb.save_json(output / f"{split}_verification.json",dict(status="PASS",predictions_checked=1152,
        maximum_score_difference=maximum))
    print("GROUPED",split,json.dumps(settings),flush=True)


if __name__ == "__main__":
    for split in ["val","test"]:
        compare(split)
