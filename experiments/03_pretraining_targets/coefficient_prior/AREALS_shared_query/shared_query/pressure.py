"""Native AREALS pressure transfer, independent LR search, and test scoring of the selected runs.

Only ``tune`` reads training and validation labels. ``score`` evaluates the two validation-selected
checkpoints of a completed selection on the test cars.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from areal_tokens.config import ExperimentConfig
from areal_tokens.field import FieldEncoder
from areal_tokens.model import GeometryAutoencoder


VERSION = "shared_query_pressure_v1"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def deterministic(device):
    # Must precede CUDA context initialization; use math attention on every run.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


class Cars:
    """Lazy native-cache reader. Geometry-only methods never open pressure files."""

    def __init__(self, index, field_dir, geometry_dir, allowed_splits=("train", "val")):
        self.index = Path(index).resolve()
        self.field_dir, self.geometry_dir = Path(field_dir).resolve(), Path(geometry_dir).resolve()
        rows = read_json(self.index)["records"]
        if not rows or len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Empty index or repeated car identity")
        self.rows = {r["id"]: r for r in rows}
        if any(r["split"] not in {"train", "val", "test"} for r in rows):
            raise ValueError("Invalid car split")
        if any(Path(i).name != i for i in self.rows):
            raise ValueError("Car IDs must be safe filenames")
        self.allowed_splits = set(allowed_splits)
        self.cache = OrderedDict()

    def ids(self, split):
        return sorted(i for i, r in self.rows.items() if r["split"] == split)

    def paths(self, identity):
        if identity not in self.rows:
            raise KeyError(identity)
        return self.field_dir / f"{identity}.npz", self.geometry_dir / f"{identity}.npz"

    def geometry(self, identity):
        _, path = self.paths(identity)
        with np.load(path, allow_pickle=False) as z:
            g = {k[2:]: z[k].copy() for k in z.files if k.startswith("a_")}
        required = {"points", "normals", "point_weights", "anchors", "anchor_normals", "anchor_weights", "center", "scale", "area"}
        if not required.issubset(g):
            raise ValueError(f"{identity}: incomplete native geometry cache: {required - set(g)}")
        for k, x in g.items():
            if not np.isfinite(x).all():
                raise ValueError(f"{identity}: nonfinite geometry {k}")
        for pos, norm, weights in [("points", "normals", "point_weights"), ("anchors", "anchor_normals", "anchor_weights")]:
            if g[pos].ndim != 2 or g[pos].shape[1] != 3 or g[norm].shape != g[pos].shape:
                raise ValueError(f"{identity}: invalid {pos}/{norm} shape")
            if g[weights].shape != (len(g[pos]),) or np.any(g[weights] < 0) or not g[weights].sum() > 0:
                raise ValueError(f"{identity}: invalid {weights}")
        if g["center"].shape != (3,) or g["scale"].shape != (1,) or not g["scale"][0] > 0:
            raise ValueError(f"{identity}: invalid geometry coordinate normalization")
        return g

    def get(self, identity):
        if self.rows[identity]["split"] not in self.allowed_splits:
            raise ValueError(f"Label access forbidden for split {self.rows[identity]['split']}")
        if identity not in self.cache:
            path, _ = self.paths(identity)
            with np.load(path, allow_pickle=False) as z:
                field = {k: np.asarray(z[k], dtype=np.float64).copy() for k in ("points", "values", "weights")}
            field["values"] = field["values"].reshape(-1)
            field["weights"] = field["weights"].reshape(-1)
            n = len(field["values"])
            if n == 0 or field["points"].shape != (n, 3) or field["weights"].shape != (n,):
                raise ValueError(f"{identity}: invalid pressure arrays")
            if any(not np.isfinite(x).all() for x in field.values()) or np.any(field["weights"] <= 0):
                raise ValueError(f"{identity}: nonfinite pressure or nonpositive area")
            field["weights"] /= field["weights"].sum()
            self.cache[identity] = dict(id=identity, field=field, geometry=self.geometry(identity))
            if len(self.cache) > 48:
                self.cache.popitem(last=False)
        self.cache.move_to_end(identity)
        return self.cache[identity]



def normalization(cars, ids):
    if not ids or any(cars.rows[i]["split"] != "train" for i in ids):
        raise ValueError("Normalization requires training cars only")
    means, second = [], []
    for i in ids:
        f = cars.get(i)["field"]
        means.append(float(f["weights"] @ f["values"]))
        second.append(float(f["weights"] @ f["values"] ** 2))
    mean = float(np.mean(means))
    std = math.sqrt(max(0., float(np.mean(second)) - mean ** 2))
    if not std > 0:
        raise ValueError("Training pressure has zero variance")
    return dict(mean=mean, std=std, training_ids=list(ids))


def make_model(config, initialization, seed, native_state=None):
    if initialization not in {"pretrained", "scratch"}:
        raise ValueError(initialization)
    torch.manual_seed(seed)
    base = GeometryAutoencoder(ExperimentConfig.from_dict(config).model)
    if initialization == "pretrained":
        if native_state is None:
            raise ValueError("Pretrained initialization requires native model state")
        base.load_state_dict(native_state, strict=True)
    # Both branches consume the same constructor RNG before creating the head.
    return FieldEncoder(base)


def encode(model, geometry, device):
    batch = {k: torch.as_tensor(v, dtype=torch.float32, device=device)[None] for k, v in geometry.items()}
    batch["complete"] = torch.ones(1, 1, device=device)
    zeros = torch.zeros(1, len(geometry["points"]), 1, device=device)
    anchors = torch.zeros(1, len(geometry["anchors"]), 1, device=device)
    return model.encode(batch, zeros, zeros, anchors, anchors)


@torch.no_grad()
def evaluate(model, cars, ids, stats, device, chunk=2048):
    model.eval()
    rows = []
    for identity in ids:
        car = cars.get(identity)
        f = car["field"]
        if device.type == "cuda": torch.cuda.synchronize(device)
        began = time.monotonic()
        latent = encode(model, car["geometry"], device)
        pred = []
        for start in range(0, len(f["values"]), chunk):
            q = torch.as_tensor(f["points"][start:start + chunk], dtype=torch.float32, device=device)[None]
            pred.append(model.predict(latent, q)[0].cpu().numpy())
        if device.type == "cuda": torch.cuda.synchronize(device)
        elapsed = time.monotonic() - began
        pred = np.concatenate(pred).astype(np.float64) * stats["std"] + stats["mean"]
        w, y = f["weights"], f["values"]
        rmse = math.sqrt(float(w @ (pred - y) ** 2))
        spread = math.sqrt(float(w @ (y - w @ y) ** 2))
        if not spread > 0 or not np.isfinite(pred).all():
            raise ValueError(f"{identity}: undefined nRMSE or nonfinite prediction")
        rows.append(dict(id=identity, nrmse=rmse / spread, rmse=rmse, reference_std=spread,
                         queries=len(y), seconds=elapsed))
    if not rows: raise ValueError("No evaluation cars")
    return dict(mean_nrmse=float(np.mean([r["nrmse"] for r in rows])), cars=rows)


def save_state(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def train_candidate(protocol, initialization, lr, out, cars, native_state, *, resume=False, stop_after=None):
    """Fixed planned budget; stop_after permits exact-resume qualification only."""
    device = torch.device(protocol["device"])
    deterministic(device)
    model = make_model(protocol["native_config"], initialization, protocol["seed"], native_state).to(device)
    config = dict(protocol, initialization=initialization, encoder_lr=float(lr))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists() and read_json(out / "config.json") != config:
        raise ValueError("Pressure run configuration changed")
    optimizer = torch.optim.AdamW([
        dict(params=model.base.parameters(), lr=lr),
        dict(params=model.head.parameters(), lr=protocol["new_lr"]),
    ], weight_decay=0.)
    start, best = 0, float("inf")
    if (out / "last.pt").exists():
        if not resume: raise ValueError(f"Existing run requires --resume: {out}")
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        if state["config"] != config: raise ValueError("Resume protocol changed")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng_cpu"])
        if device.type == "cuda": torch.cuda.set_rng_state(state["rng_cuda"], device)
        start, best = state["step"], state["best_validation"]
        # Drop log rows written after the last durable checkpoint (e.g. preemption).
        for name in ("training.jsonl", "validation.jsonl"):
            path = out / name
            if path.exists():
                kept = [line for line in path.read_text().splitlines()
                        if line.strip() and json.loads(line)["step"] <= start]
                path.write_text("\n".join(kept) + ("\n" if kept else ""))
    elif resume and (out / "training.jsonl").exists():
        raise ValueError("No resumable checkpoint exists")
    write_json(out / "config.json", config)
    end = min(protocol["steps"], stop_after) if stop_after is not None else protocol["steps"]
    stats = protocol["normalization"]
    train_ids, val_ids = protocol["training_ids"], protocol["validation_ids"]
    with (out / "training.jsonl").open("a" if start else "w", buffering=1) as log:
        for step in range(start + 1, end + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            chosen = np.random.default_rng([protocol["seed"], step, 811]).choice(train_ids, protocol["batch"], replace=True)
            losses = []
            for slot, identity in enumerate(chosen):
                car = cars.get(str(identity)); f = car["field"]
                selected = np.random.default_rng([protocol["seed"], step, slot, 912]).choice(
                    len(f["values"]), protocol["queries"], replace=True, p=f["weights"])
                q = torch.as_tensor(f["points"][selected], device=device, dtype=torch.float32)[None]
                y = torch.as_tensor((f["values"][selected] - stats["mean"]) / stats["std"], device=device, dtype=torch.float32)
                pred = model.predict(encode(model, car["geometry"], device), q)[0]
                loss = (pred - y).square().mean()
                if not torch.isfinite(loss): raise FloatingPointError(f"Nonfinite pressure loss at {step}")
                (loss / protocol["batch"]).backward()
                losses.append(float(loss.detach()))
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            log.write(json.dumps(dict(step=step, loss=float(np.mean(losses)), gradient_norm=float(grad))) + "\n")
            scheduled = step % protocol["validate_every"] == 0 or step == protocol["steps"]
            if scheduled:
                report = evaluate(model, cars, val_ids, stats, device, protocol["eval_chunk"])
                report["step"] = step
                with (out / "validation.jsonl").open("a") as f: f.write(json.dumps(report) + "\n")
                improved = report["mean_nrmse"] < best
                if improved:
                    best = report["mean_nrmse"]
                    write_json(out / "best_validation.json", report)
            else:
                improved = False
            # A stop between scheduled validations does not change selection/RNG.
            if scheduled or step == end:
                state = dict(version=VERSION, config=config, model=model.state_dict(), optimizer=optimizer.state_dict(),
                             step=step, best_validation=best, rng_cpu=torch.get_rng_state(),
                             rng_cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
                if improved: save_state(out / "best.pt", state)
                save_state(out / "last.pt", state)
            if scheduled: print(initialization, lr, step, report["mean_nrmse"], flush=True)
    if end == protocol["steps"] and start <= end:
        write_json(out / "complete.json", dict(step=end, validation=best))
    return out


def _build_protocol(args, cars, checkpoint):
    if checkpoint.get("version") != 1 or "config" not in checkpoint or "model" not in checkpoint:
        raise ValueError("Expected exported native geometry checkpoint")
    config = ExperimentConfig.from_dict(checkpoint["config"])
    subsets = read_json(args.subsets)
    ids = subsets.get(f"{args.seed}/{args.budget}")
    if not isinstance(ids, list) or len(ids) != args.budget or len(set(ids)) != len(ids):
        raise ValueError("Requested subset missing, duplicated or wrong budget")
    if any(i not in cars.rows or cars.rows[i]["split"] != "train" for i in ids):
        raise ValueError("Training subset crosses held-out split")
    val = cars.ids("val")
    if not val or not cars.ids("test"): raise ValueError("Validation and test car splits are required")
    stats = normalization(cars, ids)
    return dict(version=VERSION, native_config=config.to_dict(), seed=args.seed, budget=args.budget,
                steps=args.steps, batch=args.batch, queries=args.queries, new_lr=args.new_lr,
                validate_every=args.validate_every, eval_chunk=args.eval_chunk, device=str(torch.device(args.device)),
                rates=list(args.rates), training_ids=ids, validation_ids=val, normalization=stats,
                geometry_checkpoint=str(Path(args.checkpoint).resolve()), complete_geometry=True, sensors=0,
                precision="fp32", optimizer="AdamW", weight_decay=0., schedule="constant",
                inputs=dict(windsor_index=str(Path(args.windsor_index).resolve()),
                            field_dir=str(Path(args.field_dir).resolve()), geometry_dir=str(Path(args.geometry_dir).resolve()),
                            subsets=str(Path(args.subsets).resolve())))


def tune(args):
    out = Path(args.out).resolve()
    if (out / "selection.json").exists():
        raise ValueError("Selection already exists; use score, or a new output directory")
    cars = Cars(args.windsor_index, args.field_dir, args.geometry_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    protocol = _build_protocol(args, cars, checkpoint)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists() and read_json(out / "protocol.json") != protocol:
        raise ValueError("Tuning protocol or data changed")
    write_json(out / "protocol.json", protocol)
    candidates = []
    for initialization in ("pretrained", "scratch"):
        for index, rate in enumerate(args.rates):
            path = out / initialization / f"lr{index}"
            train_candidate(protocol, initialization, rate, path, cars, checkpoint["model"],
                            resume=args.resume, stop_after=args.stop_after)
            if not (path / "complete.json").exists(): continue
            complete = read_json(path / "complete.json")
            candidates.append(dict(initialization=initialization, lr=rate, validation=complete["validation"],
                                   checkpoint=str((path / "best.pt").relative_to(out))))
    if len(candidates) != 2 * len(args.rates):
        print("Stopped before planned budget; selection and test access remain unavailable.")
        return
    winners = {i: min((c for c in candidates if c["initialization"] == i), key=lambda c: (c["validation"], c["lr"]))
               for i in ("pretrained", "scratch")}
    selection = dict(version=VERSION, rule="minimum full-validation mean per-car area/std nRMSE; tie lower LR",
                     protocol=protocol, candidates=candidates, winners=winners)
    write_json(out / "selection.json", selection)
    print(json.dumps({i: dict(lr=w["lr"], validation=w["validation"]) for i, w in winners.items()}, indent=2))


def load_selection(path):
    path = Path(path).resolve()
    if path.name != "selection.json" or not path.is_file():
        raise ValueError("A completed selection.json is required")
    selection = read_json(path)
    protocol = selection["protocol"]
    for initialization in ("pretrained", "scratch"):
        group = [c for c in selection["candidates"] if c["initialization"] == initialization]
        if len(group) != len(protocol["rates"]): raise ValueError("Incomplete tuning search")
        winner = min(group, key=lambda c: (c["validation"], c["lr"]))
        if selection["winners"][initialization] != winner: raise ValueError("Winner is not validation selected")
        for c in group:
            cp = (path.parent / c["checkpoint"]).resolve()
            if not cp.is_relative_to(path.parent): raise ValueError("Candidate path escapes run")
            complete = read_json(cp.parent / "complete.json")
            best = read_json(cp.parent / "best_validation.json")
            if complete["step"] != protocol["steps"] or complete["validation"] != c["validation"] or best["mean_nrmse"] != c["validation"]:
                raise ValueError("Incomplete or inconsistent candidate")
    return path, selection


def score(args):
    path, selection = load_selection(args.selection)
    protocol = selection["protocol"]; inp = protocol["inputs"]
    cars = Cars(inp["windsor_index"], inp["field_dir"], inp["geometry_dir"], allowed_splits=("train", "val", "test"))
    test_ids = cars.ids("test")
    result_path = path.parent / "test_metrics.json"
    if result_path.exists(): raise ValueError("Test scores already exist; preserving prior result")
    device = torch.device(args.device); deterministic(device)
    metrics = {}
    for initialization, winner in selection["winners"].items():
        state = torch.load(path.parent / winner["checkpoint"], map_location="cpu", weights_only=False)
        if state["version"] != VERSION: raise ValueError("Wrong pressure checkpoint version")
        config = dict(protocol, initialization=initialization, encoder_lr=winner["lr"])
        if state["config"] != config: raise ValueError("Selected checkpoint protocol differs")
        model = make_model(protocol["native_config"], "scratch", protocol["seed"]).to(device)
        model.load_state_dict(state["model"], strict=True)
        metrics[initialization] = evaluate(model, cars, test_ids, protocol["normalization"], device, args.chunk)
    write_json(result_path, dict(version=VERSION, selection=str(path), test_cars=test_ids,
                                 metrics=metrics, relative_error_change=metrics["pretrained"]["mean_nrmse"] /
                                 metrics["scratch"]["mean_nrmse"] - 1))
    print(json.dumps({k: v["mean_nrmse"] for k, v in metrics.items()}, indent=2))


@torch.no_grad()
def predict(args):
    path, selection = load_selection(args.selection)
    protocol = selection["protocol"]; inp = protocol["inputs"]
    winner = selection["winners"][args.initialization]
    device = torch.device(args.device); deterministic(device)
    state = torch.load(path.parent / winner["checkpoint"], map_location="cpu", weights_only=False)
    model = make_model(protocol["native_config"], "scratch", protocol["seed"]).to(device)
    model.load_state_dict(state["model"], strict=True); model.eval()
    cars = Cars(inp["windsor_index"], inp["field_dir"], inp["geometry_dir"])
    xyz = np.load(args.queries, allow_pickle=False)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("Queries must be finite world-coordinate [N,3] .npy")
    latent = encode(model, cars.geometry(args.car), device)
    values = []
    for start in range(0, len(xyz), args.chunk):
        q = torch.as_tensor(xyz[start:start + args.chunk], dtype=torch.float32, device=device)[None]
        values.append(model.predict(latent, q)[0].cpu().numpy())
    stats = protocol["normalization"]
    prediction = np.concatenate(values) * stats["std"] + stats["mean"] if values else np.empty(0, np.float32)
    out = Path(args.out)
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, points=xyz, pressure=prediction)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("tune", help="Train both initializations and select LR on validation only")
    for flag in ["checkpoint", "windsor-index", "field-dir", "geometry-dir", "subsets", "out"]:
        t.add_argument("--" + flag, required=True)
    t.add_argument("--budget", type=int, required=True); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda"); t.add_argument("--steps", type=int, default=8000)
    t.add_argument("--batch", type=int, default=8); t.add_argument("--queries", type=int, default=2048)
    t.add_argument("--validate-every", type=int, default=250); t.add_argument("--eval-chunk", type=int, default=2048)
    t.add_argument("--rates", type=float, nargs="+", default=[2e-5, 2e-4, 1e-3]); t.add_argument("--new-lr", type=float, default=2e-4)
    t.add_argument("--resume", action="store_true"); t.add_argument("--stop-after", type=int)
    t.set_defaults(func=tune)
    s = sub.add_parser("score", help="Score the two validation-selected checkpoints on the test cars")
    s.add_argument("--selection", required=True); s.add_argument("--device", default="cuda")
    s.add_argument("--chunk", type=int, default=2048); s.set_defaults(func=score)
    s = sub.add_parser("predict", help="Predict pressure from geometry and world query coordinates only")
    for flag in ["selection", "car", "queries", "out"]: s.add_argument("--" + flag, required=True)
    s.add_argument("--initialization", choices=["pretrained", "scratch"], default="pretrained")
    s.add_argument("--device", default="cuda"); s.add_argument("--chunk", type=int, default=2048)
    s.set_defaults(func=predict)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ["budget", "steps", "batch", "queries", "validate_every", "eval_chunk", "chunk", "stop_after"]:
        value = getattr(args, name, None)
        if isinstance(value, int) and value <= 0: raise ValueError(f"{name} must be positive")
    if hasattr(args, "rates"):
        if len(set(args.rates)) != len(args.rates) or any(not math.isfinite(x) or x <= 0 for x in args.rates + [args.new_lr]):
            raise ValueError("Rates must be distinct positive finite numbers")
    args.func(args)
